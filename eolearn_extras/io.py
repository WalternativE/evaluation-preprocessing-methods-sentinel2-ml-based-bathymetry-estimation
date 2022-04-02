import os
import glob
import datetime

import rioxarray as rx
import rasterio as rio
import numpy as np
from rasterio.enums import Resampling
from eolearn.core import EOPatch, FeatureType
from sentinelhub import BBox


sentinel_2_1lc_bands = {
    0: "B01",
    1: "B02",
    2: "B03",
    3: "B04",
    4: "B05",
    5: "B06",
    6: "B07",
    7: "B08",
    8: "B8A",
    9: "B09",
    10: "B10",
    11: "B11",
    12: "B12",
}


def extract_meta_from_path(sentinel_archive):
    meta_parts = os.path.basename(sentinel_archive).split(".")[0].split("_")
    mission = meta_parts[0]
    level = meta_parts[1][-3:]
    acq_time = datetime.datetime.strptime(meta_parts[2], "%Y%m%dT%H%M%S")

    return mission, level, acq_time


def get_products_by_level(sentinel_archives, level):
    products = [
        (extract_meta_from_path(archive)[2], archive)
        for archive in sentinel_archives
        if extract_meta_from_path(archive)[1] == level
    ]
    products.sort(key=lambda x: x[0])

    return products


def construct_eopatch_from_sentinel_archive(
    sentinel_archive,
    bbox: BBox = None,
    target_shape=None,
    target_resolution=10,
    resampling_method=Resampling.bilinear,
):
    eopatch = EOPatch()

    mission, level, acq_time = extract_meta_from_path(sentinel_archive)

    bands_pattern = f"{sentinel_archive}/**/*.jp2"
    band_paths = glob.glob(bands_pattern, recursive=True)
    bands_paths = [
        (os.path.basename(path).split(".")[0].split("_")[-1], path)
        for path in band_paths
    ]

    band_data_arrays = []
    agreed_bbox = None if bbox is None else bbox
    agreed_shape = None if target_shape is None else target_shape
    used_crs = None
    for bandname in sentinel_2_1lc_bands.values():
        res_bandpath = [path for (bn, path) in bands_paths if bn == bandname]
        if len(res_bandpath) > 0:
            band_da = rx.open_rasterio(res_bandpath[0], driver="JP2OpenJPEG")

            if used_crs is None and bbox is None:
                used_crs = band_da.rio.crs
            elif used_crs is None:
                used_crs = rio.crs.CRS.from_epsg(bbox.crs.epsg)

            if bbox is not None:
                if bbox.crs.epsg != band_da.rio.crs.to_epsg():
                    bbox = bbox.transform(band_da.rio.crs.to_epsg())
                band_da = band_da.rio.clip_box(*bbox)
            else:
                agreed_bbox = band_da.rio.bounds()

            (source_resolution, _) = band_da.rio.resolution()
            source_resolution = abs(source_resolution)
            # reprojecting and clipping can lead to an unequal amount of
            # pixels per band to circumvent this we can either supply a
            # shape as the parameter or fix a shape after going
            # through the first band
            if agreed_shape is None and source_resolution != target_resolution:
                band_da = band_da.rio.reproject(
                    used_crs,
                    resolution=(target_resolution, target_resolution),
                    resampling=resampling_method,
                )
            elif agreed_shape is not None and (
                source_resolution != target_resolution
                or band_da.rio.shape != agreed_shape
            ):
                band_da = band_da.rio.reproject(
                    used_crs, shape=agreed_shape, resampling=resampling_method
                )

            # all bands need to have the same shape
            # we fix this after working with the first band
            # if no shape is given as a parameter
            if agreed_shape is None:
                agreed_shape = band_da.rio.shape

            band_data_arrays.append(band_da.values[0])

    if len(band_data_arrays) < 1:
        raise ValueError("No bands found in sentinel archive")

    temporal_dim = 1  # only one timestamp
    height, width = band_data_arrays[0].shape
    channels = len(band_data_arrays)
    eopatch_shape = temporal_dim, height, width, channels

    band_data = np.stack(band_data_arrays, axis=-1).reshape(*eopatch_shape)

    eopatch.bbox = BBox(agreed_bbox, crs=used_crs.to_epsg())
    eopatch.timestamp = [acq_time]
    eopatch[FeatureType.DATA, f"{level}_data"] = band_data
    eopatch.meta_info["mission"] = mission

    return eopatch
