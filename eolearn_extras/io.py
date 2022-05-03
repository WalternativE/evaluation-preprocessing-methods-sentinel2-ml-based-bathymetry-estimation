from argparse import ArgumentError
import os
import glob
import datetime
from pydoc import resolve

import rioxarray as rx
import rasterio as rio
import numpy as np
from rasterio.enums import Resampling
from eolearn.core import EOPatch, FeatureType, EOTask
from sentinelhub import BBox


sentinel_2_bands = {
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

sentinel_2_l2a_bands = dict([
    (k, v) for (k, v) in sentinel_2_bands.items() if v != 'B10'
])


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


_available_sentinel_band_resolutions = ["10m", "20m", "60m"]


def resolve_l2a_band_paths_highres_first(available_paths, requested_bands):
    if len(requested_bands) < 1:
        raise ArgumentError("No bands requested")

    resolved_bands = []
    band_paths = []
    for requested_band in requested_bands:
        for available_path in available_paths:
            info_parts = (
                os.path.basename(available_path).split(".")[0].split("_")
            )
            if len(info_parts) < 4:
                # proper band info ususally has 4 parts encoded in the file
                # name - quality inspection data has less
                continue

            _, _, band_name, res = info_parts
            for band_resolution in _available_sentinel_band_resolutions:
                if band_resolution == res and band_name == requested_band:
                    band_paths.append((band_name, available_path))
                    resolved_bands.append(band_name)
                    break

            if requested_band in resolved_bands:
                break

    if len(band_paths) < len(requested_bands):
        raise ValueError(
            "Not all bands could be resolved. Bands found were: "
            + f"{[x[0] for x in band_paths]}"
        )

    return band_paths


def construct_eopatch_from_sentinel_archive(
    sentinel_archive,
    bbox: BBox = None,
    target_shape=None,
    target_resolution=10,
    resampling_method=Resampling.bilinear,
    requested_bands=None,
    digital_number_to_reflectance=False,
    dn_reflectance_factor=10000,
    log_callback=None,
):
    eopatch = EOPatch()

    mission, level, acq_time = extract_meta_from_path(sentinel_archive)

    bands_pattern = f"{sentinel_archive}/**/*.jp2"
    band_paths = glob.glob(bands_pattern, recursive=True)

    requested_bands = (
        (sentinel_2_bands if level == 'L1C' else sentinel_2_l2a_bands) if
        requested_bands is None else
        requested_bands
    )

    if level == "L1C":
        bands_paths = [
            (os.path.basename(path).split(".")[0].split("_")[-1], path)
            for path in band_paths
        ]
    elif level == "L2A":
        bands_paths = resolve_l2a_band_paths_highres_first(
            band_paths, requested_bands.values()
        )
    else:
        raise ValueError(f"Level {level} not supported")

    if log_callback:
        log_callback(f'Requested {len(requested_bands)} found {len(bands_paths)}.')

    if len(bands_paths) < len(requested_bands):
        raise ValueError(f'Requested {len(requested_bands)} but only found {len(bands_paths)}.')

    band_data_arrays = []
    agreed_bbox = None if bbox is None else bbox
    agreed_shape = None if target_shape is None else target_shape
    used_crs = None
    for bandname in requested_bands.values():
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

            band_data_values = band_da.values[0]
            if digital_number_to_reflectance:
                band_data_values = np.float32(band_data_values / dn_reflectance_factor)

            band_data_arrays.append(band_data_values)

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


class ReadSentinelArchiveTask(EOTask):
    def __init__(
        self,
        bbox: BBox = None,
        target_shape=None,
        target_resolution=10,
        resampling_method=Resampling.bilinear,
        requested_bands=None,
        digital_number_to_reflectance=False,
        dn_reflectance_factor=10000,
        log_callback=None,
    ):
        self.bbox = bbox
        self.target_shape = target_shape
        self.target_resolution = target_resolution
        self.resampling_method = resampling_method
        self.requested_bands = requested_bands
        self.digital_number_to_reflectance = digital_number_to_reflectance
        self.dn_reflectance_factor = dn_reflectance_factor
        self.log_callback = log_callback

    def execute(self, sentinel_archive_path):
        return construct_eopatch_from_sentinel_archive(
            sentinel_archive_path,
            self.bbox,
            self.target_shape,
            self.target_resolution,
            self.resampling_method,
            self.requested_bands,
            self.digital_number_to_reflectance,
            self.dn_reflectance_factor,
            self.log_callback,
        )
