import os
import glob
import datetime

import numpy as np

from eolearn.core import (
    EOWorkflow,
    linearly_connect_tasks,
    OutputTask
)
from eolearn.io import ImportFromTiffTask

import rasterio as rio
import eolearn_extras as eolx


def get_acolite_band_tif_paths(folder, product_type='L2R', reflectance_type='rhos'):
    tif_pattern = f'{folder}/*_{product_type}_{reflectance_type}_*.tif'
    tifs = glob.glob(tif_pattern)
    return sorted(tifs, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))


def get_info_for_acolite_tif_path(acolite_tif_path):
    bn = os.path.basename(acolite_tif_path)
    parts = bn.split('.')[0].split('_')
    mission, instrument, year, month, day, hour, minute, second, _, product, reflectance_type, center_freq = parts

    return reflectance_type, center_freq, datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))


def get_eopatch_for_acolite_band_tif(band_tif_path, reference_bbox, feature):
    reflectance_type, center_freq, ts = get_info_for_acolite_tif_path(band_tif_path)

    feature_type, feature_name = feature
    feature = (feature_type, f'{feature_name}_{reflectance_type}_{center_freq}')

    import_acolite_band = ImportFromTiffTask(feature, band_tif_path)
    reproject_acolite_band = eolx.raster.ReprojectRasterTask(
        feature,
        target_crs=rio.crs.CRS.from_epsg(reference_bbox.crs.epsg),
        target_resolution=(10, 10)
    )
    clip_acolite_band = eolx.raster.ClipBoxTask(feature, target_bounds=reference_bbox)

    acolite_band_output_label = 'acolite_band_output'
    wf = EOWorkflow(
        linearly_connect_tasks(
            import_acolite_band,
            reproject_acolite_band,
            clip_acolite_band,
            OutputTask(acolite_band_output_label)
        )
    )

    acolite_band_patch = wf.execute().outputs[acolite_band_output_label]
    acolite_band_patch.timestamp = [ts]

    # TODO: think about a better fix for wrong atmospheric correction
    number_of_overcorrected_pixels = np.sum(acolite_band_patch[feature][acolite_band_patch[feature] < 0])
    if number_of_overcorrected_pixels > 0:
        _, f_name = feature
        print(f'Feature {f_name} had {number_of_overcorrected_pixels} overcorrected pixels. Setting to 0.')
        acolite_band_patch[feature][acolite_band_patch[feature] < 0] = 0

    return acolite_band_patch
