import numpy as np
import rasterio as rio
from rasterio import MemoryFile
from rasterio.enums import Resampling
from eolearn.core import EOTask, EOPatch
from sentinelhub import BBox


class ReprojectRasterTask(EOTask):
    def __init__(
        self,
        feature,
        target_resolution=None,
        target_width=None,
        target_height=None,
        target_crs=None,
        driver="GTiff",
        resampling=Resampling.bilinear,
    ):
        if target_resolution is None and (
            target_width is None or target_height is None
        ):
            raise ValueError(
                "Either target_resolution or target_width and " +
                "target_height must be provided"
            )

        self.feature = feature
        self.target_crs = target_crs
        self.target_width = target_width
        self.target_height = target_height
        self.target_resolution = target_resolution
        self.driver = driver
        self.resampling = resampling

    def execute(self, eopatch: EOPatch):
        times = None
        # timeless features only have 3 dimensions
        if len(eopatch[self.feature].shape) == 3:
            height, width, channels = eopatch[self.feature].shape
        else:
            times, height, width, channels = eopatch[self.feature].shape

        dtype = eopatch[self.feature].dtype
        crs = rio.crs.CRS.from_epsg(eopatch.bbox.crs.epsg)
        transform = rio.transform.from_bounds(*eopatch.bbox, width, height)

        agreed_bbox = None
        single_frame = None
        frames = []
        repeats = 1 if times is None else times
        target_crs = self.target_crs if self.target_crs is not None else crs
        for i in range(repeats):
            with MemoryFile() as src_memfile:
                with src_memfile.open(
                    driver=self.driver,
                    height=height,
                    width=width,
                    count=channels,
                    dtype=dtype,
                    crs=crs,
                    transform=transform,
                ) as src:

                    for channel in range(channels):
                        if times is None:
                            src.write(
                                eopatch[self.feature][:, :, channel],
                                channel + 1,
                            )
                        else:
                            src.write(
                                eopatch[self.feature][i, :, :, channel],
                                channel + 1,
                            )

                    target_transform, target_width, target_height = (
                        rio.warp.calculate_default_transform(
                            src.crs,
                            target_crs,
                            src.width,
                            src.height,
                            *src.bounds,
                            dst_width=self.target_width,
                            dst_height=self.target_height
                        )
                        if self.target_resolution is None
                        else rio.warp.calculate_default_transform(
                            src.crs,
                            target_crs,
                            src.width,
                            src.height,
                            *src.bounds,
                            resolution=self.target_resolution
                        )
                    )
                    kwargs = src.meta.copy()
                    kwargs.update(
                        {
                            "crs": target_crs,
                            "transform": target_transform,
                            "width": target_width,
                            "height": target_height,
                        }
                    )

                    with MemoryFile() as dst_memfile:
                        with dst_memfile.open(**kwargs) as dst:
                            for i in range(1, src.count + 1):
                                rio.warp.reproject(
                                    source=rio.band(src, i),
                                    destination=rio.band(dst, i),
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=target_transform,
                                    dst_crs=target_crs,
                                    resampling=self.resampling,
                                )

                            if agreed_bbox is None:
                                agreed_bbox = BBox(
                                    dst.bounds, crs=target_crs.to_epsg()
                                )

                            if times is not None:
                                frames.append(np.moveaxis(dst.read(), 0, -1))
                            else:
                                single_frame = np.moveaxis(dst.read(), 0, -1)

        result_eopatch = eopatch.copy()
        result_eopatch[self.feature] = (
            np.stack(frames, axis=0) if times is not None else single_frame
        )
        result_eopatch.bbox = agreed_bbox

        return result_eopatch
