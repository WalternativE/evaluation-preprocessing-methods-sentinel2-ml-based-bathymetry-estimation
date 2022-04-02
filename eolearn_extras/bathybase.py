import numpy as np
from eolearn.core import EOTask, FeatureType


class AppendBathyTimelessDataMask(EOTask):
    def __init__(
        self, src_feature, dst_feature_name="bathy_data_mask", band_index=0
    ):
        self.src_feature = src_feature
        self.dst_feature_name = dst_feature_name
        self.band_index = band_index

    def execute(self, eopatch):
        if len(eopatch[self.src_feature].shape) > 3:
            raise ValueError(
                "Feature {} is not timeless".format(self.src_feature)
            )

        bathy_data = np.copy(eopatch[self.src_feature][:, :, self.band_index])
        bathy_data[bathy_data >= 0] = 0
        bathy_data[bathy_data < 0] = 1
        bathy_data = bathy_data.astype(np.uint8)

        result_eopatch = eopatch.copy()
        result_eopatch[
            FeatureType.MASK_TIMELESS, self.dst_feature_name
        ] = np.atleast_3d(bathy_data)

        return result_eopatch
