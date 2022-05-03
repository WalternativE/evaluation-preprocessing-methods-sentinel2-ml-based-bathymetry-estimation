from enum import IntEnum
import numpy as np
from eolearn.core import FeatureType


class SplitType(IntEnum):
    Train=1,
    Test=2,
    Validation=3,
    All=4


def get_X_y_for_split(eop,
    split_type: SplitType,
    data_feature,
    label_feature,
    data_mask_feature=(FeatureType.MASK_TIMELESS, 'bathy_data_mask'),
):
    # only supporting data with time dimension for now
    _, _, _, bands = eop[data_feature].shape
    traincount = eop.meta_info['train_count']
    testcount = eop.meta_info['test_count']
    validationcount = eop.meta_info['validation_count']

    bathy_pixels = np.sum(eop[data_mask_feature] == 1)

    if split_type == SplitType.Train:
        split_feature = (FeatureType.MASK_TIMELESS, 'train_split_valid')
        sample_count = traincount
    elif split_type == SplitType.Test:
        split_feature = (FeatureType.MASK_TIMELESS, 'test_split_valid')
        sample_count = testcount
    elif split_type == SplitType.Validation:
        split_feature = (FeatureType.MASK_TIMELESS, 'validation_split_valid')
        sample_count = validationcount
    elif split_type == SplitType.All:
        split_feature = data_mask_feature
        sample_count = bathy_pixels

    split_index = np.repeat(eop[split_feature], repeats=bands, axis=-1)
    X = np.reshape(eop[data_feature][0,:,:,:][split_index == 1], (sample_count, bands))
    y = eop[label_feature][eop[split_feature] == 1]

    return X, y


def create_sdb_estimation(
    eop,
    model,
    X_all,
    mask_feature=(FeatureType.MASK_TIMELESS, 'bathy_data_mask')
):
    y_hat_all = model.predict(X_all)
    sdb_estimation = np.zeros(eop[mask_feature].shape)
    sdb_estimation[eop[mask_feature] == 1] = y_hat_all

    return y_hat_all, sdb_estimation


def get_masked_map(eop, data_feature, mask_feature):
    masked_map = np.zeros(eop[mask_feature].shape)
    masked_index = eop[mask_feature] == 1
    masked_map[masked_index] = eop[data_feature][masked_index]

    return masked_map
