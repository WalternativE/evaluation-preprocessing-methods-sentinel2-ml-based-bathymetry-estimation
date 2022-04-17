from eolearn.core import EOTask, EOPatch, FeatureType
import numpy as np


class AddValidTrainTestMasks(EOTask):
    def __init__(self,
                 train_test_maks_feature,
                 valid_data_mask_feature):
        self.train_test_maks_feature = train_test_maks_feature
        self.valid_data_mask_feature = valid_data_mask_feature

    def execute(self, eopatch: EOPatch):
        result_eop = eopatch.copy()

        result_eop[(FeatureType.MASK_TIMELESS, 'train_split_valid')] = eopatch[self.train_test_maks_feature] & eopatch[self.valid_data_mask_feature]
        # multiplication by one to do an implicit type cast to a numerical value
        result_eop[(FeatureType.MASK_TIMELESS, 'test_split_valid')] = ((eopatch[self.train_test_maks_feature] == 2) & (eopatch[self.valid_data_mask_feature] == 1)) * 1

        traincount = np.sum(result_eop[(FeatureType.MASK_TIMELESS, 'train_split_valid')] == 1)
        testcount = np.sum(result_eop[(FeatureType.MASK_TIMELESS, 'test_split_valid')] == 1)

        result_eop.meta_info['train_count'] = traincount
        result_eop.meta_info['test_count'] = testcount
        result_eop.meta_info['train_perc'] = traincount / (traincount + testcount)
        result_eop.meta_info['test_perc'] = testcount / (traincount + testcount)

        return result_eop
