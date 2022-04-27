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

        bin_values = np.unique(result_eop[self.train_test_maks_feature])

        train_marker = 1
        if len(bin_values) == 2:
            test_marker = 2
        elif len(bin_values) == 3:
            validation_marker = 2
            test_marker = 3
        else:
            raise ValueError("Only supporting train/test or train/validation/test splits.")

        # multiplication by one to do an implicit type cast to a numerical value
        result_eop[(FeatureType.MASK_TIMELESS, 'train_split_valid')] = ((result_eop[self.train_test_maks_feature] == train_marker ) & (result_eop[self.valid_data_mask_feature] == 1)) * 1
        if len(bin_values) == 3:
            result_eop[(FeatureType.MASK_TIMELESS, 'validation_split_valid')] = ((result_eop[self.train_test_maks_feature] == validation_marker) & (result_eop[self.valid_data_mask_feature] == 1)) * 1
        # multiplication by one to do an implicit type cast to a numerical value
        result_eop[(FeatureType.MASK_TIMELESS, 'test_split_valid')] = ((result_eop[self.train_test_maks_feature] == test_marker) & (result_eop[self.valid_data_mask_feature] == 1)) * 1

        traincount = np.sum(result_eop[(FeatureType.MASK_TIMELESS, 'train_split_valid')] == 1)
        if len(bin_values) == 3:
            validationcount = np.sum(result_eop[(FeatureType.MASK_TIMELESS, 'validation_split_valid')] == 1)
        else:
            validationcount = 0
        testcount = np.sum(result_eop[(FeatureType.MASK_TIMELESS, 'test_split_valid')] == 1)

        result_eop.meta_info['train_count'] = traincount
        result_eop.meta_info['test_count'] = testcount
        result_eop.meta_info['validation_count'] = validationcount
        result_eop.meta_info['train_perc'] = traincount / (traincount + testcount + validationcount)
        result_eop.meta_info['test_perc'] = testcount / (traincount + testcount + validationcount)
        result_eop.meta_info['validation_perc'] = validationcount / (traincount + testcount + validationcount)

        return result_eop
