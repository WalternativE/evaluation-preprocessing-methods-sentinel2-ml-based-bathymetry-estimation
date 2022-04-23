import numpy as np


# Code inspiration for Stumpf Log-Ratio SDB taken from
# https://github.com/balajiceg/NearShoreBathymetryPlugin/blob/master/process.py
def get_stumpf_log_ratio(eopatch, feature, data_mask, n=10000, eps_bias=0.0000000000001):
    # apply very small bias to not divide by zero
    blue_band = eopatch[feature][0,:,:,1][data_mask == 1] + eps_bias
    green_band = eopatch[feature][0,:,:,2][data_mask == 1] + eps_bias

    # in stumpf log-ratio this would correspond to z (or rel_z) before applying the constant factor c and the intercept m_0
    # we can get to these values by fitting a linear regression
    X = np.log(n * blue_band) / np.log(n * green_band)
    X = X.reshape(-1, 1)

    return X
