import numpy as np

from scipy.stats import wishart
from pp_mix.precision.precmat import PrecMat


class Wishart(object):
    def __init__(self, df, psi):
        self.df = df
        self.psi = psi
        self.psi_inv = np.linalg.inv(psi)

    def sample_prior(self, n):
        return [PrecMat(wishart.rvs(self.df, self.psi)) for _ in range(n)]

    def sample_given_data(self, data, mean, *args):
        X = data - mean
        return PrecMat(wishart.rvs(
            self.df + data.shape[0],
            np.linalg.inv(self.psi_inv + np.dot(X.T, X))))