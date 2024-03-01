import numpy as np
from scipy.linalg import eig

def eig_tensor(S):
    S11 = S['S11']
    S12 = S['S12']
    S13 = S['S13']
    S22 = S['S22']
    S23 = S['S23']
    S33 = S['S33']

    lambda1, vec1 = eig([[S11, S12, S13],
                         [S12, S22, S23],
                         [S13, S23, S33]])

    idx = np.argsort(lambda1)
    lambda1 = lambda1[idx]
    vec1 = vec1[:, idx]

    beta = 1 - lambda1[:, 0] / lambda1[:, 2]
    beta[np.isnan(beta)] = 0

    if len(lambda1.shape) == 1:
        return beta
    elif lambda1.shape[1] == 1:
        return lambda1, vec1[:, :, 0], beta
    else:
        lambda1 = lambda1.T
        vec1 = vec1.swapaxes(1, 2)
        lambda1 = lambda1.squeeze()
        vec1 = vec1.squeeze()

        return lambda1, vec1[:, :, 0], vec1[:, :, 1], vec1[:, :, 2], beta
