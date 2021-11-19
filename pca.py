import numpy as np

"""
Description:
    The below function will take the data matrix X, and boolean variables centering and scaling. X has one
    sample per row. Remember there are no labels in PCA. If centering is True, you will subtract the mean
    from each feature. If scaling is True, you will divide each feature by its standard deviation. This function
    returns the Z matrix (numpy array), which is the same size as X.
"""
def compute_Z(X, centering=True, scaling=False):
    # zero-center the data
    if centering:
        X = X - np.mean(X, axis=0)
    # scale the data
    if scaling:
        X = X / np.std(X, axis=0)
    return X


"""
Description:
    The below function will take the standardized data matrix Z and return the covariance matrix ZT Z=COV (a
    numpy array)
"""
def compute_covariance_matrix(Z):
    return np.dot(Z.T, Z)

"""
Description:
    The below function will take the covariance matrix COV and return the ordered (largest to smallest) principal
    components PCS (a numpy array where each column is an eigenvector) and corresponding eigenvalues L (a
    numpy array). 
"""
def find_pcs(COV):

    # find the eigenvalues and eigenvectors
    L, PCS = np.linalg.eigh(COV)

    # sort the eigenvalues and eigenvectors in descending order
    L, PCS = L[::-1], PCS[:,::-1]

    # return the principal components and eigenvalues
    return L, PCS




"""
Description:
    The below function will take the standardized data matrix Z, the principal components PCS, and correspond-
    ing eigenvalues L, as well as a k integer value and a var floating point value. k is the number of principal
    components you wish to maintain when projecting the data into the new space. 
--------------------------------------------------------------------------------------------------------------
Conditionals:
--------------------------------------------------------------------------------------------------------------
0 ≤k ≤D. 
    If k= 0, then we use the cumulative variance to determine the projection dimension. var is the desired cumulative variance
    explained by the projection. 

0 ≤v≤1. 
    If v = 0, then k is used instead. Assume they are never both 0 or
    both > 0. This function will return Z_star, the projected data.
"""
def project_data(Z, PCS, L, k,  var):
    """
    Project the data into the new space.
    """
    
    # verify that L and PCS are both sorted in descending order
    Lindxs = L.argsort()
    L = L[Lindxs[::-1]]
    PCS = PCS[Lindxs[::-1]]

    # if k is 0, then use the cumulative variance to determine the projection dimension
    if k == 0:
        # compute the cumulative variance explained by the projection
        cum_var = np.cumsum(L) / np.sum(L)

        # find the index of the first element in cum_var that is greater than or equal to var
        k = np.where(cum_var >= var)[0][0] + 1

        # only keep the first k columns of PCS
        PCS = PCS[:, :k]
        
    # if var is 0, then use k to determine the projection dimension
    else:
        # only keep the first k eigenvectors
        PCS = PCS[:, :k]

    Z_star = Z.dot(PCS)
    return Z_star


