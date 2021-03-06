import pca
import numpy as np

# test PCA
X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
Z = pca.compute_Z(X)
COV = pca.compute_covariance_matrix(Z)
L, PCS = pca.find_pcs(COV)
# print out matrices
print("COV:", COV)
print("L:", L)
print("PCS:", PCS)
Z_star = pca.project_data(Z, PCS, L, 1, 0)

#test compression  using PCA
import compress

X = compress.load_data('Data/Train/')
compress.compress_images(X,1000)
