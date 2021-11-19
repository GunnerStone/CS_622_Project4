### Principle Component Analysis in Python
+ Implemented the following core components of the PCA algorithm from scratch as helper functions in python

Function Name | Description
---|---|
`compute_Z(X, centering=True, scaling=False)`| calculates normalized feature vector
`compute_covariance_matrix(Z)`| finds the covariance matrix of a normalized vector `Z`
`find_pcs(COV)`| Finds principle components of a covariance matrix
`project_data(Z, PCS, L, k,  var)`| Projects  a vector `Z` onto principle components `PCS` using eigenvectors `L` for top `K` eigenvectors or onto N eigenvectors that describe `var` amount of the data's variance


+ Compressed Images of faces using PCA
	+ Projected original images onto top K principle components
	+ Reconstructed a compressed version of the original face by using the projection and the original set of principle components


| Original Image | K = 800 | K = 400 | K = 100 | K=20 |
:-------------------------:|:-------------------------:
|test|test|test|test|test|

![](https://github.com/GunnerStone/CS_622_Project4/blob/main/README_imgs/original.png)| ![](https://github.com/GunnerStone/CS_622_Project4/blob/main/README_imgs/K800.png)| ![](https://github.com/GunnerStone/CS_622_Project4/blob/main/README_imgs/K400.png) | ![](https://github.com/GunnerStone/CS_622_Project4/blob/main/README_imgs/K100.png) | ![](https://github.com/GunnerStone/CS_622_Project4/blob/main/README_imgs/K20.png) |
