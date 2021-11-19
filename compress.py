import numpy as np
import pca as pca
"""
Description:
    This function will take the flattened image data (DATA) as input, along with k, the number of principal
    components to use. This function will use PCA to find the principal components of the face images. It will
    return the compressed data. Xcompressed = Z∗UT . Where Z∗is the projected data and UT is the transpose
    of the principal components. This function will then output the compressed images into a directory named
    Output. You should use the os package to make sure that the directory is present and to create it if it does
    not exist. NOTE: Images have values from 0 to 255, so you will want to rescale them before saving them.
    You will also want to use the cmap=‘gray’ option in pyplot.imsave to save them as grayscale images
"""
def compress_images(DATA,k):

    # find covariance matrix Z (size is MxM where M is the number of images)
    Z = pca.compute_Z(DATA, centering= True, scaling= True)

    # compute the covariance matrix of Z
    COV = pca.compute_covariance_matrix(Z)

    # find the principal components (eigenvectors) of s (call this PCS)
    L, PCS = pca.find_pcs(COV)

    # project the data into the new space
    Z_star = pca.project_data(Z, PCS, L, k, 0)\
 
    # get the compressed images with top k eigenvectors
    # get top k eigenvectors from PCS, they are stored as columns so only take first k columns
    top_k_eigenvectors = PCS[:,:k]

    # get the transpose of the top k eigenvectors
    top_k_eigenvectors_transpose = top_k_eigenvectors.T

    # reconstruct the orignial data using the following formula to get the compressed data
    X_star = np.dot(Z_star, top_k_eigenvectors_transpose)

    # scale all pixel values to be between 0 and 255
    X_star = X_star - np.min(X_star)
    X_star = X_star / (np.max(X_star)-np.min(X_star))
    X_star = X_star * 255

    # create Output directory if it doesnt exist
    import os
    if not os.path.exists("Output"):
        os.makedirs("Output")

    # X_star has size N x M where N is the number of pixels and M is the number of images
    # for every image, reshape the image into its original shape
    for i in range(len(X_star[1])):
        # original resolution for these images is 60 rows, 48 cols
        curr_img = X_star[:,i].reshape(60,48)

        # save the image
        import matplotlib.pyplot as plt
        plt.imsave("Output/image_" + str(i) + ".png", curr_img, cmap=plt.cm.gray, vmin=0, vmax=255)

    return

"""
Description:
    The function below takes the input directory as input, and outputs the DATA matrix. DATA will have one
    flattened image per column, so each column represents an image and one row represents the pixel values for
    every image at a particular location. This function will use pyplot.imread to load the images. Before you
    return DATA you will want to convert it to floating point. 
"""
def load_data(input_dir):
    # get the list of files
    import os
    import matplotlib.pyplot as plt
    files = os.listdir(input_dir)
    # load the images
    images = []
    for file in files:
        #store the image as a float
        image = plt.imread(input_dir + "/" + file)
        #flatten the image
        image = image.flatten()
        #append the image to the list
        images.append(image)
    # convert the list to a matrix
    images = np.array(images, dtype=float)
    # make each image a column
    images = images.T
    return images