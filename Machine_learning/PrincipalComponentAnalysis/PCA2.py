import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples = 1000, factor = 0.3, noise = 0.05, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 0)

'''_, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8,4))
train_ax.scatter(X_train[:,0], X_train[:,1], c = y_train,)
train_ax.set_ylabel("Feature 1")
train_ax.set_xlabel("Feature 0")
train_ax.set_title("Train Datas")

test_ax.scatter(X_test[:,0], X_test[:,1], c = y_test)
test_ax.set_xlabel("Feature 0")
_ = test_ax.set_title("Test Datas")
plt.show()'''

from sklearn.decomposition import PCA, KernelPCA
pca = PCA(n_components=2)
kernel_pca = KernelPCA(
    n_components=None, kernel = "rbf", gamma =10, fit_inverse_transform = True, alpha=0.1
) #rbf = radial basis function

X_test_pca = pca.fit(X_train).transform(X_test)
X_test_kernel_pca = kernel_pca.fit(X_train).transform(X_test)

'''fig, (orig_data_ax, pca_proj_ax, kernel_pca_proj_ax) = plt.subplots(
    ncols = 3, figsize = (14, 4)
)

orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
orig_data_ax.set_ylabel("Feature 1")
orig_data_ax.set_xlabel("Feature 0")
orig_data_ax.set_title("Test Datas")

pca_proj_ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
pca_proj_ax.set_ylabel("key ingredient 1")
pca_proj_ax.set_xlabel("key ingredient 0")
pca_proj_ax.set_title("Test Datas \n Projection with PCA")

kernel_pca_proj_ax.scatter(X_test_kernel_pca[:, 0], X_test_kernel_pca[:, 1], c = y_test)
kernel_pca_proj_ax.set_ylabel("key ingredient 1")
kernel_pca_proj_ax.set_xlabel("key ingredient 0")
_ = kernel_pca_proj_ax.set_title("Test Datas \n Projection with Kernel PCA")
plt.show()'''

#reconstruction 