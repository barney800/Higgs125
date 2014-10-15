#!/usr/bin/python

# Testing manifold learning algorithms

import numpy as np

from scipy import sparse

import matplotlib.pyplot as plt

from sklearn import manifold, datasets
from sklearn.neighbors import NearestNeighbors

# Set number of points, number of neighbours to consider
n_pts = 100
n_nbrs = 2
# Set LLE regulator and reconstruction error threshold
LLE_reg = 0.01
rec_err = 0.1
# Set dimensions of input and output parameter spaces
d_in = 2
d_out = 1

# Define input data: random sample of y = x^2 on [-1, 1]
X = np.array( [ [x_val, x_val**2] for x_val in np.random.random(n_pts) ] )

# Set up plot and plot input data
fig = plt.figure(figsize=(15, 5))
plt.suptitle("Manifold Learning with %i points, %i neighbors"
             % (n_pts, n_nbrs), fontsize=14)
ax = fig.add_subplot(131)
ax.scatter(X[:, 0], X[:, 1])

Y_sklearn = manifold.LocallyLinearEmbedding(n_nbrs, d_out, eigen_solver='auto', method='standard', reg=LLE_reg).fit_transform(X)

# Find indices of nearest neighbours for all data points
nbr_list = NearestNeighbors(n_neighbors=n_nbrs+1, algorithm='auto').fit(X)
nbrs = nbr_list.kneighbors(X, return_distance=False)

# Find weighting matrix
# Uses algorithm found in http://www.cs.nyu.edu/~roweis/lle/papers/lleintro.pdf
W_rows = np.zeros( shape=(n_nbrs*n_pts) )
W_cols = np.zeros( shape=(n_nbrs*n_pts) )
W_data = np.zeros( shape=(n_nbrs*n_pts) )
for i in range(n_pts):
    W_rows[i*n_nbrs : (i+1)*n_nbrs] = [ i for nbr in range(n_nbrs) ]
    W_cols[i*n_nbrs : (i+1)*n_nbrs] = nbrs[i, 1:]
    C_jk = np.array( [ [ (X[i]-X[j]).dot(X[i]-X[k]) for k in nbrs[i, 1:] ] for j in nbrs[i, 1:] ] )
    C_jk += (LLE_reg/n_nbrs)*np.trace(C_jk)*np.identity(n_nbrs)
    w_ij = [ 1 / np.sum( C_jk[j] ) for j in range(n_nbrs) ]
    W_data[i*n_nbrs : (i+1)*n_nbrs] = w_ij / np.sum(w_ij)
W = sparse.coo_matrix( ( W_data, (W_rows, W_cols) ), shape=(n_pts, n_pts) )

# Identify poorly reconstructed points
X_rec = W.toarray().dot(X)
bad_pts = np.array( [ i for i in range(n_pts) if ( np.linalg.norm( X_rec[i]-X[i] )/np.linalg.norm( X[i] ) ) > rec_err ] )
ax = fig.add_subplot(132)
ax.scatter(X_rec[:, 0], X_rec[:, 1])
X_rec = np.delete(X_rec, bad_pts, 0)
ax = fig.add_subplot(133)
ax.scatter(X_rec[:, 0], X_rec[:, 1])

#plt.show()

# Find projected data points
# Uses algorithm found in http://www.cs.nyu.edu/~roweis/lle/papers/lleintro.pdf
M = ( sparse.identity(n_pts)-W.T-W+(W.T*W) ).toarray()
eig_vals, eig_vecs = np.linalg.eigh(M)
sort_perm = eig_vals.argsort()
eig_vals.sort()
eig_vecs = eig_vecs[sort_perm]
i_fnz = -1
for i in range(n_pts):
    if eig_vals[i] > 1e-10:
        i_fnz = i
        break
if i_fnz > -1:
    Y = eig_vecs[:, i_fnz]
else:
    print "No non-zero eigenvalues"
print i_fnz