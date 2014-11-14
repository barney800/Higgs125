#!/usr/bin/python

# Testing manifold learning algorithms

import numpy as np
from itertools import product
from scipy import sparse
from sklearn import manifold, datasets
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def ScanManifold(X_train, n_pts, ML_method='LLE', ML_kwargs={}, scan_method='uniform'):

    # Display training data info
    n_pts_train, d_high = X_train.shape
    d_low = d_high-1
    print "Training data: {0} points in {1}-dimensional parameter space".format(n_pts_train, d_high)

    # Dimensionally reduce training data
    print "Trying manifold learning method: {0}".format(ML_method)
    for key, value in ML_kwargs.iteritems():
        print "{0} = {1}".format(key, value)
    if ML_method == 'LLE':
        Y_train = manifold.LocallyLinearEmbedding(n_components=d_low, **ML_kwargs).fit_transform(X_train)
    elif ML_method == 'Isomap':
        Y_train = manifold.Isomap(n_components=d_low, **ML_kwargs).fit_transform(X_train)
    elif ML_method == 'SpectralEmbedding':
        Y_train = manifold.SpectralEmbedding(n_components=d_low, **ML_kwargs).fit_transform(X_train)
    elif ML_method == 'TSNE':
        Y_train = manifold.TSNE(n_components=d_low, **ML_kwargs).fit_transform(X_train)
    else:
        print "'{0}' is not a supported method, defaulting to LLE_standard".format(ML_method)
        Y_train = manifold.LocallyLinearEmbedding(n_components=d_low).fit_transform(X_train)

    # Scan over manifold
    print "Attempting {0} scan over {1}-dimensional manifold".format(scan_method, d_low)
    Y_max = np.array([ np.amax(Y_train[:,i]) for i in range(d_low) ])
    Y_min = np.array([ np.amin(Y_train[:,i]) for i in range(d_low) ])
    if scan_method == 'uniform':
        n_pts_d = [ int(np.ceil(n_pts**(1.0/d_low))) for i in range(d_low) ]
        n_pts = np.prod(n_pts_d)
        Y_range = np.array([ [Y_min[i]+n*(Y_max[i]-Y_min[i])/(n_pts_d[0]-1) for n in range(n_pts_d[i])] for i in range(d_low) ])
        Y = np.array([ Y_pt for Y_pt in product(*Y_range) ])
    elif scan_method == 'random':
        Y = np.array([ [ Y_min[i]+np.random.random()*(Y_max[i]-Y_min[i]) for i in range(d_low) ] for n in range(n_pts) ])
    else:
        print "'{0}' is not a supported method, defaulting to uniform scan".format(scan_method)
        n_pts_d = [ int(np.ceil(n_pts**(1.0/d_low))) for i in range(d_low) ]
        Y_range = np.array([ [Y_min[i]+n*(Y_max[i]-Y_min[i])/(n_pts_d[0]-1) for n in range(n_pts_d[i])] for i in range(d_low) ])
        Y = np.array([ Y_pt for Y_pt in product(*Y_range) ])

    # Find nearest neighbours in training data for points in scan
    n_nbrs = d_low+1
    nbr_list = NearestNeighbors(n_neighbors=n_nbrs+1, algorithm='auto').fit(Y_train)
    nbr_dists, nbrs = nbr_list.kneighbors(Y)
    nbr_dists, nbrs = np.delete(nbr_dists, n_nbrs, axis=1), np.delete(nbrs, n_nbrs, axis=1)

    # Discard points outside training region
    nbr_dists_train, nbrs_train = nbr_list.kneighbors(Y_train)
    nbr_dists_train, nbrs_train = np.delete(nbr_dists_train, 0, axis=1), np.delete(nbrs_train, 0, axis=1)
    bad_pts = [ i for i in range(n_pts) if nbr_dists[i,0] > 2*np.mean(nbr_dists_train[nbrs[i,0]]) ]
    print "{0}/{1} scan points in sparsely sampled regions: discarded".format(len(bad_pts), n_pts)
    Y = np.delete(Y, bad_pts, 0)
    nbr_dists, nbrs = np.delete(nbr_dists, bad_pts, 0), np.delete(nbrs, bad_pts, 0)
    n_pts = len(Y)

    # Find weighting matrix
    W_rows = np.zeros( shape=(n_nbrs*n_pts) )
    W_cols = np.zeros( shape=(n_nbrs*n_pts) )
    W_data = np.zeros( shape=(n_nbrs*n_pts) )
    for i in range(n_pts):
        W_rows[i*n_nbrs : (i+1)*n_nbrs] = [ i for nbr in range(n_nbrs) ]
        W_cols[i*n_nbrs : (i+1)*n_nbrs] = nbrs[i]
        C_jk = np.array([ np.append(Y_train[j], 1) for j in nbrs[i] ]).T
        w_ij = np.linalg.inv(C_jk).dot( np.append(Y[i], 1) )
        W_data[i*n_nbrs : (i+1)*n_nbrs] = w_ij
    W = sparse.coo_matrix((W_data, (W_rows, W_cols)), shape=(n_pts, n_pts_train))

    # Uplift scan to full parameter space
    print "Returning {0} points in {1}-dimensional parameter space".format(n_pts, d_high)
    return W.dot(X_train)

# Manifold learning method for training data
# Options are LLE, Isomap, SpectralEmbedding, TSNE
# Details http://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold
ML_method = 'Isomap'
# Manifold learning method args
#ML_kwargs = {'method' : 'standard', 'n_neighbors' : 20}
ML_kwargs = {'n_neighbors' : 20}
# Number of scan points to generate
n_pts = 1000

# Define training data
n_pts_train = 300
noise = 0
#X_train = np.array([ [x_val+noise*np.random.normal(), x_val**2+noise*np.random.normal()] for x_val in 1-2*np.random.random(n_pts_train) ])
X_train, color = datasets.samples_generator.make_s_curve(n_pts_train, random_state=0)
n_pts_train, d_high = X_train.shape
d_low = d_high-1

# Plot training data
fig = plt.figure(figsize=(10,5))
plt.suptitle("Manifold learning test", fontsize=14)
if d_high < 3:
    ax = fig.add_subplot(121)
    ax.scatter(X_train[:,0], X_train[:,1])
else:
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2])
    ax.view_init(4, -72)

X = ScanManifold(X_train, n_pts, ML_method, ML_kwargs)

# Plot results
n_pts = len(X)
x_min, x_max = np.amin(X_train[:,0]), np.amax(X_train[:,0])
y_min, y_max = np.amin(X_train[:,1]), np.amax(X_train[:,1])
#z_min, z_max = np.amin(X_train[:,2]), np.amax(X_train[:,2])
X = np.array([ X[i] for i in range(n_pts) if x_min <= X[i,0] <= x_max and y_min <= X[i,1] <= y_max ])
#X = np.array([ X[i] for i in range(n_pts) if x_min <= X[i,0] <= x_max and y_min <= X[i,1] <= y_max and z_min <= X[i,2] <= z_max ])
print "{0}/{1} points in plot".format(len(X), n_pts)
if d_high < 3:
    ax = fig.add_subplot(122)
    ax.scatter(X[:,0], X[:,1])
else:
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2])
    ax.view_init(4, -72)
plt.show()