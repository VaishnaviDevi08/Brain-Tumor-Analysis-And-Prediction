# -- coding: utf-8 --
"""
@author: Serena Grazia De Benedictis, Grazia Gargano, Gaetano Settembre
"""

import time
import numpy as np  # handling arrays and general math
from scipy import sparse  # working with sparse matrices
from ripser import lower_star_img  # computing topological persistence of images
from scipy.sparse.csgraph import connected_components  # compute connected components from sparse adjacency matrix
import cv2  # image processing library
from scipy import ndimage  # image smoothening
from scipy.ndimage.morphology import distance_transform_edt  # compute closest background pixel


def img_to_sparseDM(img):
    """
    Compute a sparse distance matrix from the pixel entries of a single channel image for persistent homology.
    """
    m, n = img.shape
    idxs = np.arange(m * n).reshape((m, n))

    I, J = idxs.flatten(), idxs.flatten()
    V = img.flatten()

    img[img == -np.inf] = np.inf

    # Connect 8 spatial neighbors
    tidxs = np.pad(idxs, pad_width=1, mode='constant', constant_values=np.nan)
    tD = np.pad(img, pad_width=1, mode='constant', constant_values=np.nan)

    for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        shifted_idxs = np.roll(np.roll(tidxs, di, axis=0), dj, axis=1)
        shifted_distances = np.maximum(tD, np.roll(np.roll(tD, di, axis=0), dj, axis=1))

        valid_mask = ~np.isnan(shifted_distances)
        I = np.concatenate((I, tidxs[valid_mask].flatten()))
        J = np.concatenate((J, shifted_idxs[valid_mask].flatten()))
        V = np.concatenate((V, shifted_distances[valid_mask].flatten()))

    return sparse.coo_matrix((V, (I, J)), shape=(idxs.size, idxs.size))


def connected_components_img(img):
    """
    Identify the connected components of an image.
    """
    return connected_components(img_to_sparseDM(img), directed=False)[1].reshape(img.shape)


def smoothen(img, window_size):
    """Apply smoothing to an image."""
    return ndimage.uniform_filter(img.astype("float"), size=window_size)


def add_border(img, border_width):
    """Add a border to the image with minimal value."""
    border_value = np.min(img) - 1
    img[:border_width, :] = border_value
    img[-border_width:, :] = border_value
    img[:, :border_width] = border_value
    img[:, -border_width:] = border_value
    return img


def lifetimes_from_dgm(dgm, tau=False):
    """
    Rotate a persistence diagram to show lifetimes.
    """
    dgm_lifetimes = np.vstack([dgm[:, 0], dgm[:, 1] - dgm[:, 0]]).T

    if tau:
        finite_points = dgm_lifetimes[dgm_lifetimes[:, 1] < np.inf]
        distances = np.diff(np.sort(finite_points[:, 1]))
        most_separated = np.argmax(distances)
        tau = (finite_points[most_separated, 1] + finite_points[most_separated + 1, 1]) / 2
        return dgm_lifetimes, tau

    return dgm_lifetimes


def topological_process_img(img, dgm=None, tau=None, window_size=None, border_width=None):
    """
    Perform topological processing on an image.
    """
    if dgm is None:
        if window_size:
            img = smoothen(img, window_size)
        if border_width:
            img = add_border(img, border_width)
        dgm = lower_star_img(img)

    dgm_lifetimes, tau = lifetimes_from_dgm(dgm, tau=True) if tau is None else (lifetimes_from_dgm(dgm), tau)

    idxs = np.where((tau < dgm_lifetimes[:, 1]) & (dgm_lifetimes[:, 1] < np.inf))[0]
    img_processed = np.zeros_like(img)

    for idx in idxs:
        temp_img = (img >= dgm[idx, 0]) & (img < dgm[idx, 1])
        components = connected_components_img(temp_img)
        img_processed[components > 0] = img[components > 0]

    return {"processed": img_processed}


def tda_and_mask(x_data, window_size, border_width, param_mask):
    """
    Apply Topological Data Analysis and generate masks for each image.
    """
    x_data_tda = []

    start_time = time.time()
    for image in x_data:
        TIP_img = topological_process_img(image, window_size=window_size, border_width=border_width)
        mask = TIP_img["processed"]
        mask[mask == 0] = param_mask
        x_data_tda.append(mask * (1 - image))

    elapsed_time = time.time() - start_time
    print(f"Time for Topological Data Analysis: {elapsed_time:.2f} seconds")

    return x_data_tda
