# -*- coding: utf-8 -*-

import logging

import numpy as np

DEFAULT_B0_THRESHOLD = 20


def is_normalized_bvecs(bvecs):
    """
    Check if b-vectors are normalized.

    Parameters
    ----------
    bvecs : (N, 3) array
        input b-vectors (N, 3) array

    Returns
    -------
    True/False

    """

    bvecs_norm = np.linalg.norm(bvecs, axis=1)
    return np.all(np.logical_or(np.abs(bvecs_norm - 1) < 1e-3, bvecs_norm == 0))


def normalize_bvecs(bvecs, filename=None):
    """
    Normalize b-vectors

    Parameters
    ----------
    bvecs : (N, 3) array
        input b-vectors (N, 3) array
    filename : string
        output filename where to save the normalized bvecs

    Returns
    -------
    bvecs : (N, 3)
       normalized b-vectors
    """

    bvecs_norm = np.linalg.norm(bvecs, axis=1)
    idx = bvecs_norm != 0
    bvecs[idx] /= bvecs_norm[idx, None]

    if filename is not None:
        logging.info('Saving new bvecs: {}'.format(filename))
        np.savetxt(filename, np.transpose(bvecs), "%.8f")

    return bvecs


def check_b0_threshold(force_b0_threshold, bvals_min):
    """Check if the minimal bvalue is under zero or over the default threshold.
    If `force_b0_threshold` is true, don't raise an error even if the minimum
    bvalue is suspiciously high.

    Parameters
    ----------
    force_b0_threshold : bool
        If True, don't raise an error.
    bvals_min : float
        Minimum bvalue.

    Raises
    ------
    ValueError
        If the minimal bvalue is under zero or over the default threshold, and
        `force_b0_threshold` is False.
    """
    if bvals_min != 0:
        if bvals_min < 0 or bvals_min > DEFAULT_B0_THRESHOLD:
            if force_b0_threshold:
                logging.warning(
                    'Warning: Your minimal bvalue is {}. This is highly '
                    'suspicious. The script will nonetheless proceed since '
                    '--force_b0_threshold was specified.'.format(bvals_min))
            else:
                raise ValueError('The minimal bvalue is lesser than 0 or '
                                 'greater than {}. This is highly suspicious.\n'
                                 'Please check your data to ensure everything '
                                 'is correct.\n'
                                 'Value found: {}\n'
                                 'Use --force_b0_threshold to execute '
                                 'regardless.'
                                 .format(DEFAULT_B0_THRESHOLD, bvals_min))
        else:
            logging.warning('Warning: No b=0 image. Setting b0_threshold to '
                            'the minimum bvalue: {}'.format(bvals_min))


def get_shell_indices(bvals, shell, tol=10):
    return np.where(
        np.logical_and(bvals < shell + tol, bvals > shell - tol))[0]


def get_list_bvals(bval, tol=20):
    """
    List each b-values

    Parameters
    ----------
    bvals: b-values array
    tol: The tolerated gap between the b-values to extract
               and the actual b-values.

    Return
    ------
    List of sorted b-values
    """
    bvals = list(set(bvals))
    res = bvals[:]
    for bval in bvals:
        if len(np.where(np.logical_and(res < bval + tol,
                                       res > bval - tol))[0]) > 1:
            res.remove(bval)
    return sorted(res)


def get_bval_indices(bvals, bval, tol=20):
    """
    Get indices where the b-value is `bval`

    Parameters
    ----------
    bvals: b-values array
    bval: b-value to extract indices
    tol: The tolerated gap between the b-values to extract
               and the actual b-values.

    Return
    ------
    Array of indices where the b-value is `bval`
    """
    return np.where(np.logical_and(bvals < bval + tol,
                                   bvals > bval - tol))[0]
