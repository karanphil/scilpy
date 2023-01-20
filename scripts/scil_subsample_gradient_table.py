#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Subsamples a gradient table into a number of samples. Please note that the b0s
are removed in this process.
"""

import argparse
import logging
import numpy as np
import random

from dipy.io.gradients import read_bvals_bvecs
from scilpy.io.utils import (assert_outputs_exist,
                             add_overwrite_arg, add_verbose_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p._optionals.title = "Options and Parameters"

    p.add_argument('in_bval',
                   help='Path to the bvals file.')
    p.add_argument('in_bvec',
                   help='Path to the bvecs file.')
    p.add_argument('out_basename',
                   help='Basename to the bvals and bvecs output files.')
    p.add_argument('subsamples_sizes', nargs='+', type=int,
                   help='Size of the subsamples to make from the input bvecs.')

    p.add_argument('--tolerance',
                   type=int, default=20,
                   help='The tolerated gap between the b-values to '
                        'extract and the current b-value. [%(default)s]')
    p.add_argument('--nb_iters',
                   type=int, default=1000000,
                   help='Number of configurations (iterations) to try.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def electrostatic_repulsion(bvecs, alpha=1.0):
    """
    Electrostatic-repulsion objective function. The alpha parameter controls
    the power repulsion (energy varies as $1 / ralpha$).

    Parameters
    ---------
    bvecs : array-like shape (N * 3,)
        Vectors.
    alpha : float
        Controls the power of the repulsion. Default is 1.0

    Returns
    -------
    energy : float
        sum of all interactions between any two vectors.
    """
    epsilon = 1e-9
    N = bvecs.shape[0] // 3
    bvecs = bvecs.reshape((N, 3))
    energy = 0.0
    for i in range(N):
        indices = (np.arange(N) > i)
        diffs = ((bvecs[indices] - bvecs[i]) ** 2).sum(1) ** alpha
        sums = ((bvecs[indices] + bvecs[i]) ** 2).sum(1) ** alpha
        energy += (1.0 / (diffs + epsilon) + 1.0 / (sums + epsilon)).sum()
    return energy


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    # assert_outputs_exist(parser, args, args.out_basename)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level)

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    # Remove b0s
    bvecs = bvecs[np.argwhere(bvals>args.tolerance)]
    bvecs = np.squeeze(bvecs)
    bvals = bvals[np.argwhere(bvals>args.tolerance)]

    base_energy = electrostatic_repulsion(bvecs.reshape((bvecs.shape[0] * bvecs.shape[1])))
    print("Base energy: ", base_energy)

    nb_subsamples = len(args.subsamples_sizes)

    new_bvecs = np.zeros(bvecs.shape)
    best_bvecs = []
    best_energies = np.ones(nb_subsamples) * base_energy
    if nb_subsamples > 1:
        best_energies[0] = 0
        if nb_subsamples > 2:
            best_energies[-1] = 0
    energies = np.zeros(nb_subsamples)

    for i in range(args.nb_iters):
        remaining_bvecs = bvecs
        indices = np.arange(remaining_bvecs.shape[0])
        for j, sample_size in enumerate(args.subsamples_sizes):
            random_indices = random.sample(list(indices), sample_size)
            new_bvecs[j * sample_size:(j + 1) * sample_size] = remaining_bvecs[random_indices]
            energies[j] = electrostatic_repulsion(remaining_bvecs[random_indices].reshape((remaining_bvecs[random_indices].shape[0] * remaining_bvecs[random_indices].shape[1],)))
            remaining_bvecs = np.delete(remaining_bvecs, random_indices, 0)
            indices = np.arange(remaining_bvecs.shape[0])
        if np.mean(energies) <= np.mean(best_energies) and np.std(energies) <= np.std(best_energies):
            best_energies = np.copy(energies)
            best_bvecs = np.copy(new_bvecs)

    print("Final energies: ", best_energies)
    print("Finale mean energy and std:", np.mean(best_energies), np.std(best_energies))

    for j, sample_size in enumerate(args.subsamples_sizes):
        bvecs_filename = args.out_basename + "_" + str(j) + ".bvec"
        bvals_filename = args.out_basename + "_" + str(j) + ".bval"
        np.savetxt(bvecs_filename, best_bvecs[j * sample_size:(j + 1) * sample_size].T, fmt='%.8f')
        np.savetxt(bvals_filename, bvals[j * sample_size:(j + 1) * sample_size], fmt='%.3f')


if __name__ == "__main__":
    main()
