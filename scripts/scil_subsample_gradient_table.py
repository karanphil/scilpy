#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate multi-shell gradient sampling with various processing to accelerate
acquisition and help artefact correction.

Multi-shell gradient sampling is generated as in [1], the bvecs are then
flipped to maximize spread for eddy current correction, b0s are interleaved
at equal spacing and the non-b0 samples are finally shuffled
to minimize the total diffusion gradient amplitude over a few TR.
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
    p.add_argument('out_bvec',
                   help='Path to the bvecs output file.')
    p.add_argument('nb_subsamples', type=int)

    p.add_argument('--tolerance',
                   type=int, default=20,
                   help='The tolerated gap between the b-values to '
                        'extract and the current b-value. [%(default)s]')
    p.add_argument('--nb_iter',
                   type=int, default=100,
                   help='Number of iterations.')

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

    assert_outputs_exist(parser, args, args.out_bvec)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level)

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    # Remove b0s
    bvecs = bvecs[np.argwhere(bvals>args.tolerance)]
    bvecs = np.squeeze(bvecs)

    new_bvecs = np.zeros(bvecs.shape)
    best_bvecs = []
    best_energies = np.ones(args.nb_subsamples) * 1000000
    energies = np.zeros(args.nb_subsamples)

    subsamples_length = int(np.floor(bvecs.shape[0] / args.nb_subsamples))

    for i in range(args.nb_iter):
        remaining_bvecs = bvecs
        indices = np.arange(remaining_bvecs.shape[0])
        for sample in range(args.nb_subsamples):
            random_indices = random.sample(list(indices), subsamples_length)
            new_bvecs[sample * subsamples_length:(sample + 1) * subsamples_length] = remaining_bvecs[random_indices]
            remaining_bvecs = np.delete(remaining_bvecs, random_indices, 0)
            indices = np.arange(remaining_bvecs.shape[0])
            energies[sample] = electrostatic_repulsion(new_bvecs.reshape((new_bvecs.shape[0] * new_bvecs.shape[1],)))
            print(energies)
        if np.mean(energies) < np.mean(best_energies) and np.std(energies) < np.std(best_energies):
            best_energies = energies
            best_bvecs = new_bvecs

    print(best_bvecs)


if __name__ == "__main__":
    main()
