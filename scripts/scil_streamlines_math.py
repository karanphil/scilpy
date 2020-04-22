#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performs an operation on a list of streamline files. The supported
operations are:

    difference:  Keep the streamlines from the first file that are not in
                 any of the following files.

    intersection: Keep the streamlines that are present in all files.

    union:        Keep all streamlines while removing duplicates.

    concatenate:  Keep all streamlines with duplicates.

If a file 'duplicate.trk' have identical streamlines calling the script using
the difference/intersection/union with a single input will remove these
duplicated streamlines.

To allow a soft match, use the --precision option to increase the allowed
threshold for similarity. A precision of 1 represent 10**(-1), so a
maximum distance of 0.1mm  is allowed. If the streamlines are identical, the
default value of 3 or 0.001mm distance) should work. If there is a 0.5mm shift,
use a precision of 0 (or 1mm distance) should work, but slightly slower.

The metadata (data per point, data per streamline) of the streamlines that
are kept in the output will preserved. This requires that all input files
share the same type of metadata. If this is not the case, use the option
--no_metadata to strip the metadata from the output.
"""

import argparse
import json
import logging

from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.streamlines import (difference, intersection, union, sum_sft)


OPERATIONS = {
    'difference': difference,
    'intersection': intersection,
    'union': union,
    'concatenate': 'concatenate'
}


def _build_arg_parser():

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('operation', choices=OPERATIONS.keys(), metavar='OPERATION',
                   help='The type of operation to be performed on the '
                   'streamlines. Must\nbe one of the following: '
                   '%(choices)s.')
    p.add_argument('inputs', metavar='INPUT_FILES', nargs='+',
                   help='The list of files that contain the ' +
                   'streamlines to operate on.')
    p.add_argument('output', metavar='OUTPUT_FILE',
                   help='The file where the remaining streamlines '
                   'are saved.')

    p.add_argument('--precision', '-p', metavar='NBR_OF_DECIMALS',
                   type=int, default=3,
                   help='Precision used to compare streamlines [%(default)s].')

    p.add_argument('--no_metadata', '-n', action='store_true',
                   help='Strip the streamline metadata from the output.')
    p.add_argument('--save_indices', '-s', metavar='OUT_INDEX_FILE',
                   help='Save the streamline indices to the supplied '
                        'json file.')

    add_json_args(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, args.inputs)
    assert_outputs_exist(parser, args, args.output)

    # Load all input streamlines.
    sft_list = [load_tractogram_with_reference(parser, args,
                                               f) for f in args.inputs]

    # Apply the requested operation to each input file.
    logging.info('Performing operation \'{}\'.'.format(args.operation))
    new_sft = sum_sft(sft_list, args.no_metadata)
    if args.operation == 'concatenate':
        indices = range(len(new_sft))
    else:
        streamlines_list = [sft.streamlines for sft in sft_list]
        _, indices = OPERATIONS[args.operation](streamlines_list,
                                                precision=args.precision)

    # Save the indices to a file if requested.
    if args.save_indices:
        start = 0
        out_dict = {}
        streamlines_len_cumsum = [len(sft) for sft in sft_list]
        for name, nb in zip(args.inputs, streamlines_len_cumsum):
            end = start + nb
            # Switch to int32 for json
            out_dict[name] = [int(i - start)
                              for i in indices if start <= i < end]
            start = end

        with open(args.save_indices, 'wt') as f:
            json.dump(out_dict, f,
                      indent=args.indent,
                      sort_keys=args.sort_keys)

    # Save the new streamlines (and metadata)
    logging.info('Saving {} streamlines to {}.'.format(len(indices),
                                                       args.output))
    save_tractogram(new_sft[indices], args.output)


if __name__ == "__main__":
    main()
