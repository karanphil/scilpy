#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

from dipy.io.streamline import save_tractogram

from scilpy.io.utils import (add_overwrite_arg, add_reference,
                             assert_inputs_exist, assert_outputs_exists,
                             load_tractogram_with_reference)

DESCRIPTION = """
Conversion of '.tck', '.trk', '.fib', '.vtk' files using updated file format
standard. TRK file always needs a reference file, a NIFTI, for conversion.
The FIB file format is in fact a VTK, MITK Diffusion supports it.
"""


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram', metavar='IN_TRACTOGRAM',
                   help='Tractogram filename. Format must be \n'
                        'readable by the Nibabel.')

    p.add_argument('output_name', metavar='OUTPUT_NAME',
                   help='Output filename. Format must be \n'
                        'writable by Nibabel.')

    add_reference(p)

    add_overwrite_arg(p)

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram], [args.reference])

    in_extension = os.path.splitext(args.in_tractogram)[1]
    out_extension = os.path.splitext(args.output_name)[1]

    if in_extension == out_extension:
        parser.error('Input and output cannot be of the same file format')

    assert_outputs_exists(parser, args, args.output_name)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram,
                                         bbox_check=False)
    save_tractogram(sft, args.output_name, bbox_valid_check=False)


if __name__ == "__main__":
    main()
