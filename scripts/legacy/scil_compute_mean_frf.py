#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_reconst_frf_mean import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_reconst_frf_mean.py.
Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_compute_mean_frf.py", DEPRECATION_MSG, '1.7.0')
def main():
    new_main()


if __name__ == "__main__":
    main()