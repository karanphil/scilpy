#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_dwi_concatenate import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_dwi_concatenate.py.
Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_concatenate_dwi.py", DEPRECATION_MSG, '2.1.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
