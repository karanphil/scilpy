#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_volume_stats_in_labels import main as new_main


DEPRECATION_MSG = """
This script has been renamed scil_volume_stats_in_labels.py.
Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_compute_seed_by_label.py", DEPRECATION_MSG, '2.1.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
