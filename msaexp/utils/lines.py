"""
Tools for working with emission lines
"""
import os

import numpy as np
from grizli import utils

from .general import module_data_path

__all__ = ["MolecularHydrogen"]

class MolecularHydrogen():
    def __init__(self, **kwargs):
        self.data = utils.read_catalog(
            os.path.join(module_data_path, "h2_lines.txt")
        )

