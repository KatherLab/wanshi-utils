#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Didem Cifci"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.1"
__maintainer__ = ["Didem Cifci", "Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

from fire import Fire
from .helpers import feature_cluster, tile_cluster
if __name__ == '__main__':
    Fire({
        'feature_cluster': feature_cluster,
        'tile_cluster': tile_cluster,
    })