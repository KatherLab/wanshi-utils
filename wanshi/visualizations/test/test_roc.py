#!/usr/bin/env python3

__author__ = 'Jeff'
__copyright__ = 'Copyright 2023, Kather Lab'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = ['Jeff']
__email__ = 'jiefu.zhu@tu-dresden.de'

from ..roc_curve.roc import *
import pytest

# pytest for read_table()
def test_read_table():
    import pandas as pd
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # test csv
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_csv(tmpdir / "test.csv", index=False)
        assert (read_table(tmpdir / "test.csv") == df).all().all()

        # test xlsx
        df.to_excel(tmpdir / "test.xlsx", index=False)
        assert (read_table(tmpdir / "test.xlsx") == df).all().all()

        # test unknown filetype
        with pytest.raises(ValueError):
            read_table(tmpdir / "test.txt")


