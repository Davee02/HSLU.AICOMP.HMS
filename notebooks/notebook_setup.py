import os
import sys

import pandas as pd

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

pd.set_option("display.max_columns", None)  # shows all columns when printing a dataframe
