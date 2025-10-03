# Kaggle Scripts

This directory contains scripts to make the workflow with Kaggle easier.
Because making a submission for a Kaggle competition requires executing a Jupyter notebook (and only that!) on their infrastructure, often with internet access disabled (like for our competition), one must include all code necessary to generate the submission in the notebook itself.
This can lead to a lot of code duplication and a messy notebook, especially when the code is scattered across multiple files.

Furthermore, every time a notebook is created on Kaggle, one must re-link all datasets that are used in the notebook.

To alleviate these issues, we created the following scripts:

## 1. `upload_src_to_kaggle.py`

This script uploads the `src/` directory to Kaggle as a dataset.
That directory contains a lot of helper code that is used in the notebooks.
This way, we can simply import the code from `src/` in our notebooks without having to copy-paste it.

To use this script, run:

```bash
python upload_src_to_kaggle.py --title "HSM source files" --id "dedeif/hsm-source-files/data"
```

If the dataset already exists, it will be updated.

To use the uploaded dataset in a Kaggle notebook, you need to add it as a dataset in the notebook settings.
Then, you can import the code from `src/` like this:

```python
import sys
sys.path.insert(0, "/kaggle/input/hsm-source-files")

from src.utils.constants import Constants
```

## 2. `upload_notebook_to_kaggle.py`

This script uploads a local Jupyter notebook to Kaggle as a notebook.
This way, you can easily update the notebook on Kaggle without having to copy-paste the code or re-link the datasets.

To use this script, run:

```bash
python scripts/kaggle/upload_notebook_to_kaggle.py -n /path/to/notebook.ipynb -t "HSM baselines"
```

This will upload the notebook to Kaggle with the title "HSM baselines".
By default, the competition data for the HSM competition and the custom dataset `dedeif/hsm-source-files` (which contains the `src/` directory) will be linked to the notebook.
You can also link additional / other datasets using the `-d` and `-c` flags.

```bash
python scripts/kaggle/upload_notebook_to_kaggle.py -n /path/to/notebook.ipynb -t "HSM baselines" -d "dedeif/hsm-source-files,another/dataset" -c "another-competition"
```
