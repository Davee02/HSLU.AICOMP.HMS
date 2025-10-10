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

To use this script, run (from the root of the repository):

```bash
python ./scripts/kaggle/upload_src_to_kaggle.py --title "HSM source files" --id "{your-kaggle-name}/hsm-source-files"
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
python ./scripts/kaggle/upload_notebook_to_kaggle.py -n /path/to/notebook.ipynb -t "HSM baselines"
```

This will upload the notebook to Kaggle with the title "HSM baselines".
By default, the competition data for the HSM competition and the custom dataset `dedeif/hsm-source-files` (which contains the `src/` directory) will be linked to the notebook.
You can also link additional / other datasets using the `-d` and `-c` flags.

```bash
python ./scripts/kaggle/upload_notebook_to_kaggle.py -n /path/to/notebook.ipynb -t "HSM baselines" -d "{your-kaggle-name}/hsm-source-files,{your-kaggle-name}/{your-kaggle-name}" -c "another-competition"
```

## 3. `upload_models_to_kaggle.py`

This script uploads the `models/` directory to Kaggle as a dataset.
This directory contains trained model files that can be used in Kaggle notebooks for inference without having to retrain the models.

**Important**: The subfolders within `models/` are placed directly at the root of the Kaggle dataset. For example, if you have `models/baseline/model.pth` and `models/advanced/model.pth`, they will be accessible as `/kaggle/input/hsm-models/baseline/model.pth` and `/kaggle/input/hsm-models/advanced/model.pth` (not `/kaggle/input/hsm-models/models/baseline/model.pth`).

To use this script, run:

```bash
python ./scripts/kaggle/upload_models_to_kaggle.py --title "HSM trained models" --id "{your-kaggle-name}/hsm-models"
```

If the dataset already exists, it will be updated with a new version.

To use the uploaded models in a Kaggle notebook, you need to add it as a dataset in the notebook settings.
Then, you can load the models like this:

```python
import torch

# Load a model from the models dataset
model = torch.load("/kaggle/input/hsm-models/model_name.pth")
```

You can also link the models dataset when uploading a notebook:

```bash
python ./scripts/kaggle/upload_notebook_to_kaggle.py -n notebook.ipynb -t "Inference" -d "{your-kaggle-name}/hsm-source-files,{your-kaggle-name}/hsm-models"
```

**Note:** Model files can be large, so the script displays the total size of files being uploaded.
