# HMS - Harmful Brain Activity Classification (HSLU AI/ML Competition)

This repository contains the code for the AI/ML competition (AICOMP) module at HSLU (Lucerne University of Applied Sciences and Arts) in the fall semester 2025. We worked on the [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/) challenge on Kaggle. The goal of this competition was to classify seizures and other patterns of harmful brain activity from EEG data.

## Authors

- David Hodel (david.hodel@stud.hslu.ch)
- Maiko Trede (maiko.trede@stud.hslu.ch)

## Repository Structure

- `data/`: Contains the raw dataset downloaded from Kaggle and any processed data. The folder is .gitignored, but will be created when you run the data download and (pre)processing scripts.
- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis, model development, visualization, and experiments.
  - `notebooks/models/`: Contains notebooks specifically related to model training and evaluation.
  - `notebooks/_archive/`: Contains older notebooks that are no longer actively used but are kept for reference.
- `scripts/`: Contains scripts for downloading the dataset, preprocessing data, training models, etc.
  - `scripts/kaggle/`: Contains scripts to make the workflow with Kaggle easier. They are not necessary to run the main codebase. For more details, see the [README](scripts/kaggle/readme.md) in that folder.
  - `scripts/_archive/`: Contains older scripts that are no longer actively used but are kept for reference.
- `src/`: Contains the source code for the project, including custom datasets, models, trainers, and utility functions.

## Reproduction Instructions

First, ensure you have Python installed.
We recommend using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) to manage your Python environment.
We used the latest 3.12 version (3.12.11 at the time of writing) for development.
Other versions might work, but are not guaranteed to.

To set up the environment and install the required packages, run the following commands:

```bash
conda create -n aicomp-hms python=3.12 -y
conda activate aicomp-hms
pip install -r requirements.txt
```

Then you need to download the dataset from Kaggle (you need to be logged into your Kaggle account for this step; follow [these instructions](https://www.kaggle.com/docs/api#authentication) if you haven't set up the Kaggle API before):

```bash
python scripts/01_download.py
```

After downloading the data, you can preprocess the EEG data by running the following two scripts (caution: the second script takes multiple hours to complete):

```bash
python scripts/02_preprocess_eeg_data.py
python scripts/03_create_eeg_spectrograms.py --spectrogram_types cwt,stft,mel
```

## Contributing

If you want to contribute to the codebase, please install the pre-commit hooks by running:

```bash
pre-commit install
```

This will ensure that your code adheres to the project's coding standards before each commit.
