# HMS - Harmful Brain Activity Classification (HSLU AI/ML Competition)

This repository contains the code for the AI/ML competition (AICOMP) module at HSLU (Lucerne University of Applied Sciences and Arts) in the fall semester 2025. We worked on the [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/) challenge on Kaggle. The goal of this competition was to classify seizures and other patterns of harmful brain activity from EEG data.

## Authors

- David Hodel (david.hodel@stud.hslu.ch)
- Maiko Trede (maiko.trede@stud.hslu.ch)

## Repository Structure

- `data`: Contains the raw dataset downloaded from Kaggle and any processed data. The folder is .gitignored, but will be created when you run the data download script.
- `notebooks`: Contains Jupyter notebooks for exploration, analysis, and prototyping.
- `scripts`: Contains scripts for downloading the dataset, preprocessing data, training models, etc.
- `src`: Contains the source code for the project, including custom datasets, models, trainers, and utility functions.

## Setup

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

Then you need to download the dataset from Kaggle:
```bash
python scripts/01_download.py
```

If you want to contribute to the codebase, please install the pre-commit hooks by running:

```bash
pre-commit install
```

This will ensure that your code adheres to the project's coding standards before each commit.
