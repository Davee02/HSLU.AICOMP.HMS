# AICOMP HS25: HMS - Harmful Brain Activity Classification

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/

## Authors

- David Hodel (david.hodel@stud.hslu.ch)
- Maiko Trede (maiko.trede@stud.hslu.ch)

## Structure

- `notebooks` Jupyter Notebooks go in here, try to use notebooks only for analysis and prototyping, the real training should be done via scripts
- `scripts` Python/Bash/PowerShell scripts go in here, that can be downloading the dataset, transforming the data, training a model etc.
- `src` The source code (heart) of your project
    * `datasets`: Write your [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) in here
    * `evaluation`: Write tasks used to evaluate your model(s) in this module
    * `models`: Custom model implementation go in here
    * `optimizers`: Custom optimizers are written in here
    * `schedulers`: Custom implementation of schedulers
    * `trainers`: Your model trainer(s) live here
    * `transforms`: If you need specific transformations you will create them in here
    * `utils`: Utility functions and modules

## TODOs
1. Set up your dev environment
```bash
pip install -r requirements.txt
# Install pre-commit hook
pre-commit install
```
To run all the linters on all files:
```bash
pre-commit run --all-files
```

2. Change project name and description in `pyproject.toml`

```toml
[project]
name = "Datascience Project Template"
description = "Template for an AICOMP DataScience Project"
version = "0.1.0"
authors = [
    {name = "Pascal Baumann", email = "pascal.baumann@hslu.ch"},
]
```

3. Add your requirements to `requirements.txt`
4. Create some code =)
5. Add a PyTest configuration in PyCharm
![img.png](assets/README/img.png)


## Code and test conventions
- `black` for code style
- `isort` for import sorting
- `darglint` for docstring checking
- docstring style: `sphinx`
- `pytest` for running tests
- `nbclean` cleans up your Jupyter notebooks before committing
- main/master branch is protected and needs merge request with approval

### Running on the GPU and logging to W&B
Due to us now having an enterprise license on W&B we also have our service bot which can log runs to the appropriate team.
