# Project Documentation

This repository is structured according to best practices in data-driven research and machine learning system development.  
The design aims to ensure **reproducibility**, **scalability**, and **maintainability** of experiments and source code.

---

## Directory Structure

### `data/`
This directory contains all datasets related to the project, separated into three categories:
- **`raw/`** : Unmodified, original datasets obtained directly from sources (e.g., sensor logs, CSV files, experimental measurements). These files must remain unchanged to preserve data integrity.
- **`processed/`** : Data that has been cleaned, normalized, or otherwise transformed for direct use in experiments or model training.
- **`manifests/`** : Metadata and schema-related files describing dataset structure, data dictionaries, and manifest files for reproducibility.

---

### `experiments/`
Dedicated to storing experiment artifacts, such as:
- Model checkpoints  
- Logs (e.g., training/evaluation logs)  
- Experiment configurations  
- Results and evaluation metrics  

Each experiment should be organized in a separate subdirectory (e.g., `exp_01/`, `exp_02/`) to ensure clarity and traceability.

---

### `notebooks/`
Contains Jupyter notebooks for:
- Exploratory Data Analysis (EDA)  
- Data visualization  
- Model prototyping  
- Documentation of research workflows  

Notebooks serve as both an **analytical tool** and a **living document** of the research process.

---

### `scripts/`
Standalone scripts that automate workflows and tasks, such as:
- Data preprocessing  
- Training routines  
- Evaluation pipelines  
- Utility scripts  

Scripts are intended to be lightweight and task-oriented.

---

### `SETUP_PROJECT/`
Contains environment-related configurations, dependency setups, or local development tools.  
For instance:
- Environment initialization files  
- Dependency management (if not handled via `requirements.txt`)  
- Virtual environment or setup instructions  

---

### `src/`
Source code of the project, organized into modular components. Typical contents include:
- Data loaders and preprocessing pipelines  
- Model definitions  
- Training and optimization routines  
- Utility functions and supporting modules  

This directory represents the **core implementation** of the system.

---

### `tests/`
Contains testing scripts to ensure the reliability and correctness of the code in `src/`.  
Recommended practices:
- Unit tests for individual modules  
- Integration tests for workflows  
- Use of frameworks such as `pytest`  

Testing is essential for maintaining long-term code stability.

---

## Root-Level Files

- **`.gitignore`** : Defines which files and directories should be excluded from version control (e.g., temporary files, cache, logs, large datasets).  
- **`requirements.txt`** : A comprehensive list of dependencies required to reproduce the project environment. Install using:  
  ```bash
  pip install -r requirements.txt
