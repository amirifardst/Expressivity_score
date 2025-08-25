import os
from pathlib import Path

# Create a list of files  in the current directory we want to have in our template
list_of_files = [
    "README.md",
    "requirements.txt",
    "setup.py",
    "src/__init__.py",
    "src/main.py",
    "src/utils/utils.py",
    "src/logging/logger.py",
    "src/engine/__init__.py",
    "src/engine/trainer.py",
    "src/engine/evaluator.py",
    "src/engine/predictor.py",
    "tests/__init__.py",
    "tests/test_main.py",
    "tests/test_utils.py",
    "config/config.yaml",
    "config/config.json",
    "data/dataloading.py",
    "data/preprocessing.py",
    "data/visualization.py",
    "data/transformater.py",
    "notebooks/analysis.ipynb",
    ]

# Create above files in the current directory
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:  
            pass
    else:
        print(f"File already exists: {filepath}")


        