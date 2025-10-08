## Setup and Run Instructions

Follow these steps to create a clean environment, install dependencies, and run the inference script:

```bash
# 1. Create a new Conda environment with Python 3.9.23
conda create -n exp_env python=3.9.23 -y

# 2. Activate the newly created environment
conda activate exp_env

# 3. Clone the repository
git clone https://github.com/amirifardst/Expressivity_score.git

# 4. Navigate into the project directory
cd Expressivity_score

# 5. Install required Python packages
pip install -r requirements.txt

# 6. Run the inference script
python inference.py
