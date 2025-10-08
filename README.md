conda create -n exp_env python==3.9.23

conda activate exp_env

git clone https://github.com/amirifardst/Expressivity_score.git

pip install -r requirements.txt

python .\inference.py




