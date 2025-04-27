# 1. create virtual environment
# conda create -n transformers python=3.12

# 2. activate virtual environment
# conda activate transformers

# 3. apply token
# hf_WGXorTERIgBjroNXGbeFgMSqkSZHhbqiBGN_________DEPRECATED
# from huggingface_hub import notebook_login

# notebook_login()

# 4. setup
pip install -e .

# 5. install dependencies
# refer to https://huggingface.co/docs/transformers/en/quicktour

pip install torch
pip install -U transformers datasets evaluate accelerate timm jupyter jupyterlab notebook


# 6. other dependencies
pip install opencv-python dash Pillow matplotlib numpy pandas scikit-learn kk