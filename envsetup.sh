#!/usr/bin/env bash

sudo apt update --fix-missing
sudo apt install -y python3-venv
python3 -m venv ./venv
source ./venv/bin/activate
pip install wheel
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install future
pip install -U intel-numpy
pip install scipy
