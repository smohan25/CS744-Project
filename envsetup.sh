sudo apt update --fix-missing
sudo apt install -y python3-pip
pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip3 install future
pip3 install -U intel-numpy
