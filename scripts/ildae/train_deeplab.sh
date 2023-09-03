#!/usr/bin/python

echo "started ..."

cd /nfs/home/talabi/repositories/mmsegmentation

python -m venv mmseg

source mmseg/bin/activate

pip install torch==1.12.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim
mim install mmcv
pip install mmengine

pip install -e .

echo "training deeplab autolaparo..."

python train_model.py '/nfs/home/talabi/repositories/mmsegmentation/configs/deeplabv3/deeplabv3_r50-d8_4xb4-40k_robustmis-256x256.py' 

echo "completed training deeplab autolaparo ..."