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

echo "training unet ..."

python train_model.py '/nfs/home/talabi/repositories/mmsegmentation/configs/unet/unet-s5-d16_fcn_4xb4-40k_robustmis-448x448.py' 

echo "completed training unet ..."