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

echo "training segmenter ..."

python train_model.py '/nfs/home/talabi/repositories/mmsegmentation/configs/segmenter/segmenter_vit-b_mask_8xb1-80k_synissbinary-448x896.py' 

echo "completed training segmenter..."