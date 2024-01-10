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

echo "browse dataset ..."

python tools/misc/browse_dataset.py 'configs/segmenter/segmenter_vit-b_mask_8xb1-160k_endovis2017-512x512.py' --output-dir '/nfs/home/talabi/repositories/mmsegmentation/visualization_results' 

echo "complete browse dataset ..."