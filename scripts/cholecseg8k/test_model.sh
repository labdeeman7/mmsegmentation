#!/usr/bin/python
cd /nfs/home/talabi/repositories/mmsegmentation

python -m venv mmseg

source mmseg/bin/activate

pip install torch==1.12.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim
mim install mmcv
pip install mmengine

pip install -e .

echo "evaluating model ..."
 

#robustmis
#segmenter
python scripts/cholecseg8k/segment.py

echo "completed evaluating on segmenter ..."           
echo "completed evaluating on deeplab ..."         
