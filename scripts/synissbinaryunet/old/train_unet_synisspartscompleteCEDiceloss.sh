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

echo "training unet synisspart ..."


#syniss complete with ce and dice loss
python train_model.py '/nfs/home/talabi/repositories/mmsegmentation/configs/unet/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteCEDice-448x896.py' 

echo "completed training unet synisspartcomplete syniss ce dice complete. ..."