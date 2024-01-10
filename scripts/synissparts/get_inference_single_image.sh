#!/usr/bin/python

# echo "started ..."

cd /nfs/home/talabi/repositories/mmsegmentation

python -m venv mmseg

source mmseg/bin/activate

pip install torch==1.12.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim
mim install mmcv
pip install mmengine
pip install scikit-image

pip install -e .

echo "evaluating single image ..."



python run_inference_single_image.py --config_path 'configs/unet/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteCE-448x896.py' \
        --checkpoint_path 'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteCE-448x896/iter_40000.pth' \
        --img_path '/nfs/home/talabi/data/Endovis_challenges_2023/syniss/syniss_complete/img_dir/val/0efc36d83218edbdc390.png' \
        --save_dir 'testing_single_image_inference/'
        

echo "completed evaluating on single image"

         

  

