#!/usr/bin/python

# echo "started ..."

cd /nfs/home/talabi/repositories/mmsegmentation

python -m venv mmseg

source mmseg/bin/activate

pip install torch==1.12.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim
mim install mmcv
pip install mmengine

pip install -e .

echo "evaluating model ..."

#synissbinary 
# #u-net
# python test_model.py 'configs/unet/unet-s5-d16_fcn_4xb4-80k_synissbinarycomplete-448x896.py' \
#         'work_dirs/unet-s5-d16_fcn_4xb4-80k_synissbinarycomplete-448x896/iter_80000.pth' \
#         --work-dir 'work_dirs/unet-s5-d16_fcn_4xb4-80k_synissbinarycomplete-448x896' \
#         --out 'work_dirs/unet-s5-d16_fcn_4xb4-80k_synissbinarycomplete-448x896/pred_labels/' \
#         --show-dir 'work_dirs/unet-s5-d16_fcn_4xb4-80k_synissbinarycomplete-448x896/vis_preds/' 

# echo "completed evaluating on syniss binary complete..."         

#synissparts 
python test_model.py 'configs/unet/unet-s5-d16_fcn_4xb4-80k_synisspartscompletetrain-448x896.py' \
        'work_dirs/unet-s5-d16_fcn_4xb4-80k_synisspartscomplete-448x896/iter_80000.pth' \
        --work-dir 'work_dirs/unet-s5-d16_fcn_4xb4-80k_synisspartscompletetrain-448x896' \
        --out 'work_dirs/unet-s5-d16_fcn_4xb4-80k_synisspartscompletetrain-448x896/pred_labels/' \
        --show-dir 'work_dirs/unet-s5-d16_fcn_4xb4-80k_synisspartscompletetrain-448x896/vis_preds/' 

echo "completed evaluating on syniss parts complete..."  
         

  

