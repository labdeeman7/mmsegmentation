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


# echo "completed evaluating on syniss binary complete..."         

#unet ce 
# python test_model.py 'configs/unet/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteCE-448x896.py' \
#         'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteCE-448x896/iter_40000.pth' \
#         --work-dir 'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteCE-448x896' \
#         --out 'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteCE-448x896/pred_labels/' \
#         --show-dir 'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteCE-448x896/vis_preds/' 

# echo "completed evaluating on ce complete..."  

# #unet wce
# python test_model.py 'configs/unet/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteWCE-448x896.py' \
#         'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteWCE-448x896/iter_40000.pth' \
#         --work-dir 'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteWCE-448x896' \
#         --out 'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteWCE-448x896/pred_labels/' \
#         --show-dir 'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteWCE-448x896/vis_preds/'

# echo "completed evaluating on wce complete..."

# #unet ce ohem
# python test_model.py 'configs/unet/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteOHEM-448x896.py' \
#         'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteOHEM-448x896/iter_40000.pth' \
#         --work-dir 'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteOHEM-448x896' \
#         --out 'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteOHEM-448x896/pred_labels/' \
#         --show-dir 'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteOHEM-448x896/vis_preds/'

# echo "completed evaluating on ce ohem complete..."


# python test_model.py 'configs/unet/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteWCEOHEM-448x896.py' \
#         'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteWCEOHEM-448x896/iter_40000.pth' \
#         --work-dir 'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteWCEOHEM-448x896' \
#         --out 'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteWCEOHEM-448x896/pred_labels/' \
#         --show-dir 'work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteWCEOHEM-448x896/vis_preds/'

python test_model.py 'configs/unet/unet-s5-d16_fcn_4xb4-80k_synisspartscompleteWCEOHEMAugLong-448x896.py' \
        'work_dirs/unet-s5-d16_fcn_4xb4-80k_synisspartscompleteWCEOHEMAugLong-448x896/iter_40000.pth' \
        --work-dir 'work_dirs/unet-s5-d16_fcn_4xb4-80k_synisspartscompleteWCEOHEMAugLong-448x896' \
        --out 'work_dirs/unet-s5-d16_fcn_4xb4-80k_synisspartscompleteWCEOHEMAugLong-448x896/pred_labels/' \
        --show-dir 'work_dirs/unet-s5-d16_fcn_4xb4-80k_synisspartscompleteWCEOHEMAugLong-448x896/vis_preds/'

echo "completed evaluating on ce ohem wce aug long complete..."

         

  

