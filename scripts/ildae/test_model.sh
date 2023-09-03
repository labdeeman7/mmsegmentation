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

#endovis
#segmenter
# python test_model.py 'configs/segmenter/segmenter_vit-b_mask_8xb1-160k_endovis2017-512x512.py' 'work_dirs/segmenter_vit-b_mask_8xb1-160k_endovis2017-512x512/iter_160000.pth' \
#         --work-dir 'work_dirs/segmenter_vit-b_mask_8xb1-160k_endovis2017-512x512' \
#         --out 'work_dirs/segmenter_vit-b_mask_8xb1-160k_endovis2017-512x512/preds/' \
#         --show-dir 'work_dirs/segmenter_vit-b_mask_8xb1-160k_endovis2017-512x512/vis_preds/'      

# #deeplab
# python test_model.py 'configs/deeplabv3/deeplabv3_r50-d8_4xb4-20k_endovis2017-512x512.py' 'work_dirs/deeplabv3_r50-d8_4xb4-20k_endovis2017-512x512/iter_20000.pth' \
#         --work-dir 'work_dirs/deeplabv3_r50-d8_4xb4-20k_endovis2017-512x512/' \
#         --out 'work_dirs/deeplabv3_r50-d8_4xb4-20k_endovis2017-512x512/preds/' \
#         --show-dir 'work_dirs/deeplabv3_r50-d8_4xb4-20k_endovis2017-512x512/vis_preds/'    

# #u-net
# python test_model.py 'configs/unet/unet-s5-d16_fcn_4xb4-20k_endovis2017-512x512.py' 'work_dirs/unet-s5-d16_fcn_4xb4-20k_endovis2017-512x512/iter_20000.pth' \
#         --work-dir 'work_dirs/unet-s5-d16_fcn_4xb4-20k_endovis2017-512x512' \
#         --out 'work_dirs/unet-s5-d16_fcn_4xb4-20k_endovis2017-512x512/preds/' \
#         --show-dir 'work_dirs/unet-s5-d16_fcn_4xb4-20k_endovis2017-512x512/vis_preds/'    

#autolaparo
#segmenter
# python test_model.py 'configs/segmenter/segmenter_vit-b_mask_8xb1-20k_autolaparo-512x512.py' \
#         'work_dirs/segmenter_vit-b_mask_8xb1-160k_autolaparo-512x512/iter_20000.pth' \
#         --work-dir 'work_dirs/segmenter_vit-b_mask_8xb1-160k_autolaparo-512x512' \
#         --out 'work_dirs/segmenter_vit-b_mask_8xb1-160k_autolaparo-512x512/pred_labels/' \
#         --show-dir 'work_dirs/segmenter_vit-b_mask_8xb1-160k_autolaparo-512x512/vis_preds/'  

# echo "completed evaluating on segmenter ..."           

# # #deeplab
# python test_model.py 'configs/deeplabv3/deeplabv3_r50-d8_4xb4-20k_autolaparo-512x512.py' \
#         'work_dirs/deeplabv3_r50-d8_4xb4-20k_autolaparo-512x512/iter_14000.pth' \
#         --work-dir 'work_dirs/deeplabv3_r50-d8_4xb4-20k_autolaparo-512x512' \
#         --out 'work_dirs/deeplabv3_r50-d8_4xb4-20k_autolaparo-512x512/pred_labels/' \
#         --show-dir 'work_dirs/deeplabv3_r50-d8_4xb4-20k_autolaparo-512x512/vis_preds/'  

# echo "completed evaluating on deeplab ..."         

# # #u-net
# python test_model.py 'configs/unet/unet-s5-d16_fcn_4xb4-40k_robustmis-448x448.py' \
#         'work_dirs/unet-s5-d16_fcn_4xb4-40k_robustmis-448x448/iter_40000.pth' \
#         --work-dir 'work_dirs/unet-s5-d16_fcn_4xb4-40k_robustmis-448x448' \
#         --out 'work_dirs/unet-s5-d16_fcn_4xb4-40k_robustmis-448x448/pred_labels/' \
#         --show-dir 'work_dirs/unet-s5-d16_fcn_4xb4-40k_robustmis-448x448/vis_preds/'   

#robustmis
#segmenter
python test_model.py 'configs/segmenter/segmenter_vit-b_mask_8xb1-40k_robustmis-448x448.py' \
        'work_dirs/segmenter_vit-b_mask_8xb1-40k_robustmis-448x448/iter_40000.pth' \
        --work-dir 'work_dirs/segmenter_vit-b_mask_8xb1-40k_robustmis-448x448' \
        --out 'work_dirs/segmenter_vit-b_mask_8xb1-40k_robustmis-448x448/pred_labels/' \
        --show-dir 'work_dirs/segmenter_vit-b_mask_8xb1-40k_robustmis-448x448/vis_preds/'  

echo "completed evaluating on segmenter ..."           

# #deeplab
python test_model.py 'configs/deeplabv3/deeplabv3_r50-d8_4xb4-40k_robustmis-256x256.py' \
        'work_dirs/deeplabv3_r50-d8_4xb4-40k_robustmis-256x256/iter_40000.pth' \
        --work-dir 'work_dirs/deeplabv3_r50-d8_4xb4-40k_robustmis-256x256' \
        --out 'work_dirs/deeplabv3_r50-d8_4xb4-40k_robustmis-256x256/pred_labels/' \
        --show-dir 'work_dirs/deeplabv3_r50-d8_4xb4-40k_robustmis-256x256/vis_preds/'  

echo "completed evaluating on deeplab ..."         

# #u-net
python test_model.py 'configs/unet/unet-s5-d16_fcn_4xb4-40k_robustmis-448x448.py' \
        'work_dirs/unet-s5-d16_fcn_4xb4-40k_robustmis-448x448/iter_40000.pth' \
        --work-dir 'work_dirs/unet-s5-d16_fcn_4xb4-40k_robustmis-448x448' \
        --out 'work_dirs/unet-s5-d16_fcn_4xb4-40k_robustmis-448x448/pred_labels/' \
        --show-dir 'work_dirs/unet-s5-d16_fcn_4xb4-40k_robustmis-448x448/vis_preds/'   

echo "completed evaluating on unet ..."