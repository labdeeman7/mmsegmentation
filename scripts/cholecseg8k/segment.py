from skimage.io import imsave, imread
from skimage.color import label2rgb
import numpy as np
from os.path import join
import os

import numpy as np
from skimage.io import imsave, imread
from skimage.color import label2rgb
import numpy as np
import torch
from mmengine.config import Config
from mmseg.apis import init_model, inference_model



config_path = 'configs/unet/unet-s5-d16_fcn_4xb4-80k_cholecseg8kbinary-448x896.py'
checkpoint_path = 'work_dirs/unet-s5-d16_fcn_4xb4-80k_cholecseg8kbinary-448x896/iter_80000.pth'
img_dir = '/nfs/home/talabi/data/cholecT50-challenge-val/cholect50-challenge-val_formatted/img_dir'
output_dir = 'visualization_results/made_with_cholecseg8k/u_net_tested_on_cholect50-challenge-val'

cfg = Config.fromfile(config_path)

if torch.cuda.is_available():
    model = init_model(cfg, checkpoint_path, device='cuda:0')  
else: 
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg 
    model = init_model(cfg, checkpoint_path, device='cpu') 

for img_name in os.listdir(img_dir):
    input_image_path = join(img_dir, img_name)
    output_mask_path  = join( output_dir, img_name)
    print(input_image_path)
    result = inference_model(model, input_image_path)

    pred_labels = result.pred_sem_seg.data
    pred_labels = pred_labels.detach().cpu().numpy()
    pred_labels = np.squeeze(pred_labels.astype(np.uint8))

    imsave(output_mask_path, (label2rgb(pred_labels, colors=["white"]) * 255.0).astype(np.uint8))
    
