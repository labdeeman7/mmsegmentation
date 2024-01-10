# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from os.path import join
import numpy as np
from skimage.io import imsave, imread
from skimage.color import label2rgb
import numpy as np

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmseg.apis import init_model, inference_model, show_result_pyplot


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('--config_path', help='train config file path')
    parser.add_argument('--checkpoint_path', help='checkpoint file')
    parser.add_argument('--img_path', help=('path to image on which prediction is done'))   
    parser.add_argument('--save_dir', help=('directory to save the prediction'))
    args = parser.parse_args()
   
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config_path)
    cfg.load_from = args.checkpoint_path


    
    #tta augmentation
    cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
    cfg.tta_model.module = cfg.model
    cfg.model = cfg.tta_model


    model = init_model(args.config_path, args.checkpoint_path, device='cuda:0') 
    result = inference_model(model, args.img_path)

    pred_labels = result.pred_sem_seg.data
    pred_labels = pred_labels.detach().cpu().numpy()
    pred_labels = pred_labels.astype(np.uint8)
    pred_labels = np.squeeze(pred_labels)

    img_base_name = args.img_path.split('/')[-1] + '.npy'
    save_ious_path = join(args.save_dir, img_base_name)  

    np.save(save_ious_path, pred_labels)
    
    print(pred_labels)
    print(pred_labels.dtype)

    gt_rgb = imread(args.img_path)
    class_colors = {
        1: [255, 214, 0],
        2: [138, 0, 0],
        3: [49, 205, 49]
    }
    for k, v in class_colors.items():
        roi = (
            (gt_rgb[:,:,0] == v[0]) &
            (gt_rgb[:,:,1] == v[1]) & 
            (gt_rgb[:,:,2] == v[2])
        )
        pred_labels[roi] = k

    # ================================================
    # END PARTICIPANT CODE
    # End of your code. DO NOT modify the code beyond this point.
    # ================================================

    # convert the numpy array to RGB image and save it to file
    imsave(save_ious_path + '.png', (label2rgb(pred_labels, colors=["gold", "darkred", "limegreen"]) * 255.0).astype(np.uint8))

if __name__ == '__main__':
    main()
