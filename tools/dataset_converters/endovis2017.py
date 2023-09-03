# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from os.path import join
import shutil
from pathlib import Path
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Endovis2017 dataset to mmsegmentation format')
    parser.add_argument('--dataset_path', help='path of endovis_2017_processed_folder')
    parser.add_argument('--new_dataset_path', help='path of the temporary directory')
    parser.add_argument('--resize_scale_percent', help='scale in percentage to resize image')
    parser.add_argument("--img_size", nargs="*", type=int, default=[512, 640],  # default if nothing is provided
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    DATASET_DIR = args.dataset_path
    DATASET_NEW_DIR = args.new_dataset_path
    dim = (args.img_size[1], args.img_size[0])
    print(dim)

    splits = ['cropped_train','cropped_test']
  
    new_img_dir_train = join(DATASET_NEW_DIR, 'img_dir', 'train')
    if not os.path.exists(new_img_dir_train):
        os.makedirs(new_img_dir_train) 

    new_img_dir_val = join(DATASET_NEW_DIR, 'img_dir', 'val')
    if not os.path.exists(new_img_dir_val):
        os.makedirs(new_img_dir_val) 

    new_ann_dir_train = join(DATASET_NEW_DIR, 'ann_dir', 'train')
    if not os.path.exists(new_ann_dir_train):
        os.makedirs(new_ann_dir_train)    

    new_ann_dir_val = join(DATASET_NEW_DIR, 'ann_dir', 'val')
    if not os.path.exists(new_ann_dir_val):
        os.makedirs(new_ann_dir_val) 


    new_ann_dir_train_vis = join(DATASET_NEW_DIR, 'ann_dir_vis', 'train')
    if not os.path.exists(new_ann_dir_train_vis):
        os.makedirs(new_ann_dir_train_vis)      

    new_ann_dir_val_vis = join(DATASET_NEW_DIR, 'ann_dir_vis', 'val')
    if not os.path.exists(new_ann_dir_val_vis):
        os.makedirs(new_ann_dir_val_vis) 

    for split in splits:
        data_split_folder = join(DATASET_DIR, split)
        seq_folders = [join(data_split_folder, seq_folder) for seq_folder in sorted(os.listdir(data_split_folder)) if os.path.isdir(join(data_split_folder, seq_folder))] 
        print(seq_folders)
        for seq_folder in seq_folders:
            seq_folder_name = Path(seq_folder).name
            # img_folder = join(seq_folder, 'left_frames')
            # ann_folder = join(seq_folder, 'ground_truth')
            img_folder = join(seq_folder, 'images')
            ann_folder = join(seq_folder, 'instruments_masks')
            if split == 'cropped_train':
                new_split_folder = 'train'
            elif split == 'cropped_test':  
                new_split_folder = 'val'

            for img_file in sorted(os.listdir(img_folder)):
                src_img_path = join(img_folder, img_file)
                new_img_file = img_file.split(".")[0] + '.png'
                dest_img_path =   join(DATASET_NEW_DIR, 'img_dir', new_split_folder, seq_folder_name + '_' +new_img_file) 

                img = cv2.imread(src_img_path)
                resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite(dest_img_path, resized_img)
                

            for ann_file in sorted(os.listdir(ann_folder)):
                src_annotation_path = join(ann_folder, ann_file)
                new_ann_file = ann_file.split(".")[0] + '.png'
                destination_annotation_path = join(DATASET_NEW_DIR, 'ann_dir', new_split_folder, seq_folder_name + '_' +new_ann_file)
                destination_annotation_path_vis = join(DATASET_NEW_DIR, 'ann_dir_vis', new_split_folder, seq_folder_name + '_' +new_ann_file)

                ann = cv2.imread(src_annotation_path, 0)
                resized_ann = cv2.resize(ann, dim, interpolation = cv2.INTER_NEAREST)              
                cv2.imwrite(destination_annotation_path_vis, resized_ann)

                resized_ann = (resized_ann/32).astype(np.uint8)
                cv2.imwrite(destination_annotation_path, resized_ann)

if __name__ == '__main__':
    main()
