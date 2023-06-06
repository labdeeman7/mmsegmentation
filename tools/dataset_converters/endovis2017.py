# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from os.path import join
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Endovis2017 dataset to mmsegmentation format')
    parser.add_argument('--dataset_path', help='path of endovis_2017_processed_folder')
    parser.add_argument('--new_dataset_path', help='path of the temporary directory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    DATASET_DIR = args.new_dataset_path
    DATASET_NEW_DIR = args.new_dataset_path

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
            if split == 'test':
                new_split_folder = 'val'
            else:  
                new_split_folder = 'train'

            for img_file in sorted(os.listdir(img_folder)):
                shutil.copy2(join(img_folder, img_file), join(DATASET_NEW_DIR, 'img_dir', new_split_folder, seq_folder_name + '_' +img_file))

            for ann_file in sorted(os.listdir(ann_folder)):
                shutil.copy2(join(ann_folder, ann_file), join(DATASET_NEW_DIR, 'ann_dir', new_split_folder, seq_folder_name + '_' +ann_file))  

if __name__ == '__main__':
    main()
