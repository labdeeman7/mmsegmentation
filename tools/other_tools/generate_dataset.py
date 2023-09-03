import shutil
import skimage
import math, random
from typing import List, Tuple
import numpy as np
import os
import shutil
import cv2
from math import floor
import matplotlib.pyplot as plt
import copy
import json
from os.path import join
from tqdm import tqdm
import argparse

def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))

def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points

"""#  Morphological error """
def generate_erosion_and_dilation_error_image(error_dict, label_img):
    kernel_size = error_dict['morphological_error']['kernel_size'] 
    scale = error_dict['morphological_error']['scale']
    morph_version = error_dict['morphological_error']['morph_version']
    morph_operation = error_dict['morphological_error']['morph_operation']


    #select kernels.
    if morph_version == 1:
      kernel_1 = scale*np.ones((kernel_size, 1), np.uint8)
      kernel_2 = scale*np.ones((1, kernel_size), np.uint8)
      kernel_3 = scale*np.ones((kernel_size, kernel_size), np.uint8)
    elif morph_version == 2:
      kernel_1 = scale*np.ones((kernel_size//2, kernel_size), np.uint8)
      kernel_2 = scale*np.ones((kernel_size, 3), np.uint8)
      kernel_3 = scale*np.ones((kernel_size, kernel_size), np.uint8)  
    elif morph_version == 3:
      kernel_1 = scale*np.ones((2, kernel_size), np.uint8)
      kernel_2 = scale*np.ones((kernel_size//2, kernel_size), np.uint8)
      kernel_3 = scale*np.ones((kernel_size, kernel_size//2), np.uint8)  


    if morph_operation == 'erosion':
      img_new = cv2.erode(label_img, kernel_1, iterations=1)
      img_new = cv2.erode(img_new, kernel_2, iterations=1)
      img_new = cv2.erode(img_new, kernel_3, iterations=1)

    elif morph_operation == 'dilation': 
      img_new = cv2.dilate(label_img, kernel_1, iterations=1)
      img_new = cv2.dilate(img_new, kernel_2, iterations=1)
      img_new = cv2.dilate(img_new, kernel_3, iterations=1)  

    elif morph_operation == 'opening':
      img_new = cv2.morphologyEx(label_img, cv2.MORPH_OPEN, kernel_1)
      img_new = cv2.morphologyEx(img_new, cv2.MORPH_OPEN, kernel_2)
      img_new = cv2.morphologyEx(img_new, cv2.MORPH_OPEN, kernel_3) 

    elif morph_operation == 'closing':
      img_new = cv2.morphologyEx(label_img, cv2.MORPH_CLOSE, kernel_1)
      img_new = cv2.morphologyEx(img_new, cv2.MORPH_CLOSE, kernel_2)
      img_new = cv2.morphologyEx(img_new, cv2.MORPH_CLOSE, kernel_3)   

    return img_new

"""# Instance error """

""" random patch error"""
def add_random_2d_patch_to_label_img(error_dict, label_img, all_vertices):
    no_of_error_patches = error_dict['patch_error']['no_of_error_patches']
    patch_error_types = error_dict['patch_error']['patch_error_types']
    all_classes = error_dict['general_information']['all_classes']
    classes_present = error_dict['original_image_info']['classes_present'] 

    for i in range(no_of_error_patches):
        single_patch_error_type = patch_error_types[i]
        vertices = all_vertices[i]
        if (single_patch_error_type == 'wrong_patch_on_correct_tool'):
            error_patch_possible_classes =  list(set(all_classes) - set(classes_present))
            error_patch_class = random.choice(error_patch_possible_classes)
        elif(single_patch_error_type == 'wrong_patch_in_background'):
            error_patch_class = random.choice(all_classes)
        else:
          raise ValueError('wrong input for error_type')  
        cv2.drawContours(label_img, [vertices], -1, color=(error_patch_class), thickness=cv2.FILLED)
    return label_img


def add_to_json_error_dicts(all_error_dicts, error_dict, created_id):
    all_error_dicts[created_id] = error_dict

def copy_directory(dir_path, new_dir_path):
  shutil.copytree(dir_path, new_dir_path)    

def save_error_dicts_as_json(all_error_dicts, json_save_path):
  with open(json_save_path, 'w') as fp:
      json.dump(all_error_dicts, fp)
  pass

def save_generated_error_img(error_dict, label_img): 
    label_img = (np.copy(label_img)*32).astype(np.uint8)
    save_path = error_dict['save_image_info']['save_path']
    cv2.imwrite(save_path, label_img)


def save_generated_label_as_colour_image(created_id, label_img, error_dict, coloured_save_dir_split):
    img_name = error_dict['original_image_info']['img_name']
    img_path = error_dict['original_image_info']['img_path']

    created_id = str(created_id).zfill(6)   
    created_label_name = f'created_id_{created_id}_{img_name}'
    created_label_path = os.path.join(coloured_save_dir_split, created_label_name)

    img = cv2.imread(img_path)
    generated_label_coloured = get_coloured_labels(label_img)
    image_plus_generated = 0.5*img + 0.5*generated_label_coloured
    cv2.imwrite(created_label_path, image_plus_generated)

def replace_instrument_instance(error_dict, label_img):
    classes_present = error_dict['original_image_info']['classes_present'].copy()
    
    classes_present = list(filter(lambda a: a != 0, classes_present)) # remove background
    classes_absent = error_dict['original_image_info']['classes_absent']
    number_of_error_instances = error_dict['wrong_instance_error']['number_of_error_instances']

    if number_of_error_instances > len(classes_present):
      number_of_error_instances = classes_present
      error_dict['wrong_instance_error']['number_of_error_instances'] = number_of_error_instances

    classes_absent_to_add = random.sample(classes_absent, number_of_error_instances)
    classes_present_to_remove = random.sample(classes_present, number_of_error_instances)

    for i in range(number_of_error_instances):
        label_img[label_img == classes_present_to_remove[i]] = classes_absent_to_add[i] 

    return label_img

def get_coloured_labels(labels): #this needs to be predone befroe we start working. I think generate this in colab.
      palette_list = [(0, 0, 0), 
                (255, 0, 0), 
                (0, 255, 0), 
                (0, 0, 255),
                (0, 255, 255), 
                (255, 0, 255), 
                (255, 255, 0),
                (0, 128, 255)]

      height, width = labels.shape          
      labels_coloured = np.zeros((height, width, 3)).astype(np.uint8)   

      for i_h in range(height):
        for i_w in range(width):
          class_value = labels[i_h, i_w]
          labels_coloured[i_h, i_w] = palette_list[int(class_value)]

      return labels_coloured

def get_center_pixels_for_patch_error(error_dict, label_img):
  patch_error_types = error_dict['patch_error']['patch_error_types'] 
  no_of_error_patches =  error_dict['patch_error']['no_of_error_patches']
  
  center_values = []
  for i in range(no_of_error_patches):
      single_patch_error_type = patch_error_types[i]
    
      if(single_patch_error_type == 'wrong_patch_on_correct_tool'):
          if len(error_dict['original_image_info']['classes_present']) < 2: #only background class present 
              y_idx, x_idx = np.where(label_img==0)
          else: 
              y_idx, x_idx = np.where(label_img!=0)

      elif(single_patch_error_type == 'wrong_patch_in_background'):  
          y_idx, x_idx = np.where(label_img==0)

      else:
          raise ValueError('something is wrong')   

      rand_idx = np.random.choice(np.arange(len(x_idx)))   #randomly choose any element in the x_idx list
      x = x_idx[rand_idx]
      y = y_idx[rand_idx]

      center = [x,y]
      center_values.append(center)

  return center_values    

def generate_vertices_for_patch_error(error_dict, label_img):
  center_values = get_center_pixels_for_patch_error(error_dict, label_img)

  all_vertices = []
  no_of_error_patches =  error_dict['patch_error']['no_of_error_patches'] 
  avg_radii =  error_dict['patch_error']['avg_radius']
  irregularities =  error_dict['patch_error']['irregularity']
  spikinesses = error_dict['patch_error']['spikiness']
  num_vertices = error_dict['patch_error']['num_vertices']
  
  for i in range(no_of_error_patches):
    vertices = generate_polygon(center=center_values[i],
                              avg_radius=avg_radii[i],
                              irregularity=irregularities[i],
                              spikiness=spikinesses[i],
                              num_vertices=num_vertices[i])
    vertices = np.array(vertices).astype(int)
    all_vertices.append(vertices)
  
  return all_vertices

def generate_error_dict(created_id, img_dir, ann_dir, img_file_names, ann_file_names, split, len_dataset, save_dir_split): 
  sample_id = random.randrange(0, len_dataset)



  img_name = img_file_names[sample_id]
  label_name = ann_file_names[sample_id]

  img_path = join(img_dir, img_name)
  label_path = join(ann_dir, label_name)

  label_img = cv2.imread(label_path, 0)
  label_img = label_img.astype(np.uint8)

  classes_present = np.unique(label_img)
  all_classes = [0,1,2,3,4,5,6,7,]
  classes_absent = list(set(all_classes) - set(classes_present))

  #4. get save dir

  created_id = str(created_id).zfill(6)   
  created_label_name = f'created_id_{created_id}_{img_name}'
  created_label_path = os.path.join(save_dir_split, created_label_name)

  #2. choose morphological changes to apply.
  all_morph_operations = ['dilation' , 'erosion', 'opening', 'closing']
  all_morph_versions = [1,2,3]
  morph_operation =  random.choice(all_morph_operations)
  kernel_size = random.randint(5, 20)
  morph_version = random.choice(all_morph_versions)
  scale = random.randint(1,5)

  #3choose parameters for wrong instance error to apply.
  inverse_square_on_classes_present_for_cdf = np.arange(len(classes_present), 0, -1)**2 #inverse cummulative distribution  excluding background
  prob_wrong_instance = inverse_square_on_classes_present_for_cdf/np.sum(inverse_square_on_classes_present_for_cdf)
  number_of_error_instances = np.random.choice(np.arange(0, len(classes_present)), p=prob_wrong_instance)


  #4. choose parameters for patch changes to apply.
  all_patch_error_types = ['wrong_patch_on_correct_tool',
                           'wrong_patch_in_background',]
  no_of_error_patches = random.randint(1, 4)
  patch_error_types = [random.choice(all_patch_error_types) for i in range(no_of_error_patches)] 


  avg_radius = np.random.choice(np.arange(10, 70), size=no_of_error_patches)
  irregularity = np.random.uniform(0, 1, size=no_of_error_patches) 
  spikiness = np.random.uniform(0, 0.15, size=no_of_error_patches) 
  num_vertices = np.random.choice(np.arange(3, 30), size=no_of_error_patches)

  error_dict = {
    'general_information': {
       'all_classes': [0,1,2,3,4,5,6,7,], #all possible classes
      #  'how_often_error_generated': [0.97, 0.97], #we have mophological, patch errors, we need them to be appleid each 95% of tme 
    },
    'original_image_info': {
        'img_name': img_name,
        'label_name': label_name,
        'img_path': img_path, 
        'label_path': label_path, 
        'classes_present': classes_present.tolist(), #classes in the current image, from np.unique - 
        'classes_absent': classes_absent,
    },
    'morphological_error':{
       'morph_operation': morph_operation, #dilation, erosion, opening, closing.
       'morph_version': morph_version, #each morph_operation can comes in 3 versions. select between 1, 2, 3
       'kernel_size': kernel_size, #range from 5 to 20.
       'scale': scale, #range from 1 to 4  
       'error_freq': 0.97   
    },
    'wrong_instance_error': {
        'error_freq': 0.5,
        'number_of_error_instances': int(number_of_error_instances)  
    },
    'patch_error': {
        'no_of_error_patches': no_of_error_patches, #range from 0, 4.
        'patch_error_types': patch_error_types, #'wrong_patch_on_correct_tool', 'wrong_patch_in_background'
        'avg_radius': avg_radius.tolist(), #list dep. error_patches range from 10 to 100
        'irregularity': irregularity.tolist(), #list varying from 0 to 1
        'spikiness': spikiness.tolist(), #list this needs to vary as well 0 to 0.15
        'num_vertices': num_vertices.tolist(), #list vary from 3 to 30.
        'error_freq': 0.97
    },
    'save_image_info':{
        'save_path': created_label_path,
    }
  }

  return error_dict, label_img



def generate_error_dataset(dataset_dir, save_dir = '/content/endovis_generated_dataset/', split='train', number_of_generated_images =100000):

  all_error_dicts = {}
  img_dir = join(dataset_dir, 'img_dir', split)
  ann_dir = join(dataset_dir, 'ann_dir', split)

  save_dir_split = join(save_dir, split, 'generated_labels')
  coloured_save_dir_split = join(save_dir, split, 'coloured_generated_labels')
  img_dir_split_save_dir = join(save_dir, split, 'img_dir')
  ann_dir_split_save_dir = join(save_dir, split, 'ann_dir')
  json_save_path = join(save_dir, split, 'data.json')

  if not os.path.exists(save_dir_split):
      os.makedirs(save_dir_split)
      print(f"Folder '{save_dir_split}' created.")

  if not os.path.exists(coloured_save_dir_split):
      os.makedirs(coloured_save_dir_split)
      print(f"Folder '{coloured_save_dir_split}' created.")    

  img_file_names = sorted([img_name for img_name in os.listdir(img_dir) if img_name.endswith('jpg') or img_name.endswith('png')  ] ) 
  ann_file_names = sorted([ann_name for ann_name in os.listdir(ann_dir) if ann_name.endswith('jpg') or ann_name.endswith('png') ])
  
  
  len_dataset = len(img_file_names)

  for created_id in tqdm(range(number_of_generated_images)):
    error_dict, label_img = generate_error_dict(created_id, img_dir, ann_dir, img_file_names, ann_file_names, split, len_dataset, save_dir_split)

    if np.random.uniform(0, 1) < error_dict['patch_error']['error_freq']:
        all_vertices = generate_vertices_for_patch_error(error_dict, label_img)
        
        label_img = add_random_2d_patch_to_label_img(error_dict, label_img, all_vertices)
    

    #1. morphological
    if np.random.uniform(0, 1) < error_dict['morphological_error']['error_freq']:   
        label_img = generate_erosion_and_dilation_error_image(error_dict, label_img)

        

    #2Wrong instances
    # if np.random.uniform(0, 1) < error_dict['wrong_instance_error']['error_freq']:
    #   label_img = replace_instrument_instance(error_dict, label_img)


    # 3. patch
    save_generated_error_img(error_dict, label_img)
    # save_generated_label_as_colour_image(created_id, label_img, error_dict, coloured_save_dir_split)

    add_to_json_error_dicts(all_error_dicts, error_dict, created_id)

  save_error_dicts_as_json(all_error_dicts, json_save_path)      
  


def parse_args():
    parser = argparse.ArgumentParser(description='create large generated dataset')
    parser.add_argument('--dataset_dir', help='dataset containing labels and images of endovis2017 for now')
    parser.add_argument('--save_dir', help='the dir to save logs and models')
    parser.add_argument('--split', help='train or val is needed here')
    parser.add_argument('--amount', type=int, help='amount of data to be generated')

   
    
    args = parser.parse_args()
    assert args.split == 'train' or args.split == 'val', "only 'train' and 'val' are accepted as "

    return args

def main():
   args = parse_args()
   generate_error_dataset(dataset_dir=args.dataset_dir, save_dir=args.save_dir, split=args.split, number_of_generated_images=args.amount)

if __name__ == '__main__':
    main()


