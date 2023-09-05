import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import torch
from os.path import join
import time

class ConfMatrix(object): #😉 Confusion matrix I assume
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.mat = None #🙋‍♂️ What is mat? I guess the matrix. It should be a num_class x num_class matrix.
        self.ious = torch.zeros((0,num_classes), device=device)
        self.dices = torch.zeros((0,num_classes), device=device)
        self.filenames = []
        

    def get_acc_iou_dice(self, mat):   
        h = mat.float() 
        acc = torch.diag(h).sum() / h.sum() #😉 Accuracy is the correct predictions/every prediction. 
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))  #😉 iu is tp / tp +fp +fn for each class. 
        dice = (2* torch.diag(h))/ (h.sum(1) + h.sum(0) ) #😉 dice is 2*tp / 2*tp +fp +fn for each class.  
        return acc, iou, dice       

    def update(self, pred, target, file_name): #😉 The input is a flattened array with the labels.

        self.filenames.append(file_name)

        target =  torch.Tensor(target.flatten())
        if len(pred.shape) == 4:
            pred = torch.Tensor(pred.argmax(1).flatten())
        else:
            pred = torch.Tensor(pred.flatten())   
        n = self.num_classes
        if self.mat is None: #😉 At the beginning we make a 12x12 array. 
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device) #😉 called it.nxn
        with torch.no_grad(): 
            k = (target >= 0) & (target < n) #this is for removing -1 which nyu used for unknown.
            inds = n * target[k].to(torch.int64) + pred[k]
            current_mat = torch.bincount(inds, minlength=n ** 2).reshape(n, n)

            #get accuracy, iou and dice for each image and save it. 
            _, img_iou, img_dice = self.get_acc_iou_dice(current_mat)
            self.ious = torch.vstack((self.ious, img_iou))
            self.dices = torch.vstack((self.dices, img_dice))

            self.mat += current_mat

     

    def get_metrics(self):
        '''
        #😉 Gives the mean iou, the average dice, the iou, the dice, the accuracy, Note F1 is Dice and Jacard is iou.  
        Returns a dictionary.
        ''' 
        acc, iou, dice =  self.get_acc_iou_dice(self.mat)        
        #😉 The nan's are important. You should leave them as they are. It means there was no occurence, and it should be skipped when calculating mean iou. 
        
        iu_no_nan = np.array([x.item() for x in iou if str(x.item()) != "nan"])    
        dice_no_nan = np.array([x.item() for x in dice if str(x.item()) != "nan"])  
        
        metrics = {
            "filenames": self.filenames,
            "mIoU": np.mean(iu_no_nan),
            "dice_avg": np.mean(dice_no_nan), 
            "iou": iou.cpu().numpy() ,
            "dice": dice.cpu().numpy(),
            "acc": acc.cpu().numpy(),
            "all_ious": self.ious.cpu().numpy() ,
            "all_dices": self.dices.cpu().numpy(),            
        }

        return metrics


if __name__ == '__main__':
    print('we entered main')
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--true_path',
        type=str,
        help='path where train images with ground truth are located')
    arg('--pred_path',
        type=str,
        help='path with predictions')
    arg('--syniss_challenge_name',
        type=str,
        help='syniss challenge name it can be parts or binary')
    arg('--save_ious_path',
        type=str,
        default='/nfs/home/talabi/repositories/mmsegmentation/wor_dirs_all_ious',
        help='path to sabe ious')     
    arg('--vis', action='store_true')
    args = parser.parse_args()

    if args.syniss_challenge_name == 'parts':
        class_name_list = [
            'background', 'shaft', 'wrist','jaws'
        ]
    elif args.syniss_challenge_name == 'binary':
        class_name_list = [
            'background', 'instrument',
        ]
    else:
        raise ValueError('we only support syniss')

    conf_mat_val = ConfMatrix(len(class_name_list), 'cpu')

    #for removing the first two.
    for file_name in tqdm(os.listdir(args.true_path)):
        true_file_path = os.path.join(args.true_path, file_name)
        pred_file_path = os.path.join(args.pred_path,file_name)

        y_true = cv2.imread(true_file_path, 0).astype(np.uint8)
        y_pred = cv2.imread(pred_file_path, 0).astype(np.uint8)
        
        conf_mat_val.update(torch.from_numpy(y_pred) , torch.from_numpy(y_true), file_name)

    metrics = conf_mat_val.get_metrics()
    mIoU = metrics["mIoU"]
    dice_avg = metrics["dice_avg"]
    acc = metrics["acc"]
    iu = metrics["iou"]
    dice = metrics["dice"]
    filenames = metrics["filenames"]

    filenames = np.array(filenames)
    all_ious = metrics["all_ious"]

    filenames_with_all_ious = np.hstack((np.expand_dims(filenames, axis=1), all_ious))

    exp_name = args.pred_path.split('/')[-2] + '.npy'
    save_ious_path = os.path.join( args.save_ious_path, exp_name)
    np.save(save_ious_path, filenames_with_all_ious)

    for c, class_name in enumerate(class_name_list):
        print(f'Instrument Class: {class_name} IoU={iu[c]}, dice={dice[c]}')

    print(f'mIoU is {mIoU}')
    print(f'dice_avg is {dice_avg}')
    print(f'acc for each class is {acc}')
