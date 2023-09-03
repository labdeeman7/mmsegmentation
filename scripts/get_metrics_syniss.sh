#!/usr/bin/python

cd /nfs/home/talabi/repositories/mmsegmentation/

echo "evaluating ..."


echo "metrics ..................."
python test_get_metrics.py --true_path '/nfs/home/talabi/data/Endovis_challenges_2023/syniss/syniss_complete/parts_ann_dir/train' \
                            --pred_path '/nfs/home/talabi/repositories/mmsegmentation/work_dirs/unet-s5-d16_fcn_4xb4-80k_synisspartscompletetrain-448x896/pred_labels' \
                            --syniss_challenge_name 'parts' 

echo "completed metrics parts ..."




