#!/usr/bin/python

cd /nfs/home/talabi/repositories/mmsegmentation/

echo "evaluating ..."


echo "metrics ..................."
python test_get_metrics.py --true_path '/nfs/home/talabi/data/Endovis_challenges_2023/syniss/syniss_rem_wrong_labels/parts_ann_dir/val' \
                            --pred_path '/nfs/home/talabi/repositories/mmsegmentation/work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteCE-448x896/pred_labels' \
                            --syniss_challenge_name 'parts' 

echo "completed metrics ce ..."

python test_get_metrics.py --true_path '/nfs/home/talabi/data/Endovis_challenges_2023/syniss/syniss_rem_wrong_labels/parts_ann_dir/val' \
                            --pred_path '/nfs/home/talabi/repositories/mmsegmentation/work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteOHEM-448x896/pred_labels' \
                            --syniss_challenge_name 'parts' 

echo "completed metrics ohem ..."

python test_get_metrics.py --true_path '/nfs/home/talabi/data/Endovis_challenges_2023/syniss/syniss_rem_wrong_labels/parts_ann_dir/val' \
                            --pred_path '/nfs/home/talabi/repositories/mmsegmentation/work_dirs/unet-s5-d16_fcn_4xb4-40k_synisspartscompleteWCE-448x896/pred_labels' \
                            --syniss_challenge_name 'parts' 

echo "completed metrics wce ..."





