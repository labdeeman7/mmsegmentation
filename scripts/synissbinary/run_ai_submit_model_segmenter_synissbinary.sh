# runai submit train-segmenter-synissbinary \
#   -i aicregistry:5000/talabi_mmseg:latest \
#   --gpu 1 \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/synissbinary/train_segmenter_synissbinary.sh

runai submit train-segmenter-synissbinary-complete \
  -i aicregistry:5000/talabi_mmseg:latest \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/synissbinary/train_segmenter_synissbinarycomplete.sh