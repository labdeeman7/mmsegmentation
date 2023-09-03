
# runai submit train-segmenter-autolaparo \
#   -i aicregistry:5000/talabi_mmseg:latest \
#   --gpu 1 \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/train_segmenter.sh


runai submit train-segmenter-robustmis \
  -i aicregistry:5000/talabi_mmseg:latest \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/train_segmenter.sh

