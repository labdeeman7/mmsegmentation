# runai submit train-segmenter-synissparts \
#   -i aicregistry:5000/talabi_mmseg:latest \
#   --gpu 1 \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/synissparts/train_segmenter_synissparts.sh


runai submit train-segmenter-synissparts-complete \
  -i aicregistry:5000/talabi_mmseg:latest \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/synissparts/train_segmenter_synisspartscomplete.sh
