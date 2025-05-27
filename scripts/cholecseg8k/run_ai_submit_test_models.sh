# runai submit test-autolaparo \
#   -i aicregistry:5000/talabi_mmseg:latest \
#   --gpu 1 \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/test_model.sh


# runai submit test-unet-on-choect50val \
#   -i aicregistry:5000/talabi_mmseg:latest \
#   --gpu 1 \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/cholecseg8k/test_model.sh


runai submit test-unet-on-cholectissue \
  -i aicregistry:5000/talabi_mmseg:latest \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/cholecseg8k/test_model.sh
