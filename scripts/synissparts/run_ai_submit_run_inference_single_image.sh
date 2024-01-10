
runai submit run-inference-single-image \
  -i aicregistry:5000/talabi_mmseg:latest \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/get_inference_single_image.sh