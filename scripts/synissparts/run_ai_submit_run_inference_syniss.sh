# runai submit run-inference-syniss-parts-unet \
#   -i aicregistry:5000/talabi_mmseg:latest \
#   --gpu 1 \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/get_inference_syniss.sh

# runai submit run-inference-syniss-on-train \
#   -i aicregistry:5000/talabi_mmseg:latest \
#   --gpu 1 \
#   -p talabi \
#   -v /nfs:/nfs \
#   --backoff-limit 0 \
#   --large-shm \
#   --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/get_inference_syniss.sh

runai submit run-inference-syniss \
  -i aicregistry:5000/talabi_mmseg:latest \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/get_inference_syniss.sh