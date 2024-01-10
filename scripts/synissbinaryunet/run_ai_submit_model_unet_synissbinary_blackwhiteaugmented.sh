runai submit train-unet-synissbinary-gray-augmented \
  -i aicregistry:5000/talabi_mmseg:latest \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/synissbinaryunet/train_unet_synissbinaryblackwhiteaugmented.sh