runai submit train-unet-synissparts-ce-orunet-loss \
  -i aicregistry:5000/talabi_mmseg:latest \
  --gpu 1 \
  -p talabi \
  -v /nfs:/nfs \
  --backoff-limit 0 \
  --large-shm \
  --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/synisspartsunet/train_unet_synisspartscompleteCEorunet.sh