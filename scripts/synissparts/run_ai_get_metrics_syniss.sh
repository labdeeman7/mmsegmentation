runai submit syniss-get-metrics-unet \
  -i aicregistry:5000/talabi_post_process:latest \
  --gpu 1 \
  --run-as-user \
  -p talabi \
  -v /nfs:/nfs \
  --command -- bash /nfs/home/talabi/repositories/mmsegmentation/scripts/get_metrics_syniss.sh