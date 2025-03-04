source ~/anaconda3/etc/profile.d/conda.sh
conda activate Depth-Anything-V2
python kitti_test.py  --encoder vitl   --load-from checkpoints/depth_anything_v2_metric_vkitti_vitl.pth   --max-depth 80  --input-size 518 
