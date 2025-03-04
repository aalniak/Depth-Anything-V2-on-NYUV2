source ~/anaconda3/etc/profile.d/conda.sh
conda activate Depth-Anything-V2
python nyu_test.py  --encoder vitl   --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth   --max-depth 20  --input-size 518 
