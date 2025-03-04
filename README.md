# Depth-Anything-V2-on-NYUV2
This repository contains a Python script built on top of [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2).

Running this repository requires a correctly set-up Unidepth environment.
In order to run the scripts:   
1- Clone the repository [here](https://github.com/DepthAnything/Depth-Anything-V2) and create environment / install requirements as described there.  
2- Download the respective metric depth models and put them under ./metric_depth/checkpoints
3- Put the scripts under ./metric_depth folder.
4- Change the environment name in .sh files to the one you set (If you went with the suggested name Unidepth, this step is not required).  
5- Run the respective script using:  
```bash
bash kitti_test.sh
```

It worths noting that the .sh script gives the model some priori information, such as max_depth, and which model to use... There are two suggested models, that are Hypersim (for indoor with suggested max_depth=20) and VKITTI (for outdoor with suggested max_depth=80). For this very test project, script variables are as follows:    
input_size = 518,  
model=Fine-tuned on Hypersim,    
max_depth=20     


## About the code
Once you run the script, it will try to download the dataset under /home/{your_username}/nyu_cache. All my scripts use the cache there, so if you already have it please move the dataset to there.  
  
It is further possible to change the dataset sampling by:  

```python
dataset = load_dataset("sayakpaul/nyu_depth_v2", split="validation[:654]", cache_dir=home_dir+"/nyu_cache")
dataset = dataset.select(range(0, 654, 6))  # Sample every 6th data in dataset
```



## Acknowledgment
This work is based on [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2), developed by [Binyi Kang](https://github.com/bingykang).    
Dataset used can be found at [here](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2).

