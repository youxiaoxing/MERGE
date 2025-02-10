# MERGE
This is the code for "Knowledge Completes the Vision: A Multi-modal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning". We will gradually perfect it.

- [x] Upload training and evaluation code
- [ ] Clean and upload preprocessing code
- [ ] Release detailed implementation process

# Preprocess

# Installation

# Data

# Trainning (GoodNews)
To train MERGE on the GoodNews dataset, the first step is to modify `configs/config.py`. Inside the config file, `Annotatio` parameter specifies the path to the training JSON file as described in the Data section, and `base_dir` parameter specifies the path to the image directory. Then run below command to train our code:
```
python caption_train.py
```

# Trainning (NYTimes800k)
To train MERGE on the NYTimes800k dataset, the first step is to modify `configs/config_nytimes.py`. Inside the config file, `Annotatio` parameter specifies the path to the training JSON file as described in the Data section, and `base_dir` parameter specifies the path to the image directory. Then run below command to train our code:
```
python caption_train_nytimes.py
```
# Evaluation
To evaluate the trained checkpoint, set the ckpt_file parameter in configs/config_nytimes.py to your checkpoint file path. Then run:
```
python caption_train.py
```