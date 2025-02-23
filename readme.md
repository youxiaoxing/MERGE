# MERGE
This is the code for "Knowledge Completes the Vision: A Multi-modal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning". 

# Data
Please download GoodNews and NYTimes800k datasets from the official repo of [Transform-and-Tell](https://github.com/alasdairtran/transform-and-tell).

# Preprocess

# Installation
To train our code, please run:
```
conda env create -n instructBlip --file requirements.txt
conda activate instructBlip
```

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

To compute the Caption Generation metrics, modify `compute_score.py` by setting the paths for your result JSON file and train_data.json, then run:
```
python compute_score.py
```

To compute the Named Entity Recognition metrics, set your result JSON file path in `compute_score_entity.py`, then run:
```
python compute_score_entity.py
```