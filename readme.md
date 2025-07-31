# MERGE
This is the code for "Knowledge Completes the Vision: A Multi-modal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning". 

# Data
Please download GoodNews and NYTimes800k datasets from the official repo of [Transform-and-Tell](https://github.com/alasdairtran/transform-and-tell). We use JSON format as the train file, the structure is:
```json
{
    "train": [
        "id": "4fd275a48eb7c8105d83599c_0",
            "image": "4fd275a48eb7c8105d83599c_0.jpg",
            "graph_str": "constructed background knowledge graph",
            "conversations": [
                {
                    "from": "human",
                    "value": "Please generate the informative and brief news image caption with the associated news article summary and other information.\nNews Summary: {summary}. \nRelated Sentences: {related sentences}. \nInitial reference caption: {hypo-caption} \nPossible entities in image: {entities} \nCpation : "
                },
                {
                    "from": "gpt",
                    "value": "Prime Minister Gordon Brown announced new security measures in a statement to the House of Commons on Wednesday."
                }
            ]
    ],
    "val": [],
    "test": []
}
```

# Preprocess
## Entity-central Multimodal Knowledge Base Construction
### Exists Datasets
- **Cross-Age Celebrity Dataset (CACD)**: The [dataset](https://bcsiriuschen.github.io/CARC/) contains 163,446 images from 2,000 celebrities collected from the Internet.
- **IMDb-WIKI**: The [dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) comprises a total of 523,051 celebrity face images, with 460,723 images of 20,284 celebrities sourced from IMDb and an additional 62,328 images collected from Wikipedia.
- **IMDb-Face** : The [dataset](https://github.com/fwang91/IMDb-Face) contains about 1.7 million faces, 59k identities, which is manually cleaned from 2.0 million raw images. All images are obtained from the IMDb website.
- **VGG-Face2**: The [dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) contains 3.31 million images of 9131 subjects (identities), with an average of 362.6 images for each subject. Images are downloaded from Google Image Search and have large variations in pose, age, illumination, ethnicity and profession (e.g. actors, athletes, politicians). 

In the above datasets, we only retain 5 images for each celebrity.

### Supplement Dataset
1. To supplement the entity database, we perform [entity linking](https://stanfordnlp.github.io/CoreNLP/) to connect the recognized named entities in the GoodNews dataset and NYTiems800k dataset to Wikipedia entries, while also applying [Qwen2.5-32b](https://github.com/QwenLM/Qwen2.5) to enrich the database.
2. We use [Wikipedia-api](https://github.com/martin-majlis/Wikipedia-API) to get the background knowledge of entities. Then, the LLM and prompts described in our paper are used for background knowledge subgraph construction.
3. To scrapy the images of entities, we first scrape the recognized named entities' corresponding Wikipedia page image as entity image. The code can be referenced as follows:
```python
def get_wiki_image_url(entity, max_retries=5):
    url = f"https://en.wikipedia.org/wiki/{entity}"
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0'
    }
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            html = BeautifulSoup(r.content, 'html.parser')
            og_image_tags = html.find_all('meta', property='og:image')
            if og_image_tags:
                smallest_image_tag = og_image_tags[-1]
                image_url = smallest_image_tag.get('content')
                width_tag = smallest_image_tag.find_next('meta', property='og:image:width')
                height_tag = smallest_image_tag.find_next('meta', property='og:image:height')
                width = width_tag.get('content') if width_tag else None
                height = height_tag.get('content') if height_tag else None
                return image_url, width, height
            return None, None, None
        except RequestException:
            if attempt < max_retries - 1:
                time.sleep(0.1)

    logging.error(f"Failed to get image URL for entity {entity} after {max_retries} attempts")
    return None, None, None
```
4. Then we use [icrawler](https://github.com/hellock/icrawler) to scrape five images from Google Search.
5. For Person-type entities' images, we use [InsightFace's buffalo_l](https://github.com/deepinsight/insightface/tree/master/python-package) to detect faces and extract face embeddings. For other visible entities' images, we use CLIP to extract visual embeddings.

### Hypothesis Caption-guided Multimodal Alignment 
We used [InternVL2-Llama3-76B-AWQ](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B-AWQ) to perform the cross-modal alignment as described in the paper. Before training the model, we first performed HCMA on our selected training dataset, validation dataset and test dataset. Then we only used the selected sentences, hypo-captions and summaries as the context input for InstructBlip.

### Background Knowledge Graph Construction
To get the background knowledge graph, we provide the code in `merge_graph.py`, where `final_sentences` contains our selected sentences using HCMA and `new_graph` is our background knowledge subgraph.

# Installation
To train our code, please run:
```
conda env create -n instructBlip --file requirements.txt
conda activate instructBlip
```

# Training (GoodNews)
To train MERGE on the GoodNews dataset, the first step is to modify `configs/config.py`. Inside the config file, `Annotatio` parameter specifies the path to the training JSON file as described in the Data section, and `base_dir` parameter specifies the path to the image directory. Then run below command to train our code:
```
python caption_train.py
```

# Training (NYTimes800k)
To train MERGE on the NYTimes800k dataset, the first step is to modify `configs/config_nytimes.py`. Inside the config file, `Annotatio` parameter specifies the path to the training JSON file as described in the Data section, and `base_dir` parameter specifies the path to the image directory. Then run below command to train our code:
```
python caption_train_nytimes.py
```

# Training (Visual News)
To train MERGE on the Visual News dataset, the first step is to modify `configs/config_visualnews.py`. Inside the config file, `Annotatio` parameter specifies the path to the training JSON file as described in the Data section, and `base_dir` parameter specifies the path to the image directory. Then run below command to train our code:
```
python caption_train_visualnews.py
```

# Evaluation
To evaluate the trained checkpoint, set the ckpt_file parameter in configs/config_nytimes.py to your checkpoint file path. Then run:
```
python caption_train.py --validate
```

To compute the Caption Generation metrics, modify `compute_score.py` by setting the paths for your result JSON file and train_data.json, then run:
```
python compute_score.py
```

To compute the Named Entity Recognition metrics, set the path for your result JSON file in `compute_score_entity.py`, then run:
```
python compute_score_entity.py
```