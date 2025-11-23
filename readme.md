# MERGE: Multimodal Entity-aware Retrieval-Augmented Generation for News Image Captioning

This repository contains the official implementation of **"Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning"**  (Accepted as **AAAI 2026 Oral**).

## 🌟 Overview

News image captioning goes beyond describing visual content: it demands **accurate identification of people, events, locations, and contextual details** by jointly reasoning over the news article, the image, and external knowledge. However, existing approaches struggle with three fundamental challenges:

#### **1. Incomplete Information Coverage**
Many important entities in news images (e.g., people or landmarks) are **not mentioned in the article**, causing models to miss crucial facts.

#### **2. Weak Cross-modal Alignment**
Models often fail to correctly align fine-grained visual elements (e.g., cars, clothing, numbers) with textual narrative or numerical details.

#### **3. Suboptimal Visual-Entity Grounding**
Identifying **who is who** in photos with multiple people remains challenging, especially when entities look similar or are unseen during training.

## 🎯 What MERGE Contributes

To tackle these issues, **MERGE** introduces the **first Multimodal Entity-aware Retrieval-Augmented Generation (RAG) framework** designed specifically for news image captioning. MERGE integrates visual, textual, and structured knowledge into a unified pipeline with **three key innovations**:

#### **1. Entity-centric Multimodal Knowledge Base (EMKB)**
A large multimodal repository that unifies:
- Named entities (persons, objects, locations)
- Face and object images  
- Background textual knowledge  
- Structured knowledge subgraphs  

EMKB enables MERGE to retrieve **missing contextual information**, dramatically improving caption completeness.

#### **2. Hypothesis Caption-guided Multimodal Alignment (HCMA)**
A three-stage Chain-of-Thought (CoT) prompting pipeline that:
1. Generates a "hypothesis caption"
2. Retrieves highly relevant article sentences
3. Produces a concise global summary  

This improves **fine-grained cross-modal alignment** between visual cues and textual descriptions.

#### **3. Retrieval-driven Multimodal Knowledge Integration (RMKI)**
A dynamic retrieval process that:
- Matches entities using face/visual embeddings
- Constructs a background knowledge graph from EMKB 

RMKI enables "precise visual-entity grounding", especially in photos with multiple people or subtle differences.

## 📈 Performance Highlights

MERGE achieves **state-of-the-art results** on GoodNews and NYTimes800k, and shows **exceptional generalization** on the unseen Visual News dataset:

- **+6.84 CIDEr** (GoodNews) and **+1.16 CIDEr** (NYTimes800k) over strong baselines
- **+4.14 F1** (GoodNews) and **+2.64 F1** (NYTimes800k) for named entity recognition
- **+20.17 CIDEr** and **+6.22 F1** on Visual News (generalization test)

These gains reflect MERGE's ability to combine multimodal retrieval and structured reasoning to produce **accurate, complete, and journalistically informative captions**.

## 📦 Data

Please download the **GoodNews** and **NYTimes800k** datasets from the official repository of  
[Transform-and-Tell](https://github.com/alasdairtran/transform-and-tell).

We use the following JSON format for training:

```json
{
    "train": [
        {
            "id": "4fd275a48eb7c8105d83599c_0",
            "image": "4fd275a48eb7c8105d83599c_0.jpg",
            "graph_str": "constructed background knowledge graph",
            "conversations": [
                {
                    "from": "human",
                    "value": "Please generate the informative and brief news image caption with the associated news article summary and other information.\nNews Summary: {summary}. \nRelated Sentences: {related sentences}. \nInitial reference caption: {hypo-caption} \nPossible entities in image: {entities} \nCaption: "
                },
                {
                    "from": "gpt",
                    "value": "Prime Minister Gordon Brown announced new security measures..."
                }
            ]
        }
    ],
    "val": [],
    "test": []
}
```

## 🛠️ Preprocess

### 1. Entity-centric Multimodal Knowledge Base (EMKB)

#### Exists Datasets
- **Cross-Age Celebrity Dataset (CACD)**: The [dataset](https://bcsiriuschen.github.io/CARC/) contains 163,446 images from 2,000 celebrities collected from the Internet.
- **IMDb-WIKI**: The [dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) comprises a total of 523,051 celebrity face images, with 460,723 images of 20,284 celebrities sourced from IMDb and an additional 62,328 images collected from Wikipedia.
- **IMDb-Face** : The [dataset](https://github.com/fwang91/IMDb-Face) contains about 1.7 million faces, 59k identities, which is manually cleaned from 2.0 million raw images. All images are obtained from the IMDb website.
- **VGG-Face2**: The [dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) contains 3.31 million images of 9131 subjects (identities), with an average of 362.6 images for each subject. Images are downloaded from Google Image Search and have large variations in pose, age, illumination, ethnicity and profession (e.g. actors, athletes, politicians). 

In the above datasets, we only retain 5 images for each celebrity.

#### Supplement Dataset
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

### 2. Hypothesis Caption-guided Multimodal Alignment 
We used [InternVL2-Llama3-76B-AWQ](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B-AWQ) to perform the cross-modal alignment as described in the paper. Before training the model, we first performed HCMA on our selected training dataset, validation dataset and test dataset. Then we only used the selected sentences, hypo-captions and summaries as the context input for InstructBlip.

### 3. Background Knowledge Graph Construction
To get the background knowledge graph, we provide the code in `merge_graph.py`: 
- `final_sentences`: selected sentences from HCMA
- `new_graph`: constructed background knowledge subgraph

## 🚀 Installation
To train our code, please run:
```bash
conda env create -n instructBlip --file requirements.txt
conda activate instructBlip
```

## 🏋️ Training

### GoodNews

To train MERGE on the GoodNews dataset, the first step is to modify `configs/config.py`. Inside the config file, `Annotatio` parameter specifies the path to the training JSON file as described in the Data section, and `base_dir` parameter specifies the path to the image directory. Then run below command to train our code:
```bash
python caption_train.py
```

### NYTimes800k

To train MERGE on the NYTimes800k dataset, the first step is to modify `configs/config_nytimes.py`. Inside the config file, `Annotatio` parameter specifies the path to the training JSON file as described in the Data section, and `base_dir` parameter specifies the path to the image directory. Then run below command to train our code:
```bash
python caption_train_nytimes.py
```

### Visual News

To train MERGE on the Visual News dataset, the first step is to modify `configs/config_visualnews.py`. Inside the config file, `Annotatio` parameter specifies the path to the training JSON file as described in the Data section, and `base_dir` parameter specifies the path to the image directory. Then run below command to train our code:
```bash
python caption_train_visualnews.py
```

## 🧪 Evaluation

### Caption Generation
To evaluate the trained checkpoint, set the ckpt_file parameter in configs/config_nytimes.py to your checkpoint file path. Then run:
```bash
python caption_train.py --validate
```

### Caption Metrics

To compute the Caption Generation metrics, modify `compute_score.py` by setting the paths for your result JSON file and train_data.json, then run:
```bash
python compute_score.py
```

### Named Entity Recognition Metrics

To compute the Named Entity Recognition metrics, set the path for your result JSON file in `compute_score_entity.py`, then run:
```bash
python compute_score_entity.py
```

## 📮 Contact

For questions or issues, feel free to open a GitHub issue or contact the authors. Thank you and have fun!
