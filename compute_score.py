import os
import json
from tqdm import tqdm
from collections import defaultdict
import re
import numpy as np
import types
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from unidecode import unidecode

def process_string(text):
    text = text.replace("\n", " ")
    text = text.strip()
    text = unidecode(text)
    return text

def _stat(self, hypothesis_str, reference_list):
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
    score_line = ' ||| '.join(
        ('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    score_line = score_line.replace('\n', '').replace('\r', '')
    self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()

def cal_caption_score_from_dict(result_dict):
    bleu_scorer = BleuScorer(n=4)
    rouge_scorer = Rouge()
    rouge_scores = []
    cider_scorer = CiderScorer(n=4, sigma=6.0)
    meteor_scorer = Meteor()
    meteor_scorer._stat = types.MethodType(_stat, meteor_scorer)

    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    meteor_scores = []

    for sample in tqdm(result_dict):
        # Remove punctuation
        caption = re.sub(r'[^\w\s]', '', sample["ref_caption"])
        generation = re.sub(r'[^\w\s]', '', sample["caption"])

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1

    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()

    return {
        'bleu': blue_score,
        'rouge': rouge_score,
        'cider': cider_score,
        'meteor': meteor_score
    }

def get_data_file(folder_path):
    import os
    import json
    
    # Get all json files in the folder
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
 
    print(json_files)
    all_data = []
    data_dict = {}
    
    # Load reference captions once
    with open("goodnews.json", "r") as f:
        original_data = json.load(f)
        val_data = original_data["test"]
        for item in val_data:
            data_dict[item["id"]] = item["conversations"][1]["value"]

    # Process each json file
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Process each item in the file
        for item in data:
            image_id = item["image_id"] 
            item["ref_caption"] = data_dict[image_id]
            if "</s>" in item["caption"]:
                item["caption"] = process_string(item["caption"].split("</s>")[0])
            item["ref_caption"] = process_string(item["ref_caption"].replace("</s>", ""))
            
        all_data.append(data)
        
    return all_data

all_data = get_data_file("result")
for data in all_data:
    scores = cal_caption_score_from_dict(data)
    print(scores)