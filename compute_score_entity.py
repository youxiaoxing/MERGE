import os
import json
from tqdm import tqdm
import spacy
from collections import defaultdict
import re
import numpy as np
import types
from unidecode import unidecode

def process_string(text):
    text = text.replace("\n", " ")
    text = text.strip()
    text = unidecode(text)
    return text

def contain_entity_by_gtent(entities, target, gt_first=True):
    for ent in entities:
        if gt_first:
            if ent == target['text']:
                return True
        else:
            if ent['text'] == target:
                return True
    return False

def get_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text.lower(),
            'label': ent.label_,
            'tokens': [{'text': tok.text.lower(), 'pos': tok.pos_} for tok in ent],
        })
    return entities

def compute_entities_by_gtent(caption_entities, caption_persons, caption_orgs, caption_gpes, gen_entities, c):

    c['n_caption_ents'] += len(caption_entities)
    c['n_gen_ents'] += len(gen_entities)
    for ent in gen_entities:
        if contain_entity_by_gtent(caption_entities, ent, gt_first=True):
            c['n_gen_ent_matches'] += 1
    for ent in caption_entities:
        if contain_entity_by_gtent(gen_entities, ent, gt_first=False):
            c['n_caption_ent_matches'] += 1

    gen_persons = [e for e in gen_entities if e['label'] == 'PERSON']
    c['n_caption_persons'] += len(caption_persons)
    c['n_gen_persons'] += len(gen_persons)
    for ent in gen_persons:
        if contain_entity_by_gtent(caption_persons, ent, gt_first=True):
            c['n_gen_person_matches'] += 1
    for ent in caption_persons:
        if contain_entity_by_gtent(gen_persons, ent, gt_first=False):
            c['n_caption_person_matches'] += 1

    gen_orgs = [e for e in gen_entities if e['label'] == 'ORG']
    c['n_caption_orgs'] += len(caption_orgs)
    c['n_gen_orgs'] += len(gen_orgs)
    for ent in gen_orgs:
        if contain_entity_by_gtent(caption_orgs, ent, gt_first=True):
            c['n_gen_orgs_matches'] += 1
    for ent in caption_orgs:
        if contain_entity_by_gtent(gen_orgs, ent, gt_first=False):
            c['n_caption_orgs_matches'] += 1

    gen_gpes = [e for e in gen_entities if e['label'] == 'GPE']
    c['n_caption_gpes'] += len(caption_gpes)
    c['n_gen_gpes'] += len(gen_gpes)
    for ent in gen_gpes:
        if contain_entity_by_gtent(caption_gpes, ent, gt_first=True):
            c['n_gen_gpes_matches'] += 1
    for ent in caption_gpes:
        if contain_entity_by_gtent(gen_gpes, ent, gt_first=False):
            c['n_caption_gpes_matches'] += 1
    return c

def evaluate_entity_by_gtent(ref_dict, gen_dict):
    nlp = spacy.load("en_core_web_lg")
    ent_counter = defaultdict(int)

    for ref_caption, gen_caption in tqdm(zip(ref_dict, gen_dict)):
        gen_cap = nlp(gen_caption)
        ref_cap = nlp(ref_caption)
        gen_entities = get_entities(gen_cap)
        ref_entities = get_entities(ref_cap)
        
        # 提取text字段，转换为字符串列表
        caption_entities = [e['text'] for e in ref_entities]  # 所有实体的文本
        caption_persons = [e['text'] for e in ref_entities if e['label'] == 'PERSON']  # 人名实体的文本
        caption_orgs = [e['text'] for e in ref_entities if e['label'] == 'ORG']  # 组织实体的文本
        caption_gpes = [e['text'] for e in ref_entities if e['label'] == 'GPE']  # 地理位置实体的文本
        
        compute_entities_by_gtent(caption_entities, caption_persons, caption_orgs, caption_gpes, gen_entities, ent_counter)    
    
    # counter = 0
    # gtent_keys = list(gtent_dict.keys())
    # for key,sample in tqdm(output_dict.items()):
    #     if key not in ["bleu", "other metrics"]:
    #         gen_cap = nlp(sample["gen"])
    #         caption_entities = gtent_dict[gtent_keys[counter]]["ner_cap"]
    #         caption_persons = gtent_dict[gtent_keys[counter]]["names_cap"]
    #         caption_orgs = gtent_dict[gtent_keys[counter]]["org_cap"]
    #         caption_gpes = gtent_dict[gtent_keys[counter]]["gpe_cap"]

    #         gen_entities = get_entities(gen_cap)
            
    #         compute_entities_by_gtent(caption_entities, caption_persons, caption_orgs, caption_gpes, gen_entities, ent_counter)
    #         counter += 1

    entity_results = {
        'Entity all - recall': {
            'count': ent_counter['n_caption_ent_matches'],
            'total': ent_counter['n_caption_ents'],
            'percentage': ent_counter['n_caption_ent_matches'] / ent_counter['n_caption_ents'],
        },
        'Entity all - precision': {
            'count': ent_counter['n_gen_ent_matches'],
            'total': ent_counter['n_gen_ents'],
            'percentage': ent_counter['n_gen_ent_matches'] / ent_counter['n_gen_ents'],
        },
        'Entity person (by full name) - recall': {
            'count': ent_counter['n_caption_person_matches'],
            'total': ent_counter['n_caption_persons'],
            'percentage': ent_counter['n_caption_person_matches'] / ent_counter['n_caption_persons'] if ent_counter['n_caption_persons']> 0 else 0,
        },
        'Entity person (by full name) - precision': {
            'count': ent_counter['n_gen_person_matches'],
            'total': ent_counter['n_gen_persons'],
            'percentage': ent_counter['n_gen_person_matches'] / ent_counter['n_gen_persons'] if ent_counter['n_caption_persons']> 0 else 0,
        },
        'Entity GPE - recall': {
            'count': ent_counter['n_caption_gpes_matches'],
            'total': ent_counter['n_caption_gpes'],
            'percentage': ent_counter['n_caption_gpes_matches'] / ent_counter['n_caption_gpes'],
        },
        'Entity GPE - precision': {
            'count': ent_counter['n_gen_gpes_matches'],
            'total': ent_counter['n_gen_gpes'],
            'percentage': ent_counter['n_gen_gpes_matches'] / ent_counter['n_gen_gpes'],
        },
        'Entity ORG - recall': {
            'count': ent_counter['n_caption_orgs_matches'],
            'total': ent_counter['n_caption_orgs'],
            'percentage': ent_counter['n_caption_orgs_matches'] / ent_counter['n_caption_orgs'],
        },
        'Entity ORG - precision': {
            'count': ent_counter['n_gen_orgs_matches'],
            'total': ent_counter['n_gen_orgs'],
            'percentage': ent_counter['n_gen_orgs_matches'] / ent_counter['n_gen_orgs'],
        },
        }
    
    # output_dict.update(entity_results)
    return entity_results

def main(file_name):
    ref_caption = []
    gen_caption = []
    with open(file_name, "r") as f:
        data = json.load(f)
        
        for item in data:
            if "</s>" in item["caption"]:
                item["caption"] = item["caption"].split("</s>")[0]
                item["ref_caption"] = item["ref_caption"].split("</s>")[0]
            ref_caption.append(process_string(item["ref_caption"]))
            gen_caption.append(process_string(item["caption"]))
    print(file_name)
    print(evaluate_entity_by_gtent(ref_caption, gen_caption))

main("result.json")