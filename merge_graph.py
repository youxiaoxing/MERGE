import os
# 设置 CORENLP_HOME 环境变量指向 CoreNLP 目录
os.environ["CORENLP_HOME"] = r"stanford-corenlp-4.5.7" 
import json
import torch
import numpy as np
from pymongo import MongoClient
from stanza.server import CoreNLPClient
from openai import OpenAI
import ast
from tqdm import tqdm

### 合并子图
def merge_subgraphs(subgraphs, relations=None):
    merged_graph = {
        "nodes": set(),
        "edges": set()
    }
    
    def add_edge(source, target, rel):
        merged_graph["nodes"].add(source)
        merged_graph["nodes"].add(target)
        merged_graph["nodes"].add(rel)
        merged_graph["edges"].add((source, rel))
        merged_graph["edges"].add((rel, target))
    
    # 合并所有子图
    for subgraph in subgraphs:
        if isinstance(subgraph, str):
            subgraph = json.loads(subgraph)
        
        merged_graph["nodes"].update(subgraph["node"])
        
        for edge, rel in zip(subgraph["edge"], subgraph["relationship"]):
            add_edge(edge[0], edge[1], rel)
    
    # 添加子图之间的关系（如果提供）
    if relations:
        for source, target, rel in relations:
            add_edge(source, target, rel)
    
    # 将集合转换为排序后的列表
    merged_graph["nodes"] = sorted(list(merged_graph["nodes"]))
    merged_graph["edges"] = sorted(list(merged_graph["edges"]))
    
    return merged_graph

### 将生成后的图转为GAT可以支持的形式
def convert_to_gat_format(merged_graph):
    # 1. 创建节点到索引的映射
    node_to_idx = {node: idx for idx, node in enumerate(merged_graph["nodes"])}
    
    # 2. 创建边索引
    edge_index = []
    for edge in merged_graph["edges"]:
        source, target = edge
        edge_index.append([node_to_idx[source], node_to_idx[target]])
    
    # 3. 创建节点特征（使用简单的 one-hot 编码）
    num_nodes = len(node_to_idx)
    node_features = torch.eye(num_nodes)
    
    # 4. 将边索引转换为 PyTorch 张量
    edge_index = torch.LongTensor(edge_index).t().contiguous()

    return node_features, edge_index

def read_mongodb_split(db):
    results = db.images_position.find({
         "$and": [
            { "final_sentences": { "$ne": None } },
            { "final_sentences": { "$ne": [] } },
            { "final_sentences": { "$ne": "" } },
            { "final_sentences": { "$exists": True } }
        ],
         "new_graph": {"$exists": False},
    }, {
        "_id": 1, 
        "final_sentences": 1
    })
    ret_data = []
    for item in results:
        item["text1"] = " ".join(item["final_sentences"])
        ret_data.append(item)
    ret_data.reverse()
    return ret_data

### 读取实体集合，获得所有的子图
def read_mongodb_entity(db, collection):
    results = db[collection].find({"graph": {"$exists": True, "$ne": ""}}, {"name": 1, "graph": 1})
    ret_data = {}
    for item in results:
        ret_data[item["name"]] = item["graph"]
    
    return ret_data

### 识别text1中能够链接到wikipedia的实体
def entitylink(client, text, entity_dict):
    ann = client.annotate(text)
    sentence = ann.sentence[0]
    ret_data = []
    for mention in sentence.mentions:
        if mention.wikipediaEntity:
            # print(f"entity type: {mention.entityType}, entity: {mention.wikipediaEntity}, entity text: {mention.entityMentionText}")
            if mention.wikipediaEntity in entity_dict:
                ret_data.append({"entity": mention.wikipediaEntity, "entity_text": mention.entityMentionText})
    
    return ret_data


'''
### 对识别出来的实体生成关系
relations = [
    ("Philippe_Dauman", "Division of Cowan", "visited"),
    ("Eugene, Oregon", "Viacom", "has office"),
    ("Anne Aly", "CBS_Corporation", "interviewed by")
]
'''

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.getenv("VLLM_API_KEY", "-"))
### 生成抽取出来的实体之间的关系
def graph_gen(entityList, text):
    chat_completion = client.chat.completions.create(
        model = "llama3.1-70b-awq",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": get_prompt(entityList, text)},
        ]
    )
    # print(chat_completion)
    # 提取生成的内容
    if chat_completion.choices and len(chat_completion.choices) > 0:
        content = chat_completion.choices[0].message.content
        return content.replace("```json", "").replace("```", "").replace("\n", "")

def get_prompt(entityList, text):
    entity_text = ""
    for entity in entityList:
        wikipediaEntity = entity["entity"]
        entity_text = entity_text + wikipediaEntity + ", "
    prompt = f'''Help me identify relationships between entities from the text below. For each pair of entities (A,B), only include ONE relationship in either A->B or B->A direction, not both. The relationship description should not exceed three words.

The final data should be in Python list format, where each item is a tuple containing entity a, entity b, and their relationship.
Remember, you can only return the requested format and do not return any other content beyond that.
For example:
[
("Philippe_Dauman", "Division of Cowan", "visited"),
("Eugene, Oregon", "Viacom", "has office"),
("Anne Aly", "CBS_Corporation", "interviewed by")
]
Entities :
{entity_text}

Text:
{text}
'''

    return prompt

### 获取实体子图，并将维基百科实体转换为原文中提到的文本
### 同时将所有字母变为小写字母
def get_subgraph(entityList, totalEntity):
    subgraphList = []
    for entity in entityList:
        wikipediaEntity = entity["entity"]
        entityMentionText = entity["entity_text"]
        subgraph = json.loads(totalEntity[wikipediaEntity])
        if not subgraph:
            continue
        if "node" not in subgraph and "nodes" in subgraph:
            subgraph["node"] = subgraph["nodes"]
        if "edge" not in subgraph and "edges" in subgraph:
            subgraph["edge"] = subgraph["edges"]
        if "relationship" not in subgraph and "relationships" in subgraph:
            subgraph["relationship"] = subgraph["relationships"]
        if isinstance(subgraph, list):
            subgraph = subgraph[0]
        # wikipediaEntity = wikipediaEntity.lower()
        # entityMentionText = entityMentionText.lower()
        # subgraph["node"] = [entityMentionText.replace("_", " ").lower() if node.lower() == wikipediaEntity else node.replace("_", " ").lower() for node in subgraph["node"]]
        # subgraph["edge"] = [[entityMentionText.replace("_", " ").lower() if edge.lower() == wikipediaEntity else edge.replace("_", " ").lower() for edge in edges] for edges in subgraph["edge"]]
        # subgraph["relationship"] = [relationship.replace("_", " ").lower() for relationship in subgraph["relationship"]]
        wikipediaEntity = str(wikipediaEntity).lower()
        entityMentionText = str(entityMentionText).lower()
        subgraph["node"] = [str(entityMentionText).replace("_", " ").lower() if str(node).lower() == wikipediaEntity else str(node).replace("_", " ").lower() for node in subgraph["node"]]
        subgraph["edge"] = [[str(entityMentionText).replace("_", " ").lower() if str(edge).lower() == wikipediaEntity else str(edge).replace("_", " ").lower() for edge in edges] for edges in subgraph["edge"]]
        subgraph["relationship"] = [str(relationship).replace("_", " ").lower() for relationship in subgraph["relationship"]]
        subgraphList.append(subgraph)
        
    return subgraphList

### 将relations转换为文中出现文本形式，同时转为小写
def clean_relations(entityList, relations):
    if not relations or not entityList:
        return relations
        
    # 创建entity映射字典，提高查找效率
    entity_map = {entity["entity"].lower(): entity["entity_text"].lower() 
                  for entity in entityList}
    
    # 创建新的relations列表而不是直接修改
    cleaned_relations = []
    for relation in relations:
        cleaned_relation = []
        for item in relation:
            # 将item转换为小写进行比较
            item_lower = item.lower()
            # 如果在映射字典中找到匹配项，则使用对应的entity_text
            cleaned_item = entity_map.get(item_lower, item)
            cleaned_relation.append(cleaned_item.replace("_", " "))
        cleaned_relations.append(cleaned_relation)
    
    return cleaned_relations
            
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def process_item(item, client, totalEntity, db):
    # try:
    text = item["text1"]
    
    if not text:
        return

    entityList = entitylink(client, text, totalEntity)
    if len(entityList) > 1:
        relations = graph_gen(entityList, text)
        relations = ast.literal_eval(relations)
        relations = clean_relations(entityList, relations)
    else:
        relations = None

    subgraphList = get_subgraph(entityList, totalEntity)
    merged_graph = merge_subgraphs(subgraphList, relations=relations)
    print(merged_graph)
    db.images_position.find_one_and_update({"_id": item["_id"]}, {"$set": {"new_graph": json.dumps(merged_graph)}})
    # except Exception as e:
    #     print("error========", e)

def main():
    client = MongoClient('10.4.121.10', 27018)
    db = client.nytimes
    goodnews_db = client.goodnews
    all_text = read_mongodb_split(db)
    # all_text.reverse()
    print("读取文本结束")
    personEntity = read_mongodb_entity(goodnews_db, "wikipedia_summary")
    print("读取人名实体结束")
    otherEntity = read_mongodb_entity(goodnews_db, "wikipedia_others_summary")
    print("读取其他实体结束")
    totalEntity = personEntity | otherEntity
    
    # 创建一个全局的CoreNLPClient
    with CoreNLPClient(
        annotators=['entitylink'],
        timeout=30000,
        memory='8G',
        endpoint="http://localhost:9001") as nlp_client:
        # for item in all_text:
        #     process_item(item, nlp_client, totalEntity, db)
        all_text.reverse()
        # 使用线程池执行任务
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_item, item, nlp_client, totalEntity, db) 
                      for item in all_text]
            # 使用tqdm显示进度
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                pass

if __name__ == "__main__":
    main()