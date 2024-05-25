import re
import os
import faiss
import numpy as np
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3


def parse_and_prepare_data(file_path):
    """从文本文件中解析数据，并生成键值对"""
    data = {}
    current_id = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.isdigit():
                current_id = line
                data[current_id] = {"Environment": "", "Instruction": "", "Goals": "", "Actions": "", "Key_Predicate": "", "Key_Objects": ""}
            else:
                match = re.match(r"(\w+):\s*(.*)", line)
                if match and current_id:
                    key, value = match.groups()
                    data[current_id][key] = value

    # 将 Environment 和 Instruction 组合成键
    keys = [f"{entry['Environment']}: {entry['Instruction']}" for entry in data.values()]
    return keys, data

def extract_embedding_vector(response):
    """从 CreateEmbeddingResponse 对象中提取嵌入向量"""
    if response and len(response.data) > 0:
        return response.data[0].embedding
    else:
        raise ValueError("Empty or invalid embedding response.")

def embed_and_store(llm, keys, data, index_path):
    """生成嵌入并存储在向量数据库中，同时保存元数据"""
    embeddings = np.array([extract_embedding_vector(llm.embedding(key)) for key in keys], dtype='float32')

    # 将嵌入和索引保存到 Faiss 索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    # 保存其他相关数据，包括每个 key 及其对应的 value
    metadata = [{"key": key, "value": data[key]} for key in data.keys()]
    np.save(index_path.replace(".index", "_metadata.npy"), metadata)

def search_similar(index_path, llm, environment, instruction, top_n=3):
    """搜索与给定环境和指令组合最相似的记录，并输出详细信息"""
    index = faiss.read_index(index_path)
    metadata = np.load(index_path.replace(".index", "_metadata.npy"), allow_pickle=True)

    query = f"{environment}: {instruction}"
    query_embedding = np.array([extract_embedding_vector(llm.embedding(query))], dtype='float32')
    distances, indices = index.search(query_embedding, top_n)

    results = [{"id": idx, "distance": dist, "key": metadata[idx]['key'], "value": metadata[idx]['value']}
               for dist, idx in zip(distances[0], indices[0])]
    return results


def check_index_exists(index_path):
    """检查索引文件和元数据文件是否存在"""
    index_file = index_path
    metadata_file = index_path.replace(".index", "_metadata.npy")
    return os.path.exists(index_file) and os.path.exists(metadata_file)


def search_nearest_examples(index_path, llm, instruction, top_n=5):
    """检索最接近给定指令的示例"""
    index = faiss.read_index(index_path)
    metadata = np.load(index_path.replace(".index", "_metadata.npy"), allow_pickle=True)

    # 获取嵌入向量
    query_embedding = np.array([extract_embedding_vector(llm.embedding(instruction))], dtype='float32')
    distances, indices = index.search(query_embedding, top_n)

    # 返回最接近的示例
    nearest_examples = [metadata[idx] for idx in indices[0]]
    return nearest_examples,distances





if __name__ == '__main__':
    # 假设 llm 是已经初始化的嵌入模型对象
    llm = LLMGPT3()

    # 示例路径和布尔标志
    file_path = f"{ROOT_PATH}/../test/dataset/database_cys_5.txt"
    index_path = f"{ROOT_PATH}/../test/dataset/env_instruction_vectors.index"
    should_rebuild_index = False  # 如果为 True，则重建数据库

    # 检查文件存在或决定是否重建
    if should_rebuild_index or not check_index_exists(index_path):
        keys, data = parse_and_prepare_data(file_path)
        embed_and_store(llm, keys, data, index_path)

    # 使用特定环境和指令进行查询
    environment = "4"
    instruction = "Put the apples on the corner of fridge."
    results = search_similar(index_path, llm, environment, instruction)

    # 输出详细检索结果
    for result in results:
        record_id = result['id']
        distance = result['distance']
        key = result['key']
        value = result['value']
        print(f"Record ID: {record_id}, Distance: {distance}")
        print(f"Key: {key}, Value: {value}\n")
