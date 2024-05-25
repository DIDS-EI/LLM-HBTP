import re
import os
import faiss
import numpy as np
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.llm_client.llms.gpt4 import LLMGPT4

def parse_and_prepare_data(file_path):
    """从文本文件中解析数据，并生成键值对"""
    data = {}
    current_id = None

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.isdigit():
                current_id = line
                data[current_id] = {"Environment": "", "Goals": "", "Optimal Actions": "", "Vital Action Predicates": "", "Vital Objects": "", "Cost": ""}
            else:
                match = re.match(r"(\w+(?: \w+)*):\s*(.*)", line)
                if match and current_id:
                    key, value = match.groups()
                    data[current_id][key.strip()] = value.strip()

    unique_data = {}
    for key, value in data.items():
        # combined_key = f"{value['Environment']}: {value['Goals']}"
        combined_key = f"{value['Goals']}"
        if combined_key not in unique_data:
            unique_data[combined_key] = value
        else:
            if unique_data[combined_key]['Cost'] > value['Cost']:
                unique_data[combined_key] = value

    keys = list(unique_data.keys())
    return keys, unique_data


def extract_embedding_vector(response):
    """从 CreateEmbeddingResponse """
    if response and len(response.data) > 0:
        return response.data[0].embedding
    else:
        raise ValueError("Empty or invalid embedding response.")

def embed_and_store(llm, keys, data, index_path):
    if os.path.exists(index_path) and os.path.exists(index_path.replace(".index", "_metadata.npy")):
        index = faiss.read_index(index_path)
        metadata = np.load(index_path.replace(".index", "_metadata.npy"), allow_pickle=True).tolist()
    else:
        index = None
        metadata = []

    new_embeddings = np.array([extract_embedding_vector(llm.embedding(key)) for key in keys], dtype='float32')

    dim = new_embeddings.shape[1]
    print(dim)
    if index is None:
        index = faiss.IndexFlatL2(dim)

    for i, key in enumerate(keys):
        duplicate_index = -1
        for j, item in enumerate(metadata):
            if item['key'] == key:
                duplicate_index = j
                break

        new_value = data[key]
        if duplicate_index != -1:
            if metadata[duplicate_index]['value']['Cost'] > new_value['Cost']:
                metadata[duplicate_index]['value'] = new_value
                index.reconstruct(duplicate_index, new_embeddings[i])
        else:
            index.add(new_embeddings[i].reshape(1, -1))
            metadata.append({"key": key, "value": new_value})

    faiss.write_index(index, index_path)
    np.save(index_path.replace(".index", "_metadata.npy"), metadata)


def search_similar(index_path, llm, environment, goal, top_n=3):
    index = faiss.read_index(index_path)
    metadata = np.load(index_path.replace(".index", "_metadata.npy"), allow_pickle=True)

    query = f"{environment}: {goal}"
    query_embedding = np.array([extract_embedding_vector(llm.embedding(query))], dtype='float32')
    distances, indices = index.search(query_embedding, top_n)

    results = [{"id": idx, "distance": dist, "key": metadata[idx]['key'], "value": metadata[idx]['value']}
               for dist, idx in zip(distances[0], indices[0])]
    return results

def check_index_exists(index_path):
    index_file = index_path
    metadata_file = index_path.replace(".index", "_metadata.npy")
    return os.path.exists(index_file) and os.path.exists(metadata_file)

def search_nearest_examples(index_path, llm, goal, top_n=5):
    index = faiss.read_index(index_path)
    metadata = np.load(index_path.replace(".index", "_metadata.npy"), allow_pickle=True)

    query_embedding = np.array([extract_embedding_vector(llm.embedding(goal))], dtype='float32')
    distances, indices = index.search(query_embedding, top_n)

    nearest_examples = [metadata[idx] for idx in indices[0]]
    return nearest_examples, distances

def add_data_entry(index_path, llm, environment, goal, optimal_actions, vital_action_predicates, vital_objects, cost):
    index = faiss.read_index(index_path)
    metadata = np.load(index_path.replace(".index", "_metadata.npy"), allow_pickle=True)

    new_key = f"{goal}"
    new_value = {
        "Environment": environment,
        "Goals": goal,
        "Optimal Actions": optimal_actions,
        "Vital Action Predicates": vital_action_predicates,
        "Vital Objects": vital_objects,
        "Cost": cost
    }

    new_embedding = np.array([extract_embedding_vector(llm.embedding(new_key))], dtype='float32')

    duplicate_index = -1
    for i, item in enumerate(metadata):
        if item['key'] == new_key:
            duplicate_index = i
            break

    if duplicate_index != -1:
        if metadata[duplicate_index]['value']['Cost'] > cost:
            metadata[duplicate_index]['value'] = new_value
            index.reconstruct(duplicate_index, new_embedding[0])
    else:
        index.add(new_embedding)
        metadata = np.append(metadata, [{"key": new_key, "value": new_value}])

    faiss.write_index(index, index_path)
    np.save(index_path.replace(".index", "_metadata.npy"), metadata)

def write_metadata_to_txt(index_path, output_path):
    metadata = np.load(index_path.replace(".index", "_metadata.npy"), allow_pickle=True)

    with open(output_path, 'w', encoding='utf-8') as file:
        for i, item in enumerate(metadata):
            value = item['value']
            file.write(f"{i + 1}\n")
            file.write(f"Environment:{value['Environment']}\n")
            file.write(f"Instruction:{value.get('Instruction', '')}\n")
            file.write(f"Goals:{value['Goals']}\n")
            file.write(f"Optimal Actions:{value['Optimal Actions']}\n")
            file.write(f"Vital Action Predicates:{value['Vital Action Predicates']}\n")
            file.write(f"Vital Objects:{value['Vital Objects']}\n")
            file.write("\n")

def add_to_database(llm,env, goals, priority_act_ls, key_predicates, key_objects, database_index_path, cost):
    new_environment = "1"
    new_goal=goals
    new_optimal_actions = ', '.join(priority_act_ls)
    new_vital_action_predicates = ', '.join(key_predicates)
    new_vital_objects = ', '.join(key_objects)
    add_data_entry(database_index_path, llm, new_environment, new_goal, new_optimal_actions,
                   new_vital_action_predicates, new_vital_objects, cost)
    print(f"\033[95mAdd the current data to the vector database\033[0m")

def create_empty_index(dimension, index_path):
    index = faiss.IndexFlatL2(dimension)
    faiss.write_index(index, index_path)
    print(f"Empty index created and saved to {index_path}")

if __name__ == '__main__':
    llm = LLMGPT3()

    filename = "0"
    file_path = f"{ROOT_PATH}/../test/VD_3_EXP/DATABASE/{filename}.txt"
    index_path = f"{ROOT_PATH}/../test/VD_3_EXP/DATABASE/{filename}_goal_vectors.index"
    output_path = f"{ROOT_PATH}/../test/VD_3_EXP/DATABASE/DATABASE_{filename}_metadata.txt"

    should_rebuild_index = True
    if should_rebuild_index or not check_index_exists(index_path):
        keys, data = parse_and_prepare_data(file_path)
        embed_and_store(llm, keys, data, index_path)

    write_metadata_to_txt(index_path, output_path)

    for key in data:
        print(f"ID: {key}")
        print(f"Environment: {data[key]['Environment']}")
        print(f"Goals: {data[key]['Goals']}")
        print(f"Optimal Actions: {data[key]['Optimal Actions']}")
        print(f"Vital Action Predicates: {data[key]['Vital Action Predicates']}")
        print(f"Vital Objects: {data[key]['Vital Objects']}")
        print(f"Cost: {data[key]['Vital Objects']}")
        print("-----------")

    environment = "1"
    goal = "IsClean_magazine & IsCut_apple & IsPlugged_toaster"
    results = search_similar(index_path, llm, environment, goal)

    for result in results:
        record_id = result['id']
        distance = result['distance']
        key = result['key']
        value = result['value']
        print(f"Record ID: {record_id}, Distance: {distance}")
        print(f"Key: {key}, Value: {value}\n")

    print("=============== 添加新数据后 ================")

    new_environment = "01"
    new_goal = "IsClean_magazine & IsCut_apple & IsPlugged_toaster"
    new_optimal_actions = "Walk_rag, RightGrab_rag, Walk_magazine, Wipe_magazine, Walk_toaster, PlugIn_toaster, RightPutIn_rag_toaster, Walk_kitchenknife, RightGrab_kitchenknife, Walk_apple, LeftGrab_apple, Cut_apple"
    new_vital_action_predicates = "Walk, RightGrab, Wipe, PlugIn, RightPutIn, LeftGrab, Cut"
    new_vital_objects = "rag, magazine, toaster, kitchenknife, apple"

