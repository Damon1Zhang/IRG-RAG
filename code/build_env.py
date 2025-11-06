import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "path to huggingface cache path"

from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from py2neo import Graph, Node
from tqdm import tqdm
import json

def build_triples_embedding(location, model_path, device):
    client = MilvusClient(location)

    embedding_fn = SentenceTransformer(model_path, device=device)
    dim = 2560

    schema = client.create_schema()
    schema.add_field(field_name="uuid", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="triple", datatype=DataType.JSON)
    schema.add_field(field_name="triple_str", datatype=DataType.VARCHAR, max_length=4096)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_name="embedding_index",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    collection_name = "triples"
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    batch_size = 256

    for subject in ["pathogen", "toxin", "check", "neopathy", "symptom", "drug", "cure", "department", "food"]:
        print(f"creating embedding for relations between disease and {subject}")

        if subject == "food":
            data = []

            with open("./do_eat.json", 'r', encoding='utf-8') as f:
                relations = json.load(f)
            for head, tail in relations.items():
                for entity in tail:
                    data.append(({"头实体": head, "头实体类型": "Disease", "关系类型": "do_eat", "尾实体": entity, "尾实体类型": "Food", "描述信息": ''}, json.dumps({"头实体": head, "关系类型": "宜吃", "尾实体": entity, "描述信息": ''}, ensure_ascii=False)))

            with open("./not_eat.json", 'r', encoding='utf-8') as f:
                relations = json.load(f)
            for head, tail in relations.items():
                for entity in tail:
                    data.append(({"头实体": head, "头实体类型": "Disease", "关系类型": "not_eat", "尾实体": entity, "尾实体类型": "Food", "描述信息": ''}, json.dumps({"头实体": head, "关系类型": "忌吃", "尾实体": entity, "描述信息": ''}, ensure_ascii=False)))

            data = [data[i: i+batch_size] for i in range(0, len(data), batch_size)]
            for item in tqdm(data):
                triples_str = [ele[1] for ele in item]
                embeddings = embedding_fn.encode(triples_str)
                insert_item = [{"triple": item[i][0], "triple_str": item[i][1], "embedding": embeddings[i]} for i in range(0, len(item))]
                client.insert(collection_name=collection_name, data=insert_item)
        else:
            if subject in ["pathogen", "toxin"]:
                relation_name = "cause"
                relation_name_ch = "引起"
            elif subject in ["drug", "cure"]:
                relation_name = "cure_way"
                relation_name_ch = "治疗"
            elif subject == "check":
                relation_name = "check_way"
                relation_name_ch = "诊断检查"
            elif subject == "neopathy":
                relation_name = "accompany_with"
                relation_name_ch = "并发症"
            elif subject == "symptom":
                relation_name = "has_symptom"
                relation_name_ch = "症状"
            elif subject == "department":
                relation_name = "belong_to"
                relation_name_ch = "就诊科室"

            data = []

            with open(f"./{subject}_disease_rel.json", 'r', encoding="utf-8") as f:
                relations = json.load(f)
            if subject in ["check", "cure", "drug", "symptom"]:
                for head, tail in relations.items():
                    for entity in tail:
                        data.append(({"头实体": head, "头实体类型": "Disease", "关系类型": relation_name, "尾实体": entity[subject], "尾实体类型": subject.capitalize(), "描述信息": json.dumps(entity['detail'], ensure_ascii=False) if isinstance(entity['detail'], list) else entity['detail']}, json.dumps({"头实体": head, "关系类型": relation_name_ch, "尾实体": entity[subject], "描述信息": json.dumps(entity['detail'], ensure_ascii=False) if isinstance(entity['detail'], list) else entity['detail']}, ensure_ascii=False))) # symptom存在detail是列表的情况
            elif subject in ["pathogen", "toxin"]:
                for tail, head in relations.items():
                    for entity in head:
                        data.append(({"头实体": entity[subject], "头实体类型": subject.capitalize(), "关系类型": relation_name, "尾实体": tail, "尾实体类型": "Disease", "描述信息": entity['detail']}, json.dumps({"头实体": entity[subject], "关系类型": relation_name_ch, "尾实体": tail, "描述信息": entity['detail']}, ensure_ascii=False)))
            else:
                for head, tail in relations.items():
                    for entity in tail:
                        data.append(({"头实体": head, "头实体类型": "Disease", "关系类型": relation_name, "尾实体": entity, "尾实体类型": 'Disease' if subject=='neopathy' else subject.capitalize(), "描述信息": ''}, json.dumps({"头实体": head, "关系类型": relation_name_ch, "尾实体": entity, "描述信息": ''}, ensure_ascii=False)))

            data = [data[i: i+batch_size] for i in range(0, len(data), batch_size)]
            for item in tqdm(data):
                triples_str = [ele[1] for ele in item]
                embeddings = embedding_fn.encode(triples_str)
                insert_item = [{"triple": item[i][0], "triple_str": item[i][1], "embedding": embeddings[i]} for i in range(0, len(item))]
                client.insert(collection_name=collection_name, data=insert_item)

    print("creating embedding for relations between pathogen and toxin")

    data = []

    with open("./toxin_pathogen_rel.json", 'r', encoding='utf-8') as f:
        relations = json.load(f)
    for head, tail in relations.items():
        for entity in tail:
            data.append(({"头实体": head, "头实体类型": 'Pathogen', "关系类型": "secrete", "尾实体": entity, "尾实体类型": 'Toxin', "描述信息": ''}, json.dumps({"头实体": head, "关系类型": "代谢产物", "尾实体": entity, "描述信息": ''}, ensure_ascii=False)))

    data = [data[i: i+batch_size] for i in range(0, len(data), batch_size)]
    for item in tqdm(data):
        triples_str = [ele[1] for ele in item]
        embeddings = embedding_fn.encode(triples_str)
        insert_item = [{"triple": item[i][0], "triple_str": item[i][1], "embedding": embeddings[i]} for i in range(0, len(item))]
        client.insert(collection_name=collection_name, data=insert_item)


def create_diseases_nodes(g, disease_infos, additional_attr):
    if additional_attr:
        for disease_dict in tqdm(disease_infos):
            node = Node("Disease", name=disease_dict['name'], susceptible_population=disease_dict['susceptible_population'],
                        mode_of_infection=disease_dict['mode_of_infection'], treatment_cycle=disease_dict['treatment_cycle'],
                        cure_rate=disease_dict['cure_rate'], prevent=disease_dict['prevent'], nursing_info=disease_dict['nursing_info'],
                        dietary_advice=disease_dict['dietary_advice'])
            g.create(node)
    else:
        for disease in tqdm(disease_infos):
            node = Node("Disease", name=disease, susceptible_population='', mode_of_infection='',
                        treatment_cycle='', cure_rate='', prevent='', nursing_info='', dietary_advice='')
            g.create(node)
    return

def create_node(g, label, nodes):
    for node_name in tqdm(nodes):
        node = Node(label, name=node_name)
        g.create(node)
    return

def create_relationship(g, start_node_type, end_node_type, rel_type, edges, rel_name, additional_attr):
    if additional_attr:
        for edge in tqdm(edges):
            query = '''match (p:%s), (q:%s) where p.name=$from and q.name=$to create (p)-[r:%s{name:$rel_name, detail:$detail}]->(q)''' % (
                start_node_type, end_node_type, rel_type)
            params = {"from": edge[0], "to": edge[1], "rel_name": rel_name, "detail": edge[2]}
            try:
                g.run(query, parameters=params)
            except Exception as e:
                print(e)
    else:
        for edge in tqdm(edges):
            query = '''match (p:%s), (q:%s) where p.name=$from and q.name=$to create (p)-[r:%s{name:$rel_name}]->(q)''' % (
                start_node_type, end_node_type, rel_type)
            params = {"from": edge[0], "to": edge[1], "rel_name": rel_name}
            try:
                g.run(query, parameters=params)
            except Exception as e:
                print(e)
    return

def create_index(g, label):
    g.run(f"CREATE INDEX {label}_name IF NOT EXISTS FOR (n:{label}) ON (n.name)")

def build_graph(url, user, password):
    g = Graph(url, auth=(user, password))

    print("inserting disease(source)")
    with open('./disease(source).json', 'r', encoding='utf-8') as f:
        disease_infos = json.load(f)
    create_diseases_nodes(g, disease_infos, True)
    print("inserting disease")
    with open('./disease.json', 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    create_diseases_nodes(g, nodes, False)
    for label in ["pathogen", "toxin", "check", "symptom", "drug", "cure", "department", "food"]:
        print(f"inserting {label}")
        with open(f'./{label}.json', 'r', encoding='utf-8') as f:
            nodes = json.load(f)
        create_node(g, label.capitalize(), nodes)

    print("creating relationship between pathogen and toxin")
    with open('./toxin_pathogen_rel.json', 'r', encoding='utf-8') as f:
        tmp = json.load(f)
    edges = []
    for pathogen, toxins in tmp.items():
        for toxin in toxins:
            edges.append([pathogen, toxin])
    create_relationship(g, "Pathogen", "Toxin", "secrete", edges, "代谢产物", False)
    for label in ["pathogen", "toxin", "check", "neopathy", "symptom", "drug", "cure", "department", "food"]:
        print(f"creating relationship between disease and {label}")
        if label == "food":
            with open('./do_eat.json', 'r', encoding='utf-8') as f:
                tmp = json.load(f)
            edges = []
            for disease, foods in tmp.items():
                for food in foods:
                    edges.append([disease, food])
            create_relationship(g, "Disease", "Food", "do_eat", edges, "宜吃", False)

            with open('./not_eat.json', 'r', encoding='utf-8') as f:
                tmp = json.load(f)
            edges = []
            for disease, foods in tmp.items():
                for food in foods:
                    edges.append([disease, food])
            create_relationship(g, "Disease", "Food", "not_eat", edges, "忌吃", False)
        else:
            with open(f'./{label}_disease_rel.json', 'r', encoding='utf-8') as f:
                tmp = json.load(f)
            if label == "pathogen":
                edges = []
                for disease, pathogens in tmp.items():
                    for pathogen in pathogens:
                        edges.append([pathogen['pathogen'], disease, pathogen['detail']])
                create_relationship(g, label.capitalize(), "Disease", "cause", edges, "引起", True)
            elif label == "toxin":
                edges = []
                for disease, toxins in tmp.items():
                    for toxin in toxins:
                        edges.append([toxin['toxin'], disease, toxin['detail']])
                create_relationship(g, label.capitalize(), "Disease", "cause", edges, "引起", True)
            elif label == "check":
                edges = []
                for disease, checks in tmp.items():
                    for check in checks:
                        edges.append([disease, check['check'], check['detail']])
                create_relationship(g, "Disease", label.capitalize(), "check_way", edges, "诊断检查", True)
            elif label == "neopathy":
                edges = []
                for disease, neopathies in tmp.items():
                    for neopathy in neopathies:
                        edges.append([disease, neopathy])
                create_relationship(g, "Disease", "Disease", "accompany_with", edges, "并发症", False)
            elif label == "symptom":
                edges = []
                for disease, symptoms in tmp.items():
                    for symptom in symptoms:
                        edges.append([disease, symptom['symptom'], json.dumps(symptom['detail'], ensure_ascii=False) if isinstance(symptom['detail'], list) else symptom['detail']]) # 后期发现大模型生成detail存在列表形式
                create_relationship(g, "Disease", label.capitalize(), "has_symptom", edges, "症状", True)
            elif label == "drug":
                edges = []
                for disease, drugs in tmp.items():
                    for drug in drugs:
                        edges.append([disease, drug['drug'], drug['detail']])
                create_relationship(g, "Disease", label.capitalize(), "cure_way", edges, "治疗药物", True)
            elif label == "cure":
                edges = []
                for disease, cures in tmp.items():
                    for cure in cures:
                        edges.append([disease, cure['cure'], cure['detail']])
                create_relationship(g, "Disease", label.capitalize(), "cure_way", edges, "治疗方法", True)
            elif label == "department":
                edges = []
                for disease, departments in tmp.items():
                    for department in departments:
                        edges.append([disease, department])
                create_relationship(g, "Disease", label.capitalize(), "belong_to", edges, "属于", False)

    for label in ["disease", "pathogen", "toxin", "check", "symptom", "drug", "cure", "department", "food"]:
        print(f"creating index for {label}")
        create_index(g, label.capitalize())


if __name__ == "__main__":
    build_graph("bolt://localhost:7688", "username", "password")

    build_triples_embedding("storage location", "embedding model path", "device")