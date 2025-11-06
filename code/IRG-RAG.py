import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "path to huggingface cache path"
os.environ["CUDA_VISIBLE_DEVICES"] = "device id"

import logging
# 配置log文件将控制台输出保存到文件
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    encoding='utf-8',
                    handlers=[
                        logging.FileHandler(f"log file path"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()

from unsloth import FastLanguageModel
from vllm import SamplingParams
import neo4j
import re
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

device4unsloth = "balanced"
device4embedding = "embedding model device"

# 宜吃+推荐吃->do_eat
GRAPH_SCHEMA = '''Node properties:
Disease {name: STRING, susceptible_population: STRING, mode_of_infection: STRING, treatment_cycle: STRING, cure_rate: STRING, prevent: STRING, nursing_info: STRING, dietary_advice: STRING}
Symptom {name: STRING}
Drug {name: STRING}
Cure {name: STRING}
Food {name: STRING}
Check {name: STRING}
Pathogen {name: STRING}
Toxin {name: STRING}
Department {name: STRING}

Relationship properties:
belong_to {name: STRING}
secrete {name: STRING}
cause {name: STRING, detail: STRING}
accompany_with {name: STRING}
has_symptom {name: STRING, detail: STRING}
check_way {name: STRING, detail: STRING}
cure_way {name: STRING, detail: STRING}
do_eat {name: STRING}
not_eat {name: STRING}

The relationships:
(:Disease)-[:belong_to]->(:Department)
(:Pathogen)-[:secrete]->(:Toxin)
(:Pathogen)-[:cause]->(:Disease)
(:Toxin)->[:cause]->(:Disease)
(:Disease)-[:accompany_with]->(:Disease)
(:Disease)-[:has_symptom]->(:Symptom)
(:Disease)-[:check_way]->(:Check)
(:Disease)-[:cure_way]->(:Drug)
(:Disease)-[:cure_way]->(:Cure)
(:Disease)-[:do_eat]->(:Food)
(:Disease)-[:not_eat]->(:Food)'''

relName_en2relName_ch = {
    "belong_to": "就诊科室",
    "secrete": "代谢产物",
    "cause": "引起",
    "accompany_with": "并发症",
    "has_symptom": "症状",
    "check_way": "诊断检查",
    "cure_way": "治疗",
    "do_eat": "宜吃",
    "not_eat": "忌吃"
} # 最终回答问题时组织关系使用

typeName_relName_direction = {
    "Disease": {"belong_to": "out", "cause": "in", "accompany_with": ["out", "in"], "has_symptom": "out", "check_way": "out", "cure_way": "out", "do_eat": "out", "not_eat": "out"},
    "Symptom": {"has_symptom": "in"},
    "Drug": {"cure_way": "in"},
    "Cure": {"cure_way": "in"},
    "Food": {"do_eat": "in", "not_eat": "in"},
    "Check": {"check_way": "in"},
    "Pathogen": {"secrete": "out", "cause": "out"},
    "Toxin": {"secrete": "in", "cause": "out"},
    "Department": {"belong_to": "in"}
} # 判断某类型实体是否与某类型的边存在关联 + 边方向

PROMPTS = {
    "relations_rerank": ['''# 以下内容是一个你可以与之交互的知识图谱的Schema：
{}

# 以下内容是当前确定的有助于问题的解答的三元组搜索历史：
{}

# 以下内容是你曾经做出的无用的向外探索决策：
{}
每一次决策均由[decision begin]和[decision end]所包裹。

# 以下内容是一个（或一组）支持进一步探索的候选实体：
{}

# 以下内容是一个待解决的医学问题：
```
{}
```

请根据你对该医学问题的理解，参考所给出的搜索历史和历史无用决策，从给定的图Schema中为每个候选实体选出最合适的可供进一步向外探索的边的类型。''', '''
# 请注意：
1. 不需要解决该医学问题，你的任务仅是为每个出发词挑选最合适的可供进一步向外探索的边的类型；
2. 在所给出的历史无用决策中，通过这些决策探索到的三元组被认为是对问题的解答没有帮助的，请不要让同一实体再次探索在这些决策中出现过的边类型；
3. 在得出最终答案之前，请仔细思考问题，逐步推理，然后以指定的JSON格式返回你思考后的分析结果；
4. 如果你认为某个候选实体没有合适类型的边可供进一步探索，又或者该实体合适类型的边已经在历史决策中被探索过了，可放弃对该实体的进一步探索，直接将其排除在返回结果之外。
5. 如果所有候选实体都被排除在外，请直接返回空JOSN列表。

# 分析结果返回格式：
```json
[
  {
    "实体名": "",
    "边类型": ""
  }
]
```'''],
    "triples_rerank": ['''# 以下内容是从知识图谱中检索得到的与问题的解答可能相关的三元组：
{}
在我给你的检索结果中，每个结果都是[triple X begin]...[triple X end]格式的，X代表每个三元组的数字id。

# 以下内容是一个待解决的医学问题：
```
{}
```

请根据你对该医学问题的理解，从检索得到的三元组中找出真正有助于解答该医学问题的三元组，并返回它们各自的数字id。
# 要求1：不需要解决该医学问题，你的任务仅是从检索结果中挑选出有助于解答该问题的三元组；
# 要求2：在得出最终答案之前，请仔细思考问题，逐步推理，然后以指定的JSON格式返回你思考后的分析结果；
# 要求3：如果你认为检索结果中不存在有用的三元组，请直接返回空JSON列表。

# 分析结果返回格式：
```json
[三元组id1, 三元组id2]
```'''],
    "entities_selection": ['''# 以下内容是从知识图谱中检索得到的有助于问题的解答的知识：
{}

# 以下内容是一个待解决的医学问题：
```
{}
```

请根据你对该医学问题的理解，判断当前已经获得的图谱数据是否已经具备充足的可用于回答该问题的知识。如果尚未具备充足知识，从搜索历史中挑选出最合适的可供进一步向外探索的实体节点，并返回这些实体的名称及其类型。
# 要求：
1. 不需要解决该医学问题，你的任务仅是从搜索历史中挑选出最合适的可供进一步向外探索的实体；
2. 你可以在**边信息**中出现过的所有**头实体**和**尾实体**中做出选择；
3. 在得出最终答案之前，请仔细思考问题，逐步推理，然后以指定的JSON格式返回你思考后的分析结果；
4. 当你认为现有知识已经足够用于回答该医学问题，不需要再进行进一步探索，或者没有合适的实体可供进一步探索时，请直接返回空JSON列表。''', '''

# 分析结果返回格式：
```json
[
  {
    "实体名": "",
    "实体类型": ""
  }
]
```'''],
    "answer": '''# 以下内容是根据[问题]从知识图谱中检索得到的实体及关系信息：
{}

请参考这些信息回答下方问题。
在回答之前，请仔细思考问题，确保回答逻辑清晰且准确。

# 问题：
```
{}
```

# 回答：''',
    "answer_only_question": '''以下是一个任务说明，配有提供更多信息的输入。
请输出一个恰当的回答来完成该任务。
在回答之前，请仔细思考问题，确保回答逻辑清晰且准确。

# 任务说明：
你是一位在临床推理、诊断和治疗规划方面具备丰富知识的医学专家。
请回答以下医学问题。

# 问题：
{}

# 回答：'''
}

class R1_unsloth:
    def __init__(self, model_path, max_seq_length=8192, dtype=None, load_in_4bit=False):
        # self.model, self.tokenizer = FastLanguageModel.from_pretrained(
        #     model_name = model_path,
        #     max_seq_length = max_seq_length,
        #     dtype = dtype,
        #     load_in_4bit = load_in_4bit,
        #     device_map=device4unsloth
        # ) # 模型加载

        # FastLanguageModel.for_inference(self.model) # 推理模式

        # 使用vllm推理
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            fast_inference = True,
            gpu_memory_utilization = 0.95,
            device_map = device4unsloth
        )
        self.max_seq_length = max_seq_length

    def query_llm(self, queries, max_new_tokens=4096):
        """执行推理"""
        queries = self.tokenizer.apply_chat_template(queries, add_generation_prompt=True, tokenize=False)
        # inputs = self.tokenizer(queries, add_special_tokens=False, padding=True, truncation=True, return_tensors="pt").to(device4unsloth)

        # outputs = self.model.generate(
        #     **inputs,
        #     max_new_tokens=max_new_tokens,
        #     do_sample=True,
        #     temperature=0.6,
        #     top_p=0.95,
        #     use_cache=True
        # )
        # responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # 使用vllm推理
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=self.max_seq_length)
        outputs = self.model.fast_generate(queries, sampling_params=sampling_params)
        responses = []
        for output in outputs:
            responses.append(output.prompt+output.outputs[0].text)

        return responses

    def response_extraction(self, responses):
        """分离最终回答"""
        responses_extracted = []
        for response in responses:
            response_splited = response.split('</think>')
            if len(response_splited) == 2:
                responses_extracted.append(response_splited[1].strip())
            else:
                responses_extracted.append(None)

        return responses_extracted

    def get_response(self, inputs, input_type):
        """组织输入并调用大模型"""
        if input_type == "relations_rerank":
            logger.info(f"关系筛选=>{inputs[3]} | {inputs[1]}")
            responses = self.query_llm([{"role": "user", "content": PROMPTS[input_type][0].format(GRAPH_SCHEMA, inputs[0], inputs[1], json.dumps(inputs[2], ensure_ascii=False, indent=2), inputs[3])+PROMPTS[input_type][1]}])
            logger.info(json.dumps(responses, ensure_ascii=False))
            return self.response_extraction(responses)[0]
        elif input_type == "triples_rerank":
            logger.info(f"为问题：{inputs[1]} 挑选最优三元组")
            responses = self.query_llm([{"role": "user", "content": PROMPTS[input_type][0].format(inputs[0], inputs[1])}])
            logger.info(json.dumps(responses, ensure_ascii=False))
            return self.response_extraction(responses)[0]
        elif input_type == "entities_selection":
            logger.info(f"主题节点筛选=>{inputs[1]}")
            responses = self.query_llm([{"role": "user", "content": PROMPTS[input_type][0].format(inputs[0], inputs[1])+PROMPTS[input_type][1]}])
            logger.info(json.dumps(responses, ensure_ascii=False))
            return self.response_extraction(responses)[0]
        elif input_type == "answer":
            count = 0
            questions, subgraphs = inputs
            queries = []
            for question, subgraph in zip(questions, subgraphs):
                if subgraph:
                    queries.append([{"role": "user", "content": PROMPTS[input_type].format(json.dumps(subgraph, ensure_ascii=False, indent=2), question)}])
                    count += 1
                else:
                    queries.append([{"role": "user", "content": PROMPTS[input_type+"_only_question"].format(question)}])

            logger.info("最终回答")
            responses = self.query_llm(queries)
            logger.info(json.dumps(responses, ensure_ascii=False))

            return count, responses

class Cache_manager:
    def __init__(self, triples_path="default cache path", diseases_info_path="path to **disease(source).json**"):
        self.__triples_path = Path(triples_path)
        self.__diseases_info_path = Path(diseases_info_path)

        self.__triples_cache = None
        self.__infos_cache = dict()
        self.load()

    def load(self):
        if self.__triples_path.exists():
            with open(self.__triples_path, 'r', encoding='utf-8') as f:
                self.__triples_cache = json.load(f)
        else:
            self.__triples_cache = dict()
            if not os.path.exists(self.__triples_path.parent):
                os.mkdir(self.__triples_path.parent)

        if self.__diseases_info_path.exists():
            with open(self.__diseases_info_path, 'r', encoding='utf-8') as f:
                tmp = json.load(f)

                for item in tmp:
                    self.__infos_cache[item['name']] = {"疾病名": item['name'], "易感人群": item['susceptible_population'], "传播方式": item['mode_of_infection'], "治疗周期": item['treatment_cycle'], "治愈率": item['cure_rate'], "预防措施": item['prevent'], "护理建议": item['nursing_info'], "饮食建议": item['dietary_advice']}
        else:
            logger.error("包含疾病实体详细信息的文件不存在")
            sys.exit(-1)

    def update_triples(self, entity, relation, entities):
        if entity in self.__triples_cache:
            self.__triples_cache[entity].setdefault(relation, []).extend(entities)
        else:
            self.__triples_cache[entity] = {relation: entities}

    def triples_exists(self, entity, relation):
        if entity in self.__triples_cache and relation in self.__triples_cache[entity]:
            return True
        return False

    def get_triples(self, entity, relation):
        return self.__triples_cache[entity][relation]

    def disease_info_exists(self, k):
        return k in self.__infos_cache

    def get_disease_info(self, k):
        return self.__infos_cache[k]

    def save(self):
        with open(self.__triples_path, 'w', encoding='utf-8') as f:
            json.dump(self.__triples_cache, f, ensure_ascii=False, indent=4)

class Embedding_Retrieval():
    def __init__(self, model_path, milvus_location):
        self.embedding_fn = SentenceTransformer(model_path, device=device4embedding)
        self.milvus = MilvusClient(milvus_location)

    def retrieval(self, queries, top_k=20):
        query_embeddings = self.embedding_fn.encode(queries, prompt_name="query")

        q_results = self.milvus.search(
            collection_name="triples",
            data=query_embeddings,
            limit=top_k,
            search_params={"metric_type": "COSINE"},
            output_fields=["triple", "triple_str"]
        )

        ret, ret_str = [], []
        for result in q_results:
            ret.append([])
            ret_str.append([])

            for hit in result:
                ret[-1].append(hit.get("entity").get("triple"))
                ret_str[-1].append(hit.get("entity").get("triple_str"))
        return ret, ret_str

    def get_top_k(self, query, triples, top_k=20):
        query_embedding = self.embedding_fn.encode([query], prompt_name="query")
        doc_embeddings = self.embedding_fn.encode(triples)

        similarity = self.embedding_fn.similarity(query_embedding, doc_embeddings)[0]
        indice_score_pair = list(enumerate(similarity))
        indice_score_pair.sort(key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in indice_score_pair[: top_k]]

class Retriever:
    def __init__(self, graph_database_info, embedding_model_path, milvus_location, top_k, llm, cacheManager, max_try, max_depth):
        self.driver = neo4j.GraphDatabase.driver(graph_database_info[0], auth=(graph_database_info[1], graph_database_info[2]))

        self.embedding_retriener = Embedding_Retrieval(embedding_model_path, milvus_location)
        self.top_k = top_k

        self.llm = llm

        self.cache = cacheManager
        with open('path to **entity2labels.json**', 'r', encoding='utf-8') as f:
            self.entity2labels = json.load(f)

        self.lock = Lock()

        self.max_try = max_try # 既是输出有问题时的重试次数限制，也是单步探索出错时回退次数限制
        self.max_depth = max_depth

    def close(self):
        self.driver.close()

    def json_extraction(self, string):
        """分离并转化json串"""
        pattern = r'```json(.*?)```'

        result = None
        try:
            result = re.search(pattern, string, re.DOTALL)
            if result:
                result = result.group(1).strip()
                result = json.loads(result)
            elif string.startswith('['):
                result = json.loads(string[: string.rfind(']')+1])
            else:
                result = None
        except Exception as e:
            result = None

        return result

    def get_entities_info(self, entities, existing):
        """（非抽取）疾病实体信息"""
        ret = []
        for entity in entities:
            if self.cache.disease_info_exists(entity['实体名']) and entity['实体名'] not in existing:
                # 直接在cache导入json文件查比查图快
                ret.append(self.cache.get_disease_info(entity['实体名']))
                existing.add(entity['实体名'])
        return ret

    def run_and_post_process(self, cypher, entity, label, relation, direction, have_detail):
        """cypher执行"""
        records, _, _ = self.driver.execute_query(
            cypher,
            {"name": entity},
            routing_=neo4j.RoutingControl.READ,
            database_="neo4j"
        )

        tmp = []
        ret, ret_str = [], [] # 前者用于问答，后者用于向量相似度计算
        if direction == "out":
            for record in records:
                triple = {"头实体": entity, "头实体类型": label, "关系类型": relation, "尾实体": record['to.name'], "尾实体类型": record['toLabel'], "描述信息": record['r.detail'] if have_detail else ''}
                ret.append(triple)
                ret_str.append(json.dumps({"头实体": entity, "关系类型": relName_en2relName_ch[relation], "尾实体": record['to.name'], "描述信息": record['r.detail'] if have_detail else ''}, ensure_ascii=False))

                tmp.append({'尾实体': triple['尾实体'], '尾实体类型': triple['尾实体类型'], '描述信息': triple['描述信息']})
        else:
            for record in records:
                triple = {"头实体": record['from.name'], "头实体类型": record['fromLabel'], "关系类型": relation, "尾实体": entity, "尾实体类型": label, "描述信息": record['r.detail'] if have_detail else ''}
                ret.append(triple)
                ret_str.append(json.dumps({"头实体": record['from.name'], "关系类型": relName_en2relName_ch[relation], "尾实体": entity, "描述信息": record['r.detail'] if have_detail else ''}, ensure_ascii=False))

                tmp.append({'头实体': triple['头实体'], '头实体类型': triple['头实体类型'], '描述信息': triple['描述信息']})

        if tmp:
            with self.lock:
                # 缓存写入
                self.cache.update_triples(entity, relation, tmp)

        return ret, ret_str

    def request_graph(self, entity, label, relation):
        """查库"""
        if self.cache.triples_exists(entity, relation):
            logger.info(f"检索[{entity}|{label}|{relation}]三元组信息命中缓存")
            # 命中缓存
            entities = self.cache.get_triples(entity, relation)

            ret, ret_str = [], []
            for _entity in entities:
                if label == "Disease" and relation == "accompany_with":
                    # 并发症查双向
                    if '尾实体' in _entity:
                        ret.append({"头实体": entity, "头实体类型": label, "关系类型": relation, "尾实体": _entity['尾实体'], "尾实体类型": _entity['尾实体类型'], "描述信息": _entity['描述信息']})
                        ret_str.append(json.dumps({"头实体": entity, "关系类型": relName_en2relName_ch[relation], "尾实体": _entity['尾实体'], "描述信息": _entity['描述信息']}, ensure_ascii=False))
                    else:
                        ret.append({"头实体": _entity['头实体'], "头实体类型": _entity['头实体类型'], "关系类型": relation, "尾实体": entity, "尾实体类型": label, "描述信息": _entity['描述信息']})
                        ret_str.append(json.dumps({"头实体": _entity['头实体'], "关系类型": relName_en2relName_ch[relation], "尾实体": entity, "描述信息": _entity['描述信息']}, ensure_ascii=False))
                elif typeName_relName_direction[label][relation] == "out":
                    if '尾实体' in _entity:
                        ret.append({"头实体": entity, "头实体类型": label, "关系类型": relation, "尾实体": _entity['尾实体'], "尾实体类型": _entity['尾实体类型'], "描述信息": _entity['描述信息']})
                        ret_str.append(json.dumps({"头实体": entity, "关系类型": relName_en2relName_ch[relation], "尾实体": _entity['尾实体'], "描述信息": _entity['描述信息']}, ensure_ascii=False))
                else:
                    if '头实体' in _entity:
                        ret.append({"头实体": _entity['头实体'], "头实体类型": _entity['头实体类型'], "关系类型": relation, "尾实体": entity, "尾实体类型": label, "描述信息": _entity['描述信息']})
                        ret_str.append(json.dumps({"头实体": _entity['头实体'], "关系类型": relName_en2relName_ch[relation], "尾实体": entity, "描述信息": _entity['描述信息']}, ensure_ascii=False))

            if ret:
                # 该步判断应对少部分同名不同类型实体。虽然命中缓存，但是边的方向不对
                return ret, ret_str

        logger.info(f"检索[{entity}|{label}|{relation}]三元组信息")

        detail_option = ''
        if relation in ["cause", "has_symptom", "check_way", "cure_way"]:
            detail_option = "r.detail, "

        ret, ret_str = [], []
        if label == "Disease" and relation == "accompany_with":
            # 并发症查双向
            cypher = f"MATCH (from:{label})-[r:{relation}]->(to) WHERE from.name = $name RETURN {detail_option}to.name, head(labels(to)) as toLabel"
            _ret, _ret_str = self.run_and_post_process(cypher, entity, label, relation, "out", detail_option=="r.detail, ")
            ret.extend(_ret)
            ret_str.extend(_ret_str)

            cypher = f"MATCH (from)-[r:{relation}]->(to:{label}) WHERE to.name = $name RETURN {detail_option}from.name, head(labels(from)) as fromLabel"
            _ret, _ret_str = self.run_and_post_process(cypher, entity, label, relation, "in", detail_option=="r.detail, ")
            ret.extend(_ret)
            ret_str.extend(_ret_str)
        elif typeName_relName_direction[label][relation] == "out":
            cypher = f"MATCH (from:{label})-[r:{relation}]->(to) WHERE from.name = $name RETURN {detail_option}to.name, head(labels(to)) as toLabel"
            ret, ret_str = self.run_and_post_process(cypher, entity, label, relation, "out", detail_option=="r.detail, ")
        else:
            cypher = f"MATCH (from)-[r:{relation}]->(to:{label}) WHERE to.name = $name RETURN {detail_option}from.name, head(labels(from)) as fromLabel"
            ret, ret_str = self.run_and_post_process(cypher, entity, label, relation, "in", detail_option=="r.detail, ")

        return ret, ret_str

    def triples_filter(self, subgraph_diseases_info, existing_diseases, filted_triples, filted_triples_str, question, existing_relations):
        """三元组筛选"""
        # 三元组扁平化
        _triples = ""
        for i, triple in enumerate(filted_triples_str):
            _triples += f"[triple {i} begin]\n"
            _triples += triple + '\n'
            _triples += f"[triple {i} end]\n"
        _triples = _triples[: -1]

        for times in range(self.max_try):
            logger.info(f"第{times+1}次尝试筛选三元组")

            response = self.llm.get_response((_triples, question), "triples_rerank")
            if not response:
                continue

            triple_ids = self.json_extraction(response)
            if triple_ids == None:
                continue

            ret = []
            new_diseases = []
            for Id in triple_ids:
                try:
                    if isinstance(Id, str):
                        Id = int(Id)

                    triple = filted_triples[Id]
                    ret.append(triple)
                    existing_relations.add(f"{triple['头实体']}|{triple['关系类型']}|{triple['尾实体']}")
                    if triple['头实体类型'] == "Disease":
                        new_diseases.append({"实体名": triple['头实体']})
                    if triple['尾实体类型'] == "Disease":
                        new_diseases.append({"实体名": triple['尾实体']})
                except Exception as e:
                    pass
            if not ret and triple_ids:
                # 考虑是id生成错误导致的内容缺失，重试
                continue

            subgraph_diseases_info.extend(self.get_entities_info(new_diseases, existing_diseases)) # 检索疾病实体属性信息
            return ret
        return []

    def get_triples_associated_with_entity(self, subgraph_diseases_info, existing_diseases, decision, existing_relations, question):
        """获取实体相关三元组"""
        triples, triples_str = [], []
        with ThreadPoolExecutor(max_workers=10) as executor: # 检索
            futures = [executor.submit(self.request_graph, request['实体名'], request['实体类型'], request['边类型']) for request in decision]

            for future in futures:
                search_result, search_result_str = future.result()
                if search_result:
                    triples.extend(search_result)
                    triples_str.extend(search_result_str)

        if not triples:
            return []

        # 过滤重复边
        existing_relations_tmp = set() # 这次检索结果中的已有边
        filted_triples, filted_triples_str = [], []
        for triple, triple_str in zip(triples, triples_str):
            key = f"{triple['头实体']}|{triple['关系类型']}|{triple['尾实体']}"
            if key in existing_relations or key in existing_relations_tmp:
                continue

            filted_triples.append(triple)
            filted_triples_str.append(triple_str)
            existing_relations_tmp.add(key)

        if not filted_triples:
            return []

        # 基于向量相似度的初步过滤
        if len(filted_triples) > self.top_k:
            logger.info("三元组（向量相似度）初步过滤")
            indices = self.embedding_retriener.get_top_k(question, filted_triples_str, self.top_k)
            filted_triples = [filted_triples[i] for i in indices]
            filted_triples_str = [filted_triples_str[i] for i in indices]

        return self.triples_filter(subgraph_diseases_info, existing_diseases, filted_triples, filted_triples_str, question, existing_relations)

    def judgement(self, subgraph, question):
        _subgraph = {"疾病实体信息": [], "边信息": []}
        _subgraph["疾病实体信息"] = subgraph["疾病实体信息"]
        for triple in subgraph["边信息"]:
            # 边类型转换，使用更易理解的中文
            _subgraph["边信息"].append({"头实体": triple["头实体"], "头实体类型": triple['头实体类型'], "关系类型": relName_en2relName_ch[triple["关系类型"]], "尾实体": triple["尾实体"], "尾实体类型": triple['尾实体类型'], "描述信息": triple["描述信息"]})
        _subgraph = json.dumps(_subgraph, ensure_ascii=False, indent=2)

        for times in range(self.max_try):
            logger.info(f"第{times+1}次尝试筛选主题节点")

            response = self.llm.get_response((_subgraph, question), "entities_selection")
            if not response:
                continue

            entities = self.json_extraction(response)
            if entities == None:
                continue

            ret = []
            for entity in entities:
                try:
                    if entity['实体名'] in self.entity2labels and entity['实体类型'] in self.entity2labels[entity['实体名']]:
                        ret.append({"实体名": entity['实体名'], "实体类型": entity['实体类型']})
                except Exception as e:
                    pass
            if not ret and entities:
                # 因为错误生成导致的空列表，重试
                continue
            return ret
        return []

    def make_decision(self, subgraph, decisions, maked_decisions, entities, question):
        tmp = []
        for rel in subgraph:
            tmp.append({"头实体": rel['头实体'], "关系类型": rel['关系类型'], "尾实体": rel['尾实体'], "描述信息": rel['描述信息']}) # 为对齐schema，使用原始关系类型名
        triples = json.dumps(tmp, ensure_ascii=False, indent=2) if tmp else "暂无历史信息"

        _decisions = ""
        for decision in decisions:
            _decisions += f"[decision begin]\n"
            tmp = []
            for item in decision:
                tmp.append({"实体名": item["实体名"], "边类型": item["边类型"]}) # 实体类型是后加的，去掉，避免干扰模型输出格式
            _decisions += json.dumps(tmp, ensure_ascii=False, indent=2)
            _decisions += f"\n[decision end]\n"
        _decisions = _decisions[: -1] if _decisions else "暂无任何错误决策"

        entity2type = {}
        for entity in entities:
            entity2type[entity["实体名"]] = entity["实体类型"]

        for times in range(self.max_try):
            logger.info(f"第{times+1}次尝试做出探索决策")

            response = self.llm.get_response((triples, _decisions, entities, question), "relations_rerank")
            if not response:
                continue

            decision = self.json_extraction(response)
            if decision == None:
                continue

            ret = []
            valid_decisions_count = 0
            for item in decision:
                try:
                    if item['实体名'] in entity2type and item['边类型'] in typeName_relName_direction[entity2type[item['实体名']]]:
                        valid_decisions_count += 1
                        key = entity2type[item['实体名']] + '|' + item['实体名']
                        if key in maked_decisions and item['边类型'] in maked_decisions[key]:
                            continue # 已经探索过
                        ret.append({"实体名": item["实体名"], "实体类型": entity2type[item['实体名']], "边类型": item["边类型"]})
                        maked_decisions.setdefault(key, []).append(item['边类型'])
                except Exception as e:
                    pass
            if not ret and decision and not valid_decisions_count:
                # 因为错误生成导致空列表，重试
                continue
            return ret
        return []

    def initial_run(self, questions):
        filted_triples_list, filted_triples_str_list = self.embedding_retriener.retrieval(questions, self.top_k)

        subgraphs, existing_diseases_list, existing_relations_list, entities_list = [], [], [], []
        for question, filted_triples, filted_triples_str in zip(questions, filted_triples_list, filted_triples_str_list):
            logger.info(f"问题：{question} 初始处理")

            subgraph = {"疾病实体信息": [], "边信息": []}
            existing_diseases = set()
            existing_relations = set()

            triples = self.triples_filter(subgraph["疾病实体信息"], existing_diseases, filted_triples, filted_triples_str, question, existing_relations)
            subgraph["边信息"].extend(triples)

            subgraphs.append(subgraph)
            existing_diseases_list.append(existing_diseases)
            existing_relations_list.append(existing_relations)

            entities = []
            if triples:
                entities = self.judgement(subgraph, question)
            entities_list.append(entities)

        return subgraphs, existing_diseases_list, existing_relations_list, entities_list

    def generate_subgraph(self, questions):
        _subgraphs, existing_diseases_list, existing_relations_list, entities_list = self.initial_run(questions)

        subgraphs = [None] * len(questions)
        for idx, (question, entities) in enumerate(zip(questions, entities_list)):
            try:
                subgraph = _subgraphs[idx]
                if entities:
                    logger.info(f"问题：{question} 进一步检索")

                    existing_diseases = existing_diseases_list[idx] # 控制疾病实体属性信息仅保存一次
                    existing_relations = existing_relations_list[idx] # 控制相同三元组仅处理一次
                    maked_decisions = {} # 代码层面控制某个实体某类型的边仅探索一次（大模型不一定能很好地遵循prompt）
                    entities_for_next_step = entities
                    decisions = [] # 历史决策
                    for depth in range(self.max_depth):
                        Continue = False # 是否接着探索（不管是真的需要结束还是意外导致的结束）
                        for times in range(self.max_try):
                            # 此处重试实现的是回退（反思）的思路
                            decision = self.make_decision(subgraph["边信息"], decisions, maked_decisions, entities_for_next_step, question)
                            if not decision:
                                # 做不出探索决策
                                break

                            triples = self.get_triples_associated_with_entity(subgraph['疾病实体信息'], existing_diseases, decision, existing_relations, question)
                            if not triples:
                                logger.info(f"深度 {depth} 第 {times+1} 次回退")
                                decisions.append(decision)
                                continue
                            subgraph["边信息"].extend(triples)
                            Continue = True
                            break

                        if Continue:
                            entities_for_next_step = self.judgement(subgraph, question)
                            if not entities_for_next_step:
                                # 1. 知识充足；2. 无法进一步探索
                                break
                        else:
                            break

                if subgraph["疾病实体信息"] or subgraph["边信息"]:
                    _subgraph = {"疾病实体信息": [], "边信息": []}
                    _subgraph["疾病实体信息"] = subgraph["疾病实体信息"]
                    for triple in subgraph["边信息"]:
                        # 边类型转换，使用更易理解的中文
                        # 不再需要类型信息
                        _subgraph["边信息"].append({"头实体": triple["头实体"], "关系类型": relName_en2relName_ch[triple["关系类型"]], "尾实体": triple["尾实体"], "描述信息": triple["描述信息"]})
                    subgraphs[idx] = _subgraph
            except Exception as e:
                logger.error(f"处理问题：[{question}] 时出错")
                logger.error(f"{e}")

        return subgraphs

class Executor:
    def __init__(self, model_path, graph_database_info, embedding_model_path, milvus_location, retriever_top_k, max_try, max_depth):
        self.__llm = R1_unsloth(model_path, 64*1024)
        self.__cacheManager = Cache_manager()
        self.__retriever = Retriever(graph_database_info, embedding_model_path, milvus_location, retriever_top_k, self.__llm, self.__cacheManager, max_try, max_depth)

    def run(self, questions):
        subgraphs = self.__retriever.generate_subgraph([question[0] for question in questions])

        # 带了较多额外信息，把批大小砍半
        smaller_batch_size = len(questions) // 2 if len(questions) >= 2 else len(questions)
        count = 0
        answers = []
        for i in range(0, len(questions), smaller_batch_size):
            try:
                _count, _answers = self.__llm.get_response(([question[0]+question[1] for question in questions[i: i+smaller_batch_size]], subgraphs[i: i+smaller_batch_size]), "answer")
            except Exception as e:
                logger.error(f"处理小批次{i}时出错")
                logger.error(f"{e}")
                _count = 0
                _answers = ["None"] * smaller_batch_size

            count += _count
            answers.extend(_answers)

        return count, answers

    def close(self):
        self.__retriever.close()

    def save_cache(self):
        self.__cacheManager.save()

def main(questions, model_path, batch_size, graph_database_info, embedding_model_path, milvus_location, retriever_top_k, max_try, max_depth):
    executor = Executor(model_path, graph_database_info, embedding_model_path, milvus_location, retriever_top_k, max_try, max_depth)

    writer = open("path to results file(jsonl)", 'a', encoding='utf-8')
    logger.info(f"共{len(questions)}条数据")
    questions = [questions[i: i+batch_size] for i in range(0, len(questions), batch_size)]
    answers = {}
    # answers = [] # CMExam
    num = 0
    for rank, _questions in enumerate(questions):
        logger.info(f"正在处理第{rank}批次：")
        logger.info(_questions)

        inputs = []
        for question in _questions:
            inputs.append((question[1], question[2]))
        try:
            count, results = executor.run(inputs)
        except Exception as e:
            logger.error(f"处理第{rank}批次时出错")
            logger.error(f"{e}")
            results = ["None"] * len(_questions)
            count = 0

        for question, answer in zip(_questions, results):
            answers[question[0]] = answer
            writer.write(json.dumps({"id": question[0], "answer": answer}, ensure_ascii=False)+'\n')
            # CMExam
            # answers.append({"id": question[3], "answer": answer, "golden": question[0]})
            # writer.write(json.dumps({"id": question[3], "answer": answer, "golden": question[0]}, ensure_ascii=False)+'\n')

        if (rank+1) % 100 == 0: # 每100个批次保存一次cache，和写出一次结果
            executor.save_cache()
            writer.flush()

        num += count
        logger.info("="*20)

    writer.close()
    with open("path to results file(json)", 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

    logger.info(f"共使用RAG{num}次")

    executor.save_cache()
    executor.close()

if __name__ == '__main__':
    # CMB
    with open("path to CMB-Exam file", 'r', encoding='utf-8') as f:
        tmp = json.load(f)
    questions = []
    for item in tmp:
        options = ""
        if item["option"]["A"]:
            options += '\nA. ' + item["option"]["A"]
        if item["option"]["B"]:
            options += '\nB. ' + item["option"]["B"]
        if item["option"]["C"]:
            options += '\nC. ' + item["option"]["C"]
        if item["option"]["D"]:
            options += '\nD. ' + item["option"]["D"]
        if item["option"]["E"]:
            options += '\nE. ' + item["option"]["E"]
        questions.append((item["id"], item["question"], options))

    # CMExam
    # with open("path to CMExam file", 'r', encoding='utf-8') as f:
    #     tmp = json.load(f)
    # questions = []
    # for item in tmp:
    #     questions.append((item["answer"], item["question"], item["options"], item["id"]))

    model_path = "model path"
    embedding_model_path = "embedding model path"
    batch_size = 4
    # neo4j 参数
    uri = "bolt://localhost:7688"
    username = "username"
    password = "password"
    # milvus 参数
    milvus_location = "triples embedding location"

    main(questions, model_path, batch_size, (uri, username, password), embedding_model_path, milvus_location, 20, 3, 5)