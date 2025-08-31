import os
import torch
from module.prompt import PROMPTS
from misc import logger
from module.storage import BaseVectorStorage, BaseKVStorage, TextChunkSchema
import random
from PIL import Image
from flmr import create_searcher, search_custom_collection
from easydict import EasyDict
async def naive_text_query(
    query,
    question_vdb: BaseVectorStorage,
    VQA_docs_db: BaseKVStorage[TextChunkSchema],
    top_k = 3,
    query_embedding_dict=None
): 

    results = await question_vdb.query(query, top_k=top_k, query_embedding_dict=query_embedding_dict)

    if not len(results):
        return PROMPTS["fail_response"]
    
    question_ids = [r["id"] for r in results]
    VQA_docs = await VQA_docs_db.get_by_ids(question_ids)
    for i in range(len(VQA_docs)):
            VQA_docs[i].update({'confidence':results[i]['distance']})  # TODO 这里的置信度值是负数，越小越好
    return VQA_docs


async def random_query(
    # query,
    # question_vdb: BaseVectorStorage,
    VQA_docs_db: BaseKVStorage[TextChunkSchema],
    top_k=3,
    # query_embedding_dict=None
): 
    # Random Choice 1
    # ntotal = question_vdb._index.ntotal
    # random_index = random.sample(range(ntotal), top_k)
    # results = await question_vdb.query(query, top_k=ntotal, query_embedding_dict=query_embedding_dict)
    # results = random.sample(results, top_k)
    # idx_list = list(question_vdb._metadata.keys())
    # results = [question_vdb._metadata[idx_list[i]] for i in random_index]
    # if not len(results):
        # return PROMPTS["fail_response"]
    # question_ids = [r["id"] for r in results]
    
    ntotal = len(VQA_docs_db._data)
    random_index = random.sample(range(ntotal), top_k)
    idx_list = list(VQA_docs_db._data.keys())
    VQA_docs = [VQA_docs_db._data[idx_list[i]] for i in random_index]
    # VQA_docs = await VQA_docs_db.get_by_ids(question_ids)
    for i in range(len(VQA_docs)):
        VQA_docs[i].update({'confidence':0})  # TODO 这里的置信度值是负数，越小越好
    return VQA_docs



def _hybrid_embedding_func(
        query_texts,
        img_path,
        pre_flmr_model,
        image_processor,
):
    module = EasyDict(
            {"type": "QuestionInput", "option": "default", "separation_tokens": {"start": "", "end": ""}}
        )
    
    instruction = "Question:"
    for i in range(len(query_texts)):
        query_texts[i] = instruction + query_texts[i]
    query_encoding = pre_flmr_model.query_tokenizer(query_texts, max_length=512) # TODO yannqi 这里是为了保证和docs的长度对齐，因为我们都是问题形似检索
    image = Image.open(img_path).convert("RGB")
    query_pixel_values = image_processor(image, return_tensors="pt")['pixel_values']
    inputs = dict(
        input_ids=query_encoding['input_ids'],
        attention_mask=query_encoding['attention_mask'],
        pixel_values=query_pixel_values,
    )
    # Run model query encoding
    res = pre_flmr_model.query(**inputs)  
    query_embedding = res.late_interaction_output
    return query_embedding

async def hybrid_query(
    query,
    img_path,
    mapping_index2question_id_json,
    searcher,
    image_processor,
    pre_flmr_model,
    VQA_docs_db: BaseKVStorage[TextChunkSchema],
    top_k=3,
    query_embedding_dict=None,
): 
    
    if query_embedding_dict['embedding_path'] != None:
        if os.path.exists(query_embedding_dict['embedding_path']):
            query_embedding = torch.load(query_embedding_dict['embedding_path'])
        else:
            query_embedding = _hybrid_embedding_func(
                query_texts=[query],
                img_path=img_path,
                pre_flmr_model=pre_flmr_model,
                image_processor=image_processor,
            )  # [1,320,128]
            torch.save(query_embedding, query_embedding_dict["embedding_path"])
    else:
        query_embedding = _hybrid_embedding_func(
                query_texts=[query],
                img_path=img_path,
                pre_flmr_model=pre_flmr_model,
                image_processor=image_processor,
            )
    # Search the collection
    num_queries = 1
    queries = {i: query for i in range(num_queries)}
    ranking = search_custom_collection(
        searcher=searcher,
        queries=queries,
        query_embeddings=query_embedding,
        num_document_to_retrieve=top_k, # how many documents to retrieve for each query # * 检索的Top-K
    ) 

    # Analyse retrieved documents
    ranking_dict = ranking.todict()

    for i in range(num_queries):
        retrieved_docs = ranking_dict[i]
        retrieved_docs_indices = [doc[0] for doc in retrieved_docs]
        retrieved_doc_scores = [doc[2] for doc in retrieved_docs]
        question_ids = [mapping_index2question_id_json[str(doc_idx)] for doc_idx in retrieved_docs_indices]

        # data = {
        #     "Confidence": retrieved_doc_scores,
        #     "Content": retrieved_doc_texts,
        # }
     
        VQA_docs = await VQA_docs_db.get_by_ids(question_ids)
        for i in range(len(VQA_docs)):
            VQA_docs[i].update({'confidence':retrieved_doc_scores[i]}) 

    return VQA_docs

