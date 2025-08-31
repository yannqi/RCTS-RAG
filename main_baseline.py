import os, sys
import argparse
import asyncio
import torch
import faiss
import gc
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import json
from module.RAG.RCTS_RAG import RCTS_RAG
from data import ScienceQA, MathV, VSR_MC, VizWiz, MMMU
from misc import create_logging, create_output_folders
from typing import Type, cast
from module.storage import BaseKVStorage, JsonKVStorage, StorageNameSpace
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import time
from functools import partial
from openai import OpenAI
from misc.logger import logger
from module.model.llm import limit_async_func_call_with_multi_node, vqa_model_func_with_multi_node
from module.RAG.answer import answer_woMCTS, answer_wMCTS
import nest_asyncio
nest_asyncio.apply()
import copy


async def main(
            LOG_DIR: str,
            DATASET_NAME:str,
            DATASET_PATH: dict,
            USE_RAG: bool,
            TOP_K: int = 1,
            FAISS_DIR: str='null',
            COT_CSV_PATH: str='null',
            QUERY_MODE: str='null',
            INDEX_TYPE: str='null',
            INDEX_MODEL: dict={},
            USE_MCTS: bool=False,
            MCTS_TOP_K: int=3,
            MCTS_ROLLOUTS: int=5,
            LLM_MODEL_MAX_ASYNC: int = 10,
            ENABLE_LLM_CACHE: bool = False,
            api_extra_body: dict = {},
            reward_api_extra_body: dict = {},
            SAVE_EMBEDDING_PATH: str='null',
            REWARD_CONFIG_DICT: dict={},
            **kwargs    
            ):
    
    path_to_save_preds = os.path.join(LOG_DIR, 'outputs')


    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    
    
    
   
    if DATASET_NAME == 'ScienceQA':
        dataset = ScienceQA(data_root=DATASET_PATH['DATA_ROOT'], split='test')
        api_extra_body.update({'skip_special_tokens': False})
    elif DATASET_NAME == 'MathV':
        dataset = MathV(data_root=DATASET_PATH['DATA_ROOT'], split='testmini')
    elif DATASET_NAME == 'VSR_MC_zeroshot' or DATASET_NAME == 'VSR_MC_random':
        dataset = VSR_MC(data_root=DATASET_PATH['DATA_ROOT'], split='test', data_type=DATASET_NAME.split('_')[-1])
    elif DATASET_NAME == 'VizWiz':
        dataset = VizWiz(data_root=DATASET_PATH['DATA_ROOT'], split='val')
    elif DATASET_NAME == 'MMMU':
        dataset = MMMU(data_root=DATASET_PATH['DATA_ROOT'], split='dev')
    async def _vqa_done(llm_response_cache):
        tasks = []
        for storage_inst in [llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)


    if COT_CSV_PATH is not None:
        CoT_df = pd.read_csv(COT_CSV_PATH)
    else:
        CoT_df = None
        
    os.makedirs(SAVE_EMBEDDING_PATH, exist_ok=True)
    if QUERY_MODE == 'random':
        whole_prompt_list_path = os.path.join(SAVE_EMBEDDING_PATH, f'{DATASET_NAME}_whole_prompt_list_{TOP_K}_random.json')
    else:
        whole_prompt_list_path = os.path.join(SAVE_EMBEDDING_PATH, f'{DATASET_NAME}_whole_prompt_list_{TOP_K}.json')
    if USE_RAG:
        if os.path.exists(whole_prompt_list_path):
            with open(whole_prompt_list_path, 'r') as f:
                whole_prompt_list = json.load(f)
            
        else:
            yannqi_rag = RCTS_RAG(
                log_dir=FAISS_DIR,
                index_type=INDEX_TYPE,
                index_model=INDEX_MODEL
            )
            whole_prompt_list = []
    else:
        whole_prompt_list = []
    exist_question_id_list = []
    exist_preds_df_path = os.path.join(path_to_save_preds, 'final_preds.csv')
    if os.path.exists(exist_preds_df_path):
        exist_preds_df = pd.read_csv(exist_preds_df_path, usecols=[0])
        exist_question_id_list = exist_preds_df['question_id'].tolist()
        exist_question_id_list = [str(x) for x in exist_question_id_list]
    if len(whole_prompt_list) == 0:
       
        for (sample_question_id, sample_image_path, sample_question, sample_answer) in tqdm(dataset, total=len(dataset)):        
            if sample_question_id in exist_question_id_list:
                continue
            context_dict_list = None
            if USE_RAG:  
                if isinstance(sample_image_path, list):
                    query_image_path = sample_image_path[0]
                else:
                    query_image_path = sample_image_path
                query_embedding_dict = yannqi_rag.get_query_embdding_dict(sample_question_id, sample_question, query_image_path, DATASET_NAME, INDEX_MODEL, SAVE_EMBEDDING_PATH)
                context_dict_list = await yannqi_rag.query(sample_question, query_mode=QUERY_MODE, top_k=(TOP_K+1), query_embedding_dict=query_embedding_dict)   
                if sample_question_id in [context_dict['question_id'].split('_')[-1] for context_dict in context_dict_list]:
                    context_dict_list = [context_dict for context_dict in context_dict_list if context_dict['question_id'].split('_')[-1] != sample_question_id]
                    print(f"remove same sample_question for fair compair: {sample_question_id}")
                else:
                    context_dict_list = context_dict_list[:-1]
                if len(context_dict_list) != TOP_K:
                    raise ValueError(f"context_dict_list length is not equal to TOP_K, {len(context_dict_list)} != {TOP_K}")
            
 
            whole_prompt_list.append({'question': sample_question, 'answer': sample_answer, 'image_path': sample_image_path, 'question_id': sample_question_id, 'context_dict_list': copy.deepcopy(context_dict_list)})
        for prompt_dict in whole_prompt_list:
            prompt_dict['question_id'] = str(prompt_dict['question_id'])

        if USE_RAG:
            # save whole_prompt_list
            with open(whole_prompt_list_path, 'w') as f:
                json.dump(whole_prompt_list, f)
            del yannqi_rag       
            gc.collect()
            torch.cuda.empty_cache()

    
    else:
        to_remove_index = []
        for i, prompt_dict in enumerate(whole_prompt_list):
            if str(prompt_dict['question_id']) in exist_question_id_list:
                to_remove_index.append(i)
        whole_prompt_list = [whole_prompt_list[i] for i in range(len(whole_prompt_list)) if i not in to_remove_index]
    llm_response_cache = (
            key_string_value_json_storage_cls(
                namespace="llm_response_cache", save_dir=path_to_save_preds
            )
            if ENABLE_LLM_CACHE
            else None
        )
    vqa_model = limit_async_func_call_with_multi_node(LLM_MODEL_MAX_ASYNC, max_node=len(os.environ.get("BASE_URL").split(';')))(
            partial(vqa_model_func_with_multi_node, hashing_kv=llm_response_cache)
        )
    
    
    
    # reward_vqa_model = limit_async_func_call(10)(
    #         partial(reward_vqa_model_func, hashing_kv=llm_response_cache)
    #     )
    
    reward_vqa_model = vqa_model # TODO yannqi 我们暂时让这两个一模一样
    
    logger.info(f"Start processing {len(whole_prompt_list)} samples")
    already_processed = 0 
    question_id_list = []
    pred_answer_list = []
    ori_answer_list = []
    api_count_list = [] 
    reranking_context_list = []
    start_time = time.time()
    async def _process_single_content(prompt_dict):
        nonlocal already_processed, question_id_list, start_time, pred_answer_list, ori_answer_list, api_count_list, reranking_context_list
        # start_time = time.time()
        sample_question_id = prompt_dict['question_id']
        if USE_MCTS:
            pred_answer, reranking_context, ori_answer, api_count = await answer_wMCTS(prompt_dict=prompt_dict, dataset=dataset, top_k=MCTS_TOP_K, CoT_df=CoT_df, vqa_model=vqa_model, api_extra_body=api_extra_body, max_rollouts=MCTS_ROLLOUTS, reward_config_dict=REWARD_CONFIG_DICT , reward_vqa_model=reward_vqa_model, reward_api_extra_body=reward_api_extra_body)
        else:
            pred_answer = await answer_woMCTS(prompt_dict, dataset, CoT_df, vqa_model, api_extra_body, USE_RAG)
        sample_question_id = prompt_dict['question_id']
        pred_answer_list.append(pred_answer)
        question_id_list.append(sample_question_id)
        ori_answer_list.append(ori_answer) if USE_MCTS else None
        api_count_list.append(api_count) if USE_MCTS else None
        reranking_context_list.append(reranking_context) if USE_MCTS else None
        already_processed += 1
        if already_processed % 50 == 0:  
            logger.info(f"Processed {already_processed} samples, date: {time.strftime('%H:%M', time.localtime())}")
            header = not os.path.exists(os.path.join(path_to_save_preds, f'final_preds.csv'))
            vqa_model_preds_df = pd.DataFrame({'question_id': question_id_list, 'pred_answer': pred_answer_list})
            vqa_model_preds_df['ori_answer'] = ori_answer_list if USE_MCTS else None
            vqa_model_preds_df['api_count'] = api_count_list if USE_MCTS else None
            vqa_model_preds_df['reranking_context'] = reranking_context_list if USE_MCTS else None
            vqa_model_preds_df.to_csv(os.path.join(path_to_save_preds, f'final_preds.csv'), index=False, mode='a' if os.path.exists(os.path.join(path_to_save_preds, f'final_preds.csv')) else 'w', header=header)
            question_id_list.clear()
            pred_answer_list.clear()
            ori_answer_list.clear() if USE_MCTS else None
            api_count_list.clear() if USE_MCTS else None
            reranking_context_list.clear() if USE_MCTS else None
            logger.info(f"Processed {already_processed} samples, remain {len(whole_prompt_list) - already_processed} samples. cost {time.time() - start_time}s")
            if ENABLE_LLM_CACHE:
                await _vqa_done(llm_response_cache)
            start_time = time.time()
  
      
    await tqdm_asyncio.gather(*[_process_single_content(prompt_dict) for prompt_dict in whole_prompt_list]) #TODO yannqi 只拿前10个来验证

        # save the predictions
   
    vqa_model_preds_df = pd.DataFrame({'question_id': question_id_list, 'pred_answer': pred_answer_list})
    vqa_model_preds_df['ori_answer'] = ori_answer_list if USE_MCTS else None
    vqa_model_preds_df['api_count'] = api_count_list if USE_MCTS else None
    vqa_model_preds_df['reranking_context'] = reranking_context_list if USE_MCTS else None
    header = not os.path.exists(os.path.join(path_to_save_preds, f'final_preds.csv'))
    vqa_model_preds_df.to_csv(os.path.join(path_to_save_preds, f'final_preds.csv'), index=False, mode='a' if os.path.exists(os.path.join(path_to_save_preds, f'final_preds.csv')) else 'w', header=header)
    if ENABLE_LLM_CACHE:
        await _vqa_done(llm_response_cache)
 
    logger.info("Finish!")
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tool for extracting vectorDB.')
    parser.add_argument('--config', dest='config', help='path to config file', type=str, required=True)
    args = parser.parse_args()
    args_dict = OmegaConf.load(args.config)
    for key in args_dict.keys():
        if key.endswith('_CONFIG'):
            temp_dict = OmegaConf.load(args_dict[key]) 
            args_dict = OmegaConf.merge(args_dict, temp_dict)
    openai_client = OpenAI(api_key= os.environ.get("API_KEY"), base_url=os.environ.get("BASE_URL").split(';')[0])
    model_cards =  openai_client.models.list()
    model_name = model_cards.data[0].id
    model_name = model_name.split('/')[-1]
    args_dict.PROJECT_NAME = model_name + '_' + args_dict.PROJECT_NAME
    LOG_DIR = create_output_folders(project_name=args_dict.PROJECT_NAME, output_dir=args_dict.LOG_DIR_ORI, config=args_dict, with_time=args_dict.WITH_TIME)
    create_logging(logger, LOG_DIR)
    asyncio.run(main(LOG_DIR=LOG_DIR, **args_dict))
    print('Finish!')
    sys.exit(0)
