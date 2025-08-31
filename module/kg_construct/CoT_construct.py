import os
import ast
import asyncio
import torch
import time
import pandas as pd 
import numpy as np 
from PIL import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import torch
import base64
from typing import Type, cast
from misc.logger import logger

import copy
from module.storage import BaseKVStorage, JsonKVStorage, StorageNameSpace
from functools import partial
from .CoT_prompt import CoT_PROMPTS
from module.RAG.utils import make_interleave_content

async def Self_CoT_extract(
    vqa_model,
    dataset, 
    dataset_name,
    path_to_save_preds,
    CoT_loop_times,
    Reward_loop_times,
    api_extra_body: dict,
    ):
    
    whole_prompt_list = []
    
    # API Setting
    csv_path = os.path.join(path_to_save_preds, f'{dataset.split}_pred_CoT.csv')
    if os.path.exists(csv_path):
        current_preds_df = pd.read_csv(csv_path)
        current_question_list = current_preds_df['question_id'].tolist()
        current_question_list = [str(i) for i in current_question_list]
    else:
        current_question_list = []
        
    
    for i, (sample_question_id, sample_image_path, sample_question, sample_answer) in tqdm(enumerate(dataset), total=len(dataset)):
        if str(sample_question_id) in current_question_list:
            continue
        whole_prompt_list.append({'question': sample_question, 'answer': sample_answer, 'image_path': sample_image_path, 'question_id': sample_question_id})
  
    async def _vqa_done(llm_response_cache):
        tasks = []
        for storage_inst in [llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    Reward_loop_api_extra_body = copy.deepcopy(api_extra_body)
    Reward_loop_api_extra_body.update({'n': Reward_loop_times})
    already_processed = 0
    question_id_list, pred_judge_list, best_pred_CoT_list= [], [], []
    async def _process_single_content(prompt_dict):
        nonlocal already_processed, question_id_list, pred_judge_list, best_pred_CoT_list 
        sample_question = prompt_dict['question']
        sample_answer = prompt_dict['answer']
        sample_image_path = prompt_dict['image_path']
        sample_question_id = prompt_dict['question_id']
       
       
        pred_judge_content = np.zeros(CoT_loop_times).tolist()
       
        system_prompt = CoT_PROMPTS["get_CoT_VQA_v1_system_prompt"]
        if hasattr(dataset, 'get_question_prompt'):
            sample_question_round1 = dataset.get_question_prompt(sample_question, use_CoT=False)
        else:
            sample_question_round1 = sample_question
        user_prompt = CoT_PROMPTS["get_CoT_VQA_v1"]
        user_prompt = user_prompt.format(question=sample_question_round1, answer=sample_answer)
   
        sample_content_round1 = make_interleave_content(user_prompt, sample_image_path)
        messages_round1 = [
                    {
                        "role": "system", 'content': f'{system_prompt}',
                    }
            ]

        messages_round1.append({'role': 'user', 'content':  sample_content_round1 })
      
        pred_CoT_list = []
        system_prompt_round2 = dataset.get_system_prompt('woRAG')
        
        
            
        for index in range(CoT_loop_times):
     
            pred_CoT = await vqa_model(prompt=messages_round1, extra_body=api_extra_body)  # CoT steps

            # pred_loop_content.append(pred_CoT)
            # CoT Reward steps
            pred_CoT_list.append(pred_CoT)

            if hasattr(dataset, 'get_question_prompt'):
                sample_question_round2 = dataset.get_question_prompt(sample_question, use_CoT=False)
            else:
                sample_question_round2 = sample_question
            user_prompt_round2 = CoT_PROMPTS["Answer_with_CoT_VQA_v1"]
            user_prompt_round2 = user_prompt_round2.format(question=sample_question_round2, thinking_process=pred_CoT)
            
            sample_content_round2 = make_interleave_content(user_prompt_round2, sample_image_path)
            messages_round2 = [{'role': 'system', 'content': f"{system_prompt_round2}"}]
            messages_round2.append({'role': 'user', 'content': sample_content_round2})
            
            pred_answer_list = await vqa_model(prompt=messages_round2, extra_body=Reward_loop_api_extra_body)    
            # pred_answer_content.append(pred_answer)

            if not hasattr(dataset, 'judge_answer'):
                raise NotImplementedError('You should implement judge_answer')
            for pred_answer in pred_answer_list:
                if dataset_name == 'ScienceQA' or dataset_name == 'VSR_MC_random' or dataset_name == 'VSR_MC_zeroshot' or dataset_name == 'VizWiz' or dataset_name == 'MMMU':
                    
                    pred_judge = dataset.judge_answer(sample_answer, pred_answer) # Judge Answer return True or False
                
                    if pred_judge == True:
                        pred_judge_content[index] += 1
                    elif pred_judge == False:
                        pred_judge_content[index] += 0
                    elif pred_judge == 'Bad':
                        pred_judge_content[index] += -1
                    else:
                        raise NotImplementedError('judge_answer should return True or False')
                elif dataset_name == 'MathV':
                    pred_judge = await dataset.judge_answer(pred_answer, sample_question_id, vqa_model, api_extra_body)
                    if pred_judge == True:
                        pred_judge_content[index] += 1
                    elif pred_judge == False:
                        pred_judge_content[index] += 0
                    elif pred_judge == 'Bad':
                        pred_judge_content[index] += -1
                    else:
                        raise NotImplementedError('judge_answer should return True or False')
            if pred_judge_content[index] == len(pred_answer_list):
                break  
        
        pred_judge_list.append(pred_judge_content)
        question_id_list.append(sample_question_id)
        best_pred_CoT_list.append(pred_CoT_list[np.argmax(pred_judge_content)])
        already_processed += 1
        if already_processed % 50 == 0 and already_processed > 0:  # save preds every 10 samples
            logger.info(f"Processed {already_processed} samples, date: {time.strftime('%H:%M', time.localtime())}")
            header = not os.path.exists(csv_path)
            preds_df = pd.DataFrame({'question_id': question_id_list, 'pred_CoT': best_pred_CoT_list, 'pred_judge': pred_judge_list})
            preds_df.to_csv(csv_path, index=False, mode='a' if os.path.exists(csv_path) else 'w', header=header)
            question_id_list.clear()
            pred_judge_list.clear()
            best_pred_CoT_list.clear()
            

    await tqdm_asyncio.gather(*[_process_single_content(prompt_dict) for prompt_dict in whole_prompt_list])
    # save the predictions
    header = not os.path.exists(csv_path)
    preds_df = pd.DataFrame({'question_id': question_id_list, 'pred_CoT': best_pred_CoT_list, 'pred_judge': pred_judge_list})  
    preds_df.to_csv(csv_path, index=False, mode='a' if os.path.exists(csv_path) else 'w', header=header)

    return preds_df


