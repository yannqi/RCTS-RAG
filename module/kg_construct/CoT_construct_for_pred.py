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
    
    csv_path = os.path.join(path_to_save_preds, f'{dataset.split}_pred_CoT.csv')
    qid2CoT = {}
    CoT_path = csv_path.split('Pred_')[-1]
    CoT_path = './logs/' + CoT_path

    if os.path.exists(CoT_path):
        current_preds_df = pd.read_csv(CoT_path)
        current_question_list = current_preds_df['question_id'].tolist()
        current_question_list = [str(i) for i in current_question_list]
        current_CoT_list = current_preds_df['pred_CoT'].tolist()
        for i, qid in enumerate(current_question_list):
            qid2CoT[qid] = current_CoT_list[i]
    else:
        raise NotImplementedError('You should implement the first time CoT extract')
    
    for i, (sample_question_id, sample_image_path, sample_question, sample_answer) in tqdm(enumerate(dataset), total=len(dataset)):
        CoT = qid2CoT.get(str(sample_question_id))
        whole_prompt_list.append({'question': sample_question, 'answer': sample_answer, 'image_path': sample_image_path, 'question_id': sample_question_id, 'CoT': CoT})
  
    Reward_loop_api_extra_body = copy.deepcopy(api_extra_body)

    already_processed = 0
    question_id_list, pred_judge_list, best_pred_CoT_list= [], [], []
    previous_answer_list = []

    async def _process_single_content(prompt_dict):
        nonlocal already_processed, question_id_list, pred_judge_list, best_pred_CoT_list 
        sample_question = prompt_dict['question']
        sample_answer = prompt_dict['answer']
        sample_image_path = prompt_dict['image_path']
        sample_question_id = prompt_dict['question_id']
        sample_CoT = prompt_dict['CoT']
       
        pred_judge_content = np.zeros(CoT_loop_times).tolist()
       
        pred_CoT_list = []
        system_prompt_round2 = dataset.get_system_prompt('woRAG')
        
        if hasattr(dataset, 'get_question_prompt'):
            sample_question_round2 = dataset.get_question_prompt(sample_question, use_CoT=False)
        else:
            sample_question_round2 = sample_question
        user_prompt_round2 = CoT_PROMPTS["Answer_with_CoT_VQA_v1"]
        user_prompt_round2 = user_prompt_round2.format(question=sample_question_round2, thinking_process=sample_CoT)
        
        sample_content_round2 = make_interleave_content(user_prompt_round2, sample_image_path)
        messages_round2 = [{'role': 'system', 'content': f"{system_prompt_round2}"}]
        messages_round2.append({'role': 'user', 'content': sample_content_round2})
        
        pred_answer = await vqa_model(prompt=messages_round2, extra_body=Reward_loop_api_extra_body)    
        # pred_answer_content.append(pred_answer)
      
        if not hasattr(dataset, 'judge_answer'):
            raise NotImplementedError('You should implement judge_answer')
       
        if dataset_name == 'ScienceQA' or dataset_name == 'VSR_MC_random' or dataset_name == 'VSR_MC_zeroshot' or dataset_name == 'VizWiz' or dataset_name == 'MMMU':
            
            pred_judge_content = dataset.judge_answer(sample_answer, pred_answer) # Judge Answer return True or False

        elif dataset_name == 'MathV':
            pred_judge_content = await dataset.judge_answer(pred_answer, sample_question_id, vqa_model, api_extra_body)
         
        previous_answer_list.append(pred_answer)
   
        pred_judge_list.append(pred_judge_content)
        question_id_list.append(sample_question_id)

        already_processed += 1
        if already_processed % 50 == 0 and already_processed > 0:  # save preds every 10 samples
            logger.info(f"Processed {already_processed} samples, date: {time.strftime('%H:%M', time.localtime())}")
            header = not os.path.exists(csv_path)
            preds_df = pd.DataFrame({'question_id': question_id_list,  'pred_judge': pred_judge_list, 'pred_answer': previous_answer_list})
            preds_df.to_csv(csv_path, index=False, mode='a' if os.path.exists(csv_path) else 'w', header=header)
            question_id_list.clear()
            pred_judge_list.clear()
            previous_answer_list.clear()
            

    await tqdm_asyncio.gather(*[_process_single_content(prompt_dict) for prompt_dict in whole_prompt_list])
    # save the predictions
    header = not os.path.exists(csv_path)
    preds_df = pd.DataFrame({'question_id': question_id_list,  'pred_judge': pred_judge_list, 'pred_answer': previous_answer_list})
    preds_df.to_csv(csv_path, index=False, mode='a' if os.path.exists(csv_path) else 'w', header=header)

    return preds_df


