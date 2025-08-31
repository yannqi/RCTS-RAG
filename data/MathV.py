import os 
import torch
import numpy as np
import pandas as pd
import json
import re
from latex2sympy2 import latex2sympy
from .data_prompt import DATA_PROMPTS
from misc.logger import logger
from module.evaluate.utils.MathV_utils import can_infer, load, dump, build_mathv_gpt4_prompt, post_check, is_equal, MATH_V_auxeval_for_MCTS, MATH_V_auxeval, is_equal

import pickle

FAIL_MSG = 'Failed to obtain answer via API.'
def is_equal(asw: str, gt_asw: str) -> bool:
    if not isinstance(asw, str) != str or not isinstance(gt_asw, str):
        print('Warning: input is not string')
        print(asw, gt_asw)
    asw = str(asw).lower().strip()
    gt_asw = str(gt_asw).lower().strip()
    if gt_asw == asw:
        return True
    try:
        a = eval(gt_asw)
        b = eval(asw)
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    try:
        a = latex2sympy(gt_asw)
        b = latex2sympy(asw)
        if abs(eval(str(a)) - eval(str(b))) < 1e-6:
            return True
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    return False
def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}

# ref https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/image_vqa.py
class MathV(torch.utils.data.Dataset):
    def __init__(self, data_root, split='testmini', skip_noimg=True):
        self.name = 'MathV'
        self.split = split
        self.data_root = data_root
        if split == 'testmini':
            tsv_root = os.path.join(self.data_root + '/data/', 'MathVision_MINI.tsv')
        elif split == 'test':
            tsv_root = os.path.join(self.data_root + '/data/', 'MathVision.tsv')
            tsv_root_mini = os.path.join(self.data_root + '/data/', 'MathVision_MINI.tsv')
        else:
            raise ValueError(f"split: {split} is not supported.")
        data = load(tsv_root, fmt='tsv')
        data.drop(columns=['image'], inplace=True)
        
        temp_question_id_list_mini = []
        if split == 'test':
            data_mini = load(tsv_root_mini, fmt='tsv')
            data_mini.drop(columns=['image'], inplace=True)
            common_question_ids = data_mini['index'].tolist()
            # 使用 isin 方法找到 data 中与 data_mini 相同的 question_id
            common_rows = data[data['index'].isin(common_question_ids)]

            # 使用 drop 方法删除这些行
            data.drop(common_rows.index, inplace=True)
            print(f"Remove {len(common_rows)} rows from MathVision.tsv")
        self.lines = [data.iloc[i] for i in range(len(data))]
        temp_question_id_list = data['index'].tolist()
        temp_question_id_list = [str(x) for x in temp_question_id_list if x not in temp_question_id_list_mini]
    
        self.indices = {str(temp_question_id_list[i]): i for i in range(len(temp_question_id_list))}
        
        if skip_noimg and 'image' in data:
            data = data[~pd.isna(data['image'])]
        data['index'] = [str(x) for x in data['index']]
        if np.all([isinstance(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]
 
        index_list = data['index'].tolist()
        for index in index_list:
            image_path = 'images/' + str(index) + '.jpg'   
            # Add image path to the data
            data.loc[data['index'] == index, 'image_path'] = image_path
        self.question_id_list = [str(x) for x in data['index']]
        self.image_path_list = [str(x) for x in data['image_path']]
        self.question_list = [str(x) for x in data['question']]
        self.answer_list = [str(x) for x in data['answer']]
        # Note: The MathV-Test dataset contains MathV-Testmini. So the indices are the same. 
        knowledge_base_data = load(os.path.join(self.data_root + '/data/', 'MathVision.tsv'), fmt='tsv')
        knowledge_base_data.drop(columns=['image'], inplace=True)
        self.knowledge_lines = [knowledge_base_data.iloc[i] for i in range(len(knowledge_base_data))]
        knowledge_question_id_list = knowledge_base_data['index'].tolist()
        self.knowledge_indices = {str(knowledge_question_id_list[i]): i for i in range(len(knowledge_question_id_list))}

    def __getitem__(self, idx):
        question_id = self.question_id_list[idx]
        question = self.question_list[idx]
        answer = f"The answer is \\boxed{{{self.answer_list[idx]}}}."
        # answer = f"\\boxed{{{self.answer_list[idx]}}}."
        image_path = os.path.join(self.data_root, self.image_path_list[idx])
        return question_id, image_path, question, answer
    def __len__(self):
        return len(self.question_id_list)

    @staticmethod
    def get_system_prompt(system_dtype):

        if system_dtype == "RAG":
            system_prompt = DATA_PROMPTS["RAG_MathV"]
        elif system_dtype == "woRAG":
            system_prompt = DATA_PROMPTS["RAG_MathV_woRAG"]
        elif system_dtype == "RAG_CoT_Refine":
            system_prompt = DATA_PROMPTS["RAG_MathV_CoT_refine"]
        elif system_dtype == "RAG_CoT":
            system_prompt = DATA_PROMPTS["RAG_MathV_CoT"]
        elif system_dtype == "RAG_Refine":
            system_prompt = DATA_PROMPTS["RAG_MathV_refine"]
        elif system_dtype == "Self_Consistency":
            system_prompt = DATA_PROMPTS["Self_Consistency_MathV"]
        
        else:
            raise ValueError(f"system_dtype: {system_dtype} is not supported.")
        return system_prompt
       
    @staticmethod
    # ref https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/vlm/internvl/utils.py
    def get_question_prompt(question, use_CoT):
    
        def build_mcq_cot_prompt(question):
            cot_prompt = (
                "Answer the preceding multiple choice question. The format of your response should be of the following format: "
                "'The answer is \\boxed{X}.\nBECAUSE: xxx' (without quotes) where X must be one of options. Think step by step before answering."
            )
            
            prompt = question + '\n' + cot_prompt

            return prompt


        def build_qa_cot_prompt(question):
            cot_prompt = (
                "Answer the preceding multiple choice question. The format of your response should be of the following format: "
                "'The answer is \\boxed{$FINAL_ANSWER}.\nBECAUSE: xxx' (without quotes) where 'FINAL_ANSWER' is your conclusion based on the reasoning provided. Think step by step before answering."
            )
       
            prompt = question + '\n' + cot_prompt

            return prompt
        
        def build_mcq_prompt(question):
            prompt = (
                "Answer the preceding multiple choice question. The format of your response should follow "
                "this format: 'The answer is \\boxed{X}.' (without quotes), where 'X' must be one of the options. "
            )
            
            prompt = question + '\n' + prompt
            return prompt
            
            
        def build_qa_prompt(question):
            prompt = (
                "Answer the preceding question. The format of your response should follow this format: "
                "'The answer is \\boxed{$YOUR_ANSWER}.' (without quotes), where 'YOUR_ANSWER' is your conclusion."
            )
            prompt = question + '\n' + prompt
            return prompt
        if use_CoT:
            if 'correct option letter' in question:
                refined_question = build_mcq_cot_prompt(question)
            else:
                refined_question = build_qa_cot_prompt(question)
        
        else:
            if 'correct option letter' in question:
                refined_question = build_mcq_prompt(question)
            else:
                refined_question = build_qa_prompt(question)
        return refined_question


    async def judge_answer(self, pred_answer, sample_question_id, vqa_model, api_extra_body):
        line = self.lines[self.indices[str(sample_question_id)]]
        line['pred_answer'] = pred_answer
        result_extract = await MATH_V_auxeval(line=line, vqa_model=vqa_model, api_extra_body=api_extra_body, retry=1)
        line['extract'] = result_extract['res']
        fetch, hit = False, False
        if result_extract['log'] == 'Prefetch succeed':
            fetch = True
        if post_check(line, prefetch=False):
            hit = True
        
        if hit:
            return True
        elif fetch:
            return False
        else:
            return 'Bad'
    
    def judge_answer_for_mcts(self, pred_answer, sample_question_id):
        # Note only support reverse reward
        sample_question_id = sample_question_id.split('_')[-1]
        line = self.knowledge_lines[self.knowledge_indices[str(sample_question_id)]]
        pred_answer = re.split(r'BECASUE:|BECAUSE:|Because:|because:', pred_answer)[0]
        pred_answer = pred_answer.strip()
        pred_answer = pred_answer.strip('\n')
        pred_answer = pred_answer.strip('**')
        match = re.search(r'\\boxed\{([^{}]+)\}', pred_answer)
        if match:
            # 提取到的内容
            pred_answer = match.group(1)
            pred_answer = pred_answer.strip('$')
            if pred_answer == 'X':
                return 0 # X is not a valid answer

        
        gt_answer = line['answer']
        
       
        pattern = r'[\(\{\[](.*?)[\)\}\]]' # filter out the choice with (), {} or []
        match = re.search(pattern, pred_answer)
        if match:
            pred_answer = match.group(1)
        match = re.search(pattern, gt_answer) 
        if match:
            gt_answer = match.group(1)
        hit = is_equal(pred_answer, gt_answer)

        if hit:
            return 1
        else:
            return 0

    async def judge_early_stop_for_mcts(self, answer_list, sample_question_id, vqa_model, api_extra_body):
        
        line = self.lines[self.indices[str(sample_question_id)]]
        extract_answer_list = []
        for i, pred_answer in enumerate(answer_list):
            if i != 0:
                pred_answer = re.split(r'BECASUE:|BECAUSE:|Because:|because:', pred_answer)[0]
                # pred_answer = re.split(r'\*\*final answer:\*\*|\*\*FINAL ANSWER:\*\*|FINAL ANSWER:|Final Answer\:', pred_answer)[-1]
                pred_answer = pred_answer.strip()
                pred_answer = pred_answer.strip('\n')
                pred_answer = pred_answer.strip('**')
            match = re.search(r'\\boxed\{([^{}]+)\}', pred_answer)
            if match:
                # 提取到的内容
                line['pred_answer'] = match.group(1)
                line['pred_answer'] = line['pred_answer'].strip('$')
                if line['pred_answer'] == 'X': # X is not a valid answer
                    return False
            else:
                line['pred_answer'] = pred_answer
                
            result_extract = await MATH_V_auxeval_for_MCTS(line=line, vqa_model=vqa_model, api_extra_body=api_extra_body, retry=1)
            extract_answer_list.append(result_extract)
        temp_extract_answer = extract_answer_list[0]
        for extract_answer in extract_answer_list:
            if extract_answer != temp_extract_answer:
                return False
        return True
    
    @staticmethod
    def judge_answer_for_mcts_self_eval(gt_answer, pred_answer):
        match = re.search(r'\\boxed\{([^{}]+)\}', gt_answer)
        if match:
            # 提取到的内容
            gt_answer = match.group(1)
            gt_answer = gt_answer.strip('$')
            if gt_answer == 'X':
                return 0 # X is not a valid answer
        
        
        match = re.search(r'\\boxed\{([^{}]+)\}', pred_answer)
        if match:
            # 提取到的内容
            pred_answer = match.group(1)
            pred_answer = pred_answer.strip('$')
        hit = is_equal(pred_answer, gt_answer)
        if hit:
            return 1
        else:
            return 0