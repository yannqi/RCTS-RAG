import os 
import torch
import numpy as np
import pandas as pd
import json
import re
from .data_prompt import DATA_PROMPTS
import hashlib
from module.evaluate.VizWiz_evaluate_acc import EvalAIAnswerProcessor
option_list = [
        'above', 'across', 'across from', 'adjacent to', 'against', 'ahead of', 
        'along', 'alongside', 'around', 'at the back of', 'at the edge of', 
        'at the left side of', 'at the right side of', 'at the side of', 
        'attached to', 'away from', 'behind', 'below', 'beneath', 'beside', 
        'between', 'beyond', 'by', 'close to', 'connected to', 'contains', 
        'consists of', 'detached from', 'down from', 'enclosed by', 'facing', 
        'facing away from', 'far away from', 'far from', 'has as a part', 'in', 
        'in front of', 'in the middle of', 'inside', 'into', 'left of', 'near', 
        'next to', 'off', 'on', 'on top of', 'opposite to', 'out of', 'outside', 
        'over', 'parallel to', 'part of', 'past', 'perpendicular to', 'right of', 
        'surrounding', 'toward', 'touching', 'under', 'within', 'with'
        ]
class VSR_MC(torch.utils.data.Dataset):
    def __init__(self, data_root, split, data_type='random'):
        self.name = 'VSR_MC'
        self.split = split
        self.data_json = []
        if split == 'trainval':
            json_path = os.path.join(data_root, 'splits', data_type, 'train_options.jsonl')
            with open(json_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        j_line = json.loads(line)
                        self.data_json.append(j_line)
            json_path = os.path.join(data_root, 'splits', data_type, 'dev_options.jsonl')
            with open(json_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        j_line = json.loads(line)
                        self.data_json.append(j_line)
        elif split == 'test':
            json_path = os.path.join(data_root, 'splits', data_type, 'test_options.jsonl')
            with open(json_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        j_line = json.loads(line)
                        self.data_json.append(j_line)
        

        self.question_id_list, self.question_list, self.image_path_list, self.answer_list = [], [], [], []
        self.choice_list, self.lecture_list, self.solution_list = [], [], []
        
        total_relations = []
        for data in self.data_json:
            image_name = data['image']
            image_path = os.path.join(data_root, 'images', image_name)
            subject = data['subj']
            
            object = data['obj']
            answer = data['relation']
            
            problem = f"The {subject} (is) ___ the {object}."
            choice_list = data['options']

            answer = self._get_answer(answer, choice_list, options=["A", "B", "C", "D", "E", "F"])
            choice = self._get_choice_text(choice_list, options=["A", "B", "C", "D", "E", "F"])
            problem = f"{problem}\nChoices: {choice}"
            qid_hash = problem + answer
            hash_object = hashlib.sha256(qid_hash.encode())
            hex_dig = hash_object.hexdigest()
            unique_id = str(int(hex_dig, 16))
            unique_id = unique_id[:16]
            question_id = image_name.split('.')[0] + str(unique_id)
            self.question_list.append(problem)
            self.question_id_list.append(question_id)
            self.image_path_list.append(image_path)
            self.answer_list.append(answer)

    
    def __getitem__(self, idx):
        answer = f"The answer is {self.answer_list[idx]}."
        return self.question_id_list[idx], self.image_path_list[idx], self.question_list[idx], answer
      
    def __len__(self):
        return len(self.question_id_list)
    
    def _get_answer(self, answer, choices_list, options):
        if answer not in choices_list:
            raise ValueError(f"{answer} is not in choices_list")
        index_answer = choices_list.index(answer)
        answer = options[index_answer]
        return answer
    def _get_choice_text(self, choices, options):
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(options[i], c))
        choice_txt = " ".join(choice_list)
        #print(choice_txt)
        return choice_txt
    
    @staticmethod
    def get_question_prompt(question, use_CoT):
        if use_CoT:
            # refined_question = f"{question}\nAnswer the preceding multiple choice question. The last line of your response should be of the following format: '**FINAL ANSWER:**\nThe answer is X.' (without quotes) where X must be one of options. Think step by step before answering."
            refined_question = f"{question}\nAnswer the preceding multiple choice question. The format of your response should be of the following format: 'The answer is X.\nBECAUSE: xxx' (without quotes) where X must be one of options. Think step by step before answering."
        
        else:
            refined_question = f"{question}\nAnswer with the option letter from the given choices in the following format: 'The answer is X.' (without quotes) where X must be one of options."
        return refined_question
    @staticmethod
    def get_system_prompt(system_dtype):
        if system_dtype == "RAG":
            system_prompt = DATA_PROMPTS["RAG_VSR_MC"]
        elif system_dtype == "woRAG":
            system_prompt = DATA_PROMPTS["RAG_VSR_MC_woRAG"]
        elif system_dtype == "RAG_CoT_Refine":
            system_prompt = DATA_PROMPTS["RAG_VSR_MC_CoT_refine"]
        elif system_dtype == "RAG_CoT":
            system_prompt = DATA_PROMPTS["RAG_VSR_MC_CoT"]
        elif system_dtype == "RAG_Refine":
            system_prompt = DATA_PROMPTS["RAG_VSR_MC_refine"]
        elif system_dtype == "Self_Consistency":
            system_prompt = DATA_PROMPTS["Self_Consistency_VSR_MC"]
        else:
            raise ValueError(f"system_dtype: {system_dtype} is not supported.")
        return system_prompt
       

    @staticmethod
    def judge_answer(gt_answer, pred_answer):
        """The answer stype must be like: {The answer is X. xxx}"""
        pattern = re.compile(r'The answer is ([A-Z]).') 
        pred_answer = pred_answer.strip('{}')
        pred_answer = pred_answer.strip('[]')
        pred_answer = pred_answer.strip('"')
        pred_answer = pred_answer.strip()
        
        pred_choice = pattern.findall(pred_answer)

        gt_answer = gt_answer.strip('{}')
        gt_answer = gt_answer.strip('[]')
        gt_answer = gt_answer.strip('"')
        gt_answer = gt_answer.strip()
        gt_choice = pattern.findall(gt_answer)
        
        try:
            if len(pred_choice[0]) == 1:
                return pred_choice[0] == gt_choice[0]
        except:
            if len(gt_choice[0]) != 1:
                raise ValueError(f"gt_choice: {gt_choice}, pred_choice: {pred_answer}")
            else:
                print(f"gt_choice: {gt_choice}, pred_choice: {pred_answer}")
                return 'Bad'
    @staticmethod
    def judge_answer_for_mcts(gt_answer, pred_answer):
        """The answer stype must be like: {The answer is X. xxx}"""
       
        pattern = re.compile(r'The answer is ([A-Z]).') 
        pred_answer = pred_answer.strip('{}')
        pred_answer = pred_answer.strip('[]')
        pred_answer = pred_answer.strip('"')
        pred_answer = pred_answer.strip()
        pred_choice = pattern.findall(pred_answer)

        gt_answer = re.split(r'\*\*final answer:\*\*|\*\*FINAL ANSWER:\*\*|FINAL ANSWER:|Final Answer\:', gt_answer)[-1]
        pred_answer = pred_answer.strip('**')
        gt_answer = gt_answer.strip('{}')
        gt_answer = gt_answer.strip('[]')
        gt_answer = gt_answer.strip('"')
        gt_answer = gt_answer.strip()
        gt_choice = pattern.findall(gt_answer)
        try:
            if gt_choice[0] == 'X' or pred_choice[0] == 'X':
                return 0
            if len(pred_choice[0]) == 1:
                if pred_choice[0] == gt_choice[0]:
                    return 1
                else:
                    return 0
        except:
            try:
                if len(gt_choice[0]) == 1:
                    return 0
            except:
                print(f"gt_choice: {gt_answer}, pred_choice: {pred_answer}")
                return 0
                # raise ValueError(f"gt_choice: {gt_choice}, pred_choice: {pred_choice}")
                
                
    @staticmethod
    def judge_early_stop_for_mcts(answer_list):
        """The answer stype must be like: {The answer is X. xxx}"""
        pattern = re.compile(r'The answer is ([A-Z]).') 
        pred_choice_list = []
        for pred_answer in answer_list:
            pred_answer = pred_answer.strip('{}')
            pred_answer = pred_answer.strip('[]')
            pred_answer = pred_answer.strip('"')
            pred_answer = pred_answer.strip()
            pred_choice = pattern.findall(pred_answer)
            try:
                if len(pred_choice[0]) == 1:
                    pred_choice_list.append(pred_choice[0])
            except:
                return False
        for pred_choice in pred_choice_list:
            if pred_choice != pred_choice_list[0]:
                return False
        return True
               