import os 
import torch
import numpy as np
import pandas as pd
import json
import re
from .data_prompt import DATA_PROMPTS
class ScienceQA(torch.utils.data.Dataset):
    def __init__(self, data_root, split):
        self.name = 'ScienceQA'
        self.split = split
        problems = json.load(open(os.path.join(data_root, 'problems.json')))
        pid_splits = json.load(open(os.path.join(data_root, 'pid_splits.json')))
        # captions = json.load(open(caption_file))["captions"]  # TODO 暂时没加
        
       
        self.question_id_list = pid_splits['%s' % (self.split)] 
        self.question_list, self.image_path_list, self.answer_list, self.context_list = [], [], [], []
        self.choice_list, self.lecture_list, self.solution_list = [], [], []
        for qid in self.question_id_list:
            
            problem = problems[qid]
            self.question_list.append(self._get_question_text(problem))
            is_image = problem['image']
            img_dir = os.path.join(data_root, problem['split'])
            if is_image:
                img_path = os.path.join(img_dir, f'{qid}/{is_image}')
            else:
                img_path = None
            self.image_path_list.append(img_path)
            self.answer_list.append(self._get_answer(problem, options=["A", "B", "C", "D", "E"]))
            self.context_list.append(self._get_context_text(problem, use_caption=False))
            self.choice_list.append(self._get_choice_text(problem, options=["A", "B", "C", "D", "E"]))
            self.lecture_list.append(self._get_lecture_text(problem))
            self.solution_list.append(self._get_solution_text(problem))
        self.img_feat_path_list, self.text_feat_path_list = [], []

    
    def __getitem__(self, idx):
        # ref: https://github.com/lupantech/ScienceQA/blob/main/models/base_prompt.py
        # TODO question with context. Answer with lecture and solution.
        question = f"{self.question_list[idx]}\nContext: {self.context_list[idx]}\nChoices: {self.choice_list[idx]}\n" 
        answer = f"The answer is {self.answer_list[idx]}.\nBECAUSE: {self.lecture_list[idx]} {self.solution_list[idx]}"
        # answer_dict = {"answer": f"The answer is {self.answer_list[idx]}.", "lecture": self.lecture_list[idx], "solution": self.solution_list[idx]}
        return self.question_id_list[idx], self.image_path_list[idx], question, answer
      
    def __len__(self):
        return len(self.question_id_list)



    def _get_question_text(self, problem):
        question = problem['question']
        return question


    def _get_context_text(self, problem, use_caption):
        txt_context = problem['hint']
        img_context = problem['caption'] if use_caption else ""
        context = " ".join([txt_context, img_context]).strip()
        if context == "":
            context = "N/A"
        return context


    def _get_choice_text(self, probelm, options):
        choices = probelm['choices']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(options[i], c))
        choice_txt = " ".join(choice_list)
        #print(choice_txt)
        return choice_txt


    def _get_answer(self, problem, options):
        return options[problem['answer']]


    def _get_lecture_text(self, problem):
        lecture = problem['lecture']
        return lecture

    def _get_solution_text(self, problem):
        solution = problem['solution']
        return solution
    @staticmethod
    def get_system_prompt(system_dtype):

        if system_dtype == "RAG":
            system_prompt = DATA_PROMPTS["RAG_ScienceQA"]
        elif system_dtype == "woRAG":
            system_prompt = DATA_PROMPTS["RAG_ScienceQA_woRAG"]
        elif system_dtype == "RAG_CoT_Refine":
            system_prompt = DATA_PROMPTS["RAG_ScienceQA_CoT_refine"]
        elif system_dtype == "RAG_CoT":
            system_prompt = DATA_PROMPTS["RAG_ScienceQA_CoT"]
        elif system_dtype == "RAG_Refine":
            system_prompt = DATA_PROMPTS["RAG_ScienceQA_refine"]
        elif system_dtype == "Self_Consistency":
            system_prompt = DATA_PROMPTS["Self_Consistency_ScienceQA_v1"]
        
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
                raise ValueError(f"gt_choice: {gt_choice}, pred_choice: {pred_choice}")
            else:
                print(f"gt_choice: {gt_choice}, pred_choice: {pred_choice}")
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

        gt_answer = gt_answer.strip('{}')
        gt_answer = gt_answer.strip('[]')
        gt_answer = gt_answer.strip('"')
        gt_answer = gt_answer.strip()
        gt_choice = pattern.findall(gt_answer)
        try:
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
                pred_answer = pred_answer.split('.')[0]
                pred_choice_list.append(pred_answer)
        for pred_choice in pred_choice_list:
            if pred_choice != pred_choice_list[0]:
                return False
        return True
               