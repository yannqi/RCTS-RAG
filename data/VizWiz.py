import os 
import torch
import numpy as np
import pandas as pd
import json
import re
from .data_prompt import DATA_PROMPTS
from misc.logger import logger
from module.evaluate.VizWiz_evaluate_acc import EvalAIAnswerProcessor

# ref https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/eval/vqa/evaluate_vqa.py
class VizWiz(torch.utils.data.Dataset):
    def __init__(self, data_root, split='train'):
        self.name = 'VizWiz'
        self.split = split
        self.data_root = data_root

        jsonl_root = os.path.join(self.data_root + f'vizwiz_{split}.jsonl')
        data_list = []
        with open(jsonl_root, 'r', encoding='utf-8') as file:
            for line in file:
                data_list.append(json.loads(line.strip()))
  
        self.question_id_list = [data_list[i]['question_id'] for i in range(len(data_list))]
        self.image_path_list = [data_list[i]['image'].split('/vizwiz/')[-1] for i in range(len(data_list))]
        self.question_list = [data_list[i]['question'] for i in range(len(data_list))]
        self.answer_list = [data_list[i]['answer'] for i in range(len(data_list))]

        knowledge_base_annotation = json.load(open(os.path.join(self.data_root, 'vizwiz_train_annotations.json'), 'r'))['annotations']
        self.question_id2gt_answers = {}
        for item in knowledge_base_annotation:
            question_id = str(item['question_id'])
            answers = [answer['answer'] for answer in item['answers']]
            self.question_id2gt_answers[question_id] = answers
        
        
    def __getitem__(self, idx):
        question_id = self.question_id_list[idx]
        question = self.question_list[idx]
        answer = self.answer_list[idx]
        image_path = os.path.join(self.data_root, self.image_path_list[idx])
        return question_id, image_path, question, answer
    def __len__(self):
        return len(self.question_id_list)


    @staticmethod
    def get_question_prompt(question, use_CoT):
        base_prompt = 'Answer the question using a single word or phrase.'
        vizwiz_prompt = "When the provided information is insufficient, respond with 'unanswerable'."
        if use_CoT:
            # refined_question = f"{question}\nAnswer the preceding multiple choice question. The last line of your response should be of the following format: '**FINAL ANSWER:**\nThe answer is X.' (without quotes) where X must be one of options. Think step by step before answering."
            refined_question = f"{question}\n{base_prompt}. The format of your response should be of the following format: '[Your Answer]\nBECAUSE: [Your Reasoning]' (without quotes). {vizwiz_prompt} Think step by step before answering."
        else:
            refined_question = f"{question}\n{base_prompt} {vizwiz_prompt}"
        return refined_question
    
    @staticmethod
    def get_system_prompt(system_dtype):

        if system_dtype == "RAG":
            system_prompt = DATA_PROMPTS["RAG_VizWiz"]
        elif system_dtype == "woRAG":
            system_prompt = DATA_PROMPTS["RAG_VizWiz_woRAG"]
        elif system_dtype == "RAG_CoT_Refine":
            system_prompt = DATA_PROMPTS["RAG_VizWiz_CoT_refine"]
        elif system_dtype == "RAG_CoT":
            system_prompt = DATA_PROMPTS["RAG_VizWiz_CoT"]
        elif system_dtype == "RAG_Refine":
            system_prompt = DATA_PROMPTS["RAG_VizWiz_refine"]
        elif system_dtype == "Self_Consistency":
            system_prompt = DATA_PROMPTS["Self_Consistency_VizWiz"]
        else:
            raise ValueError(f"system_dtype: {system_dtype} is not supported.")
        return system_prompt
       



    @staticmethod
    def judge_answer(gt_answer: str, pred_answer: str):
        """The answer stype must be like: {The answer is X. xxx}"""
        answer_processor = EvalAIAnswerProcessor()
        pred_answer = answer_processor(pred_answer)
        
        gt_answer = gt_answer.lower()
        pred_answer = pred_answer.lower()
        
        
        if gt_answer == pred_answer:
            return True
        else:
            return 'Bad'
    


    def judge_answer_for_mcts(self, pred_answer, sample_question_id):
        answer_processor = EvalAIAnswerProcessor()
        pred_answer = re.split(r'\*\*final answer:\*\*|\*\*FINAL ANSWER:\*\*|FINAL ANSWER:|Final Answer\:', pred_answer)[-1]
        pred_answer = pred_answer.strip()
        pred_answer = pred_answer.strip('\n')
        pred_answer = pred_answer.strip('**')
        pred_answer = answer_processor(pred_answer)
        sample_question_id = str(sample_question_id.split('_')[-1])
        gt_answers = self.question_id2gt_answers[sample_question_id]
        
            
        def _compute_answer_scores(raw_answers):
            """
            compute the accuracy (soft score) of human answers
            """
            answers = [answer_processor(a) for a in raw_answers]
            assert len(answers) == 10
            gt_answers = list(enumerate(answers))
            unique_answers = set(answers)
            unique_answer_scores = {}

            for unique_answer in unique_answers:
                accs = []
                for gt_answer in gt_answers:
                    other_answers = [item for item in gt_answers if item != gt_answer]
                    matching_answers = [
                        item for item in other_answers if item[1] == unique_answer
                    ]
                    acc = min(1, float(len(matching_answers)) / 3)
                    accs.append(acc)
                unique_answer_scores[unique_answer] = sum(accs) / len(accs)

            return unique_answer_scores
        
        unique_answer_scores = _compute_answer_scores(gt_answers)
        score = unique_answer_scores.get(pred_answer, 0.0)
        
        return score
        
       
                # raise ValueError(f"gt_choice: {gt_choice}, pred_choice: {pred_choice}")
                
                
    @staticmethod
    def judge_early_stop_for_mcts(answer_list):
        
        answer_processor = EvalAIAnswerProcessor()
        
        extract_answer_list = []
        for pred_answer in answer_list:
            # pred_answer = re.split(r'\*\*final answer:\*\*|\*\*FINAL ANSWER:\*\*|FINAL ANSWER:|Final Answer\:', pred_answer)[-1]
            pred_answer = pred_answer.strip()
            pred_answer = pred_answer.strip('\n')
            pred_answer = pred_answer.strip('**')
            pred_answer = answer_processor(pred_answer)
            pred_answer = pred_answer.lower()
            extract_answer_list.append(pred_answer)
        temp_extract_answer = extract_answer_list[0]
        for extract_answer in extract_answer_list:
            if extract_answer != temp_extract_answer:
                return False
        return True

    @staticmethod
    def judge_answer_for_mcts_self_eval(gt_answer, pred_answer):
        answer_processor = EvalAIAnswerProcessor()
        
        # pred_answer = re.split(r'\*\*final answer:\*\*|\*\*FINAL ANSWER:\*\*|FINAL ANSWER:|Final Answer\:', pred_answer)[-1]
        pred_answer = pred_answer.strip()
        pred_answer = pred_answer.strip('\n')
        pred_answer = pred_answer.strip('**')
        pred_answer = answer_processor(pred_answer)
        pred_answer = pred_answer.lower()
        
        # gt_answer = re.split(r'\*\*final answer:\*\*|\*\*FINAL ANSWER:\*\*|FINAL ANSWER:|Final Answer\:', gt_answer)[-1]
        gt_answer = gt_answer.strip()
        gt_answer = gt_answer.strip('\n')
        gt_answer = gt_answer.strip('**')
        gt_answer = answer_processor(gt_answer)
        gt_answer = gt_answer.lower()
        if gt_answer == pred_answer:
            return 1
        else:
            return 0
        