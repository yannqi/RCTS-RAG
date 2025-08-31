import os 
import torch
import numpy as np
import pandas as pd
import json
import re
import ast
from io import BytesIO
from datasets import load_dataset, concatenate_datasets
from .data_prompt import DATA_PROMPTS
from .dataset_prepare.Mini_Image_resize import resize_image
import hashlib
# ref: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/prompts.yaml
CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

class MMMU(torch.utils.data.Dataset):
    def __init__(self, data_root, split): 
        self.name = 'MMMU'
        self.split = split
        self.question_id_list = []
        self.question_list = []
        self.image_path_list = []
        self.answer_list = []
        
        save_image_path = os.path.join(data_root, f'{split}_images')
        os.makedirs(save_image_path, exist_ok=True)

        # run for each subject
        sub_dataset_list = []
        for subject in CAT_SHORT2LONG.values():
            sub_dataset = load_dataset(data_root, subject, split=split)
            sub_dataset_list.append(sub_dataset)

        # merge all dataset
        dataset = concatenate_datasets(sub_dataset_list)
        self.cnt = 0 
        for data in dataset:
            question_id = data['id']
            
            
            prompt, answer = self.mmmu_doc_to_text(data)
            # if len(ast.literal_eval(str(data["options"]))) == 0:
            #     continue # skip the question with open answer
            
            # self.image_path_list.append(temp_image_list)
            
            qid_hash = prompt + question_id + answer
            hash_object = hashlib.sha256(qid_hash.encode())
            hex_dig = hash_object.hexdigest()
            unique_id = str(int(hex_dig, 16))
            question_id = unique_id
      
            
            images = self.origin_mmmu_doc_to_visual(data)
            temp_image_list = []
            for i, image in enumerate(images):
                image_name = f"{question_id}_{i}.png"
                image_path = os.path.join(save_image_path, image_name)
                temp_image_list.append(image_path)
                if os.path.exists(image_path):
                    continue
                image.save(image_path, format='PNG')
                resize_image(image_path, image_path)
            self.question_id_list.append(question_id)
            self.image_path_list.append(temp_image_list)
            self.answer_list.append(answer)            
            self.question_list.append(prompt)  
        csv_path = os.path.join(data_root, f'{split}.csv')
        if not os.path.exists(csv_path):
            df = pd.DataFrame({'question_id': self.question_id_list, 'image_path': self.image_path_list, 'question': self.question_list, 'answer': self.answer_list})
            df.to_csv(os.path.join(data_root, f'{split}.csv'), index=False)   
        print(f'During MMMU_preproc in Evaluation, {self.cnt} open questions are re-formulated to multi-choice ones. ')

    def __getitem__(self, idx):
        # ref: https://github.com/lupantech/MMMU_Pro/blob/main/models/base_prompt.py
        # TODO question with context. Answer with lecture and solution.
        question = self.question_list[idx]
        answer = f"The answer is {self.answer_list[idx]}."

        return self.question_id_list[idx], self.image_path_list[idx], question, answer
     
    def __len__(self):
        return len(self.question_id_list)

    def origin_mmmu_doc_to_visual(self, doc):
        visual = []
        for i in range(1,8):
            if not doc[f'image_{i}']:
                break
            visual.append(doc[f'image_{i}'])
        return visual



    def _MMMU_preproc(self, doc):
    # ref https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/utils/multiple_choice.py
        options = ast.literal_eval(str(doc["options"]))
        options.append(doc["answer"])
        options.append("Other Answers")
        doc["options"] = str(options)
        doc["answer"] = 'A'
        self.cnt += 1

        return doc
        

    def mmmu_doc_to_text(self, doc):
        
        def _parse_options(options):
            option_letters = [chr(ord("A") + i) for i in range(len(options))]
            choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
            return choices_str
       
        def _construct_prompt(doc):
            question = doc["question"]
            if len(ast.literal_eval(str(doc["options"]))) == 0:
                doc = self._MMMU_preproc(doc)
            parsed_options = _parse_options(ast.literal_eval(str(doc["options"])))
            question = f"{question}\nChoices:\n{parsed_options}"
            answer = doc["answer"]
            return question, answer
        
        def _replace_images_tokens(input_string):
            # if "<image 2>" in input_string:
            #     return input_string
            for i in range(1, 8):
                question_text = f"<image {i}>"
                query_text = "<image>"
                if question_text in input_string:
                    input_string = input_string.replace(question_text, query_text)
            return input_string
        
        question, answer = _construct_prompt(doc)
    
        return _replace_images_tokens(question), answer
   

    
    

    # ref https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/vlm/internvl/utils.py
    @staticmethod
    def get_question_prompt(question, use_CoT):
        if use_CoT:
            # refined_question = f"{question}\nAnswer the preceding multiple choice question. The last line of your response should be of the following format: '**FINAL ANSWER:**\nThe answer is X.' (without quotes) where X must be one of options. Think step by step before answering."
            refined_question = f"{question}\nAnswer the preceding multiple choice question. The format of your response should be of the following format: 'The answer is X.\nBECAUSE: xxx' (without quotes) where X must be one of options. Think step by step before answering."  
        else:
            refined_question = f"{question}\nAnswer with the option letter from the given choices in the following format: 'The answer is X.' (without quotes) where X must be one of options."
        # return refined_question
        # def build_mcq_cot_prompt(question):
        #     cot_prompt = (
        #         "Answer the preceding multiple choice question. The last line of your response should follow "
        #         "this format: 'Answer: The answer is X.' (without quotes), where X must be one of the options. "
        #         "If you are uncertain or the problem is too complex, make a reasoned guess based on the "
        #         "information provided. Avoid repeating steps indefinitely—provide your best guess even if "
        #         "unsure. Think step by step logically, considering all relevant information before answering."
        #     )
           
        #     prompt = question + '\n' + cot_prompt

        #     return prompt
        
        # def build_mcq_prompt(question):
        #     prompt = (
        #         "Answer the preceding multiple choice question. The format of your response should follow "
        #         "this format: 'The answer is X.' (without quotes), where X must be one of the options. "
        #     )
            
            # prompt = question + '\n' + prompt
            # return prompt
            
        # if use_CoT:
        #     refined_question = build_mcq_cot_prompt(question)
        # else:
        #     refined_question = build_mcq_prompt(question)

        return refined_question
    @staticmethod
    def get_system_prompt(system_dtype):

        if system_dtype == "RAG":
            system_prompt = DATA_PROMPTS["RAG_MMMU"]
        elif system_dtype == "woRAG":
            system_prompt = DATA_PROMPTS["RAG_MMMU_woRAG"]
        elif system_dtype == "RAG_CoT_Refine":
            system_prompt = DATA_PROMPTS["RAG_MMMU_CoT_refine"]
        elif system_dtype == "RAG_CoT":
            system_prompt = DATA_PROMPTS["RAG_MMMU_CoT"]
        elif system_dtype == "RAG_Refine":
            system_prompt = DATA_PROMPTS["RAG_MMMU_refine"]
        elif system_dtype == "Self_Consistency":
            system_prompt = DATA_PROMPTS["Self_Consistency_MMMU"]
        
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
            if gt_choice[0] == 'X' or pred_choice[0] == 'X':
                return 0
            if len(pred_choice[0]) == 1:
                return pred_choice[0] == gt_choice[0]
        except:
            if len(gt_choice[0]) != 1:
                raise ValueError(f"gt_choice: {gt_choice}, pred_choice: {pred_choice}")
            else:
                pred_choice = pred_answer.split('.')[0]
                gt_choice = gt_choice[0]
                return pred_choice == gt_choice
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
                # pred_answer = pred_answer.split('.')[0]
                # pred_choice_list.append(pred_answer)
                return False
        for pred_choice in pred_choice_list:
            if pred_choice != pred_choice_list[0]:
                return False
        return True
               