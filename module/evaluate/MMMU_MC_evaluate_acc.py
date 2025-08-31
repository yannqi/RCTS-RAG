import os, sys
sys.path.append(os.getcwd())
import re
import ast
import json
import argparse
import warnings
import pandas as pd
from functools import partial
warnings.filterwarnings('ignore')
from tqdm.asyncio import tqdm_asyncio
import hashlib
import asyncio
from module.model.llm import openai_complete_if_cache, limit_async_func_call, vqa_model_func
from module.evaluate.VizWiz_evaluate_acc import EvalAIAnswerProcessor
from data.MMMU import CAT_SHORT2LONG
from datasets import load_dataset, concatenate_datasets

1
def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string

def mmmu_doc_to_text(doc):
      
    def _MMMU_preproc( doc):
    # ref https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/utils/multiple_choice.py
        options = ast.literal_eval(str(doc["options"]))
        options.append(doc["answer"])
        options.append("Other Answers")
        doc["options"] = str(options)
        doc["answer"] = 'A'
        return doc
        
    def _parse_options(options):
        option_letters = [chr(ord("A") + i) for i in range(len(options))]
        choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
        return choices_str
    
    def _construct_prompt(doc):
        question = doc["question"]
        if len(ast.literal_eval(str(doc["options"]))) == 0:
            doc = _MMMU_preproc(doc)
        parsed_options = _parse_options(ast.literal_eval(str(doc["options"])))
        question = f"{question}\nChoices:\n{parsed_options}"
        return question
    
    def _replace_images_tokens(input_string):
        # if "<image 2>" in input_string:
        #     return input_string
        for i in range(1, 8):
            question_text = f"<image {i}>"
            query_text = "<image>"
            if question_text in input_string:
                input_string = input_string.replace(question_text, query_text)
        return input_string
    
    question = _construct_prompt(doc)
    
    return _replace_images_tokens(question)
    # return question


def get_scores(result_file, data_file):
    # read result file
    # results = json.load(open(result_file))["results"] #TODO 绕过用json
    results_df = pd.read_csv(result_file, index_col=0)


    num = len(results_df)

    # read data file
    if 'MMMU_Pro' in data_file:
        dataset = load_dataset(data_file, split='test')
    else:
        # run for each subject
        sub_dataset_list = []
        for subject in CAT_SHORT2LONG.values():
            sub_dataset = load_dataset(data_file, subject, split='dev')
            sub_dataset_list.append(sub_dataset)

        # merge all dataset
        dataset = concatenate_datasets(sub_dataset_list)

    # update data

    failures = []
    total = 0
    correct = 0
    for data in dataset:      
        question_id = data['id']
        
        prompt = mmmu_doc_to_text(data)
        gt_answer = data['answer']
        qid_hash = prompt + question_id + gt_answer
        hash_object = hashlib.sha256(qid_hash.encode())
        hex_dig = hash_object.hexdigest()
        unique_id = str(int(hex_dig, 16))
        question_id = unique_id
        try:
            pred_choice = results_df.loc[question_id]['pred_choice']
        except:
            print(f"Warning: Question {question_id} not found in result file!")
            continue

        if str(gt_answer) == str(pred_choice):
            correct += 1
        else:
            failures.append({'question_id': question_id, 'gt_answer': gt_answer, 'pred_answer': pred_choice})
        total += 1
    return correct / total, failures
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--result_file', type=str)
    args = parser.parse_args()
    
    print("Data file: ", args.data_file)
    print("Result file: ", args.result_file)

    
    results_df = pd.read_csv(args.result_file)
    temp_pred_answer_list = results_df['pred_answer'].tolist()
    temp_question_id_list = results_df['question_id'].tolist()
    answer_processor = EvalAIAnswerProcessor()
    if 'pred_choice' not in results_df.columns or True:
        temp_pred_choice_list = []
        # asyncio.run(get_choice(args.result_file))  # TODO 这里是用LLM来洗数据，可以按照论文的格式，直接用模板来洗数据

        failed_answer = []
        
        for index, pred_answer in enumerate(temp_pred_answer_list):
            # pred_answer = re.split(r'final answer\:|answer\:', pred_answer)[-1]      
            pred_answer.strip()
            for i in '#$%^&*()[],;:{}':
                pred_answer = pred_answer.replace(i, '')
            pattern = re.compile(r'The answer is ([A-Z]).')
            res = pattern.findall(pred_answer)
            try:
                if len(res[0]) == 1:
                    temp_pred_choice_list.append(res[0])
                else:
                    failed_answer.append({'question_id':temp_question_id_list[index], 'pred_answer':pred_answer})
                    temp_pred_choice_list.append('Unknown')
            except:
                pred_answer = pred_answer.split('.')[0]
                if len(pred_answer) == 1:
                    temp_pred_choice_list.append(pred_answer)
                else:
                    print(f"{pred_answer} is not in pattern")
                    failed_answer.append({'question_id':temp_question_id_list[index], 'pred_answer':pred_answer})
                    temp_pred_choice_list.append('Unknown')
                
        results_df['pred_choice'] = temp_pred_choice_list
        results_df.to_csv(args.result_file, index=False)
        failed_answer_df = pd.DataFrame(failed_answer)
        failed_answer_df.to_csv(args.result_file.replace('.csv', '_failed_answer.csv'), index=False)
    scores, failures = get_scores(args.result_file, args.data_file)
    print(f"Scores: {scores}")
    # save failures
    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(args.result_file.replace('.csv', '_failures.csv'), index=False)