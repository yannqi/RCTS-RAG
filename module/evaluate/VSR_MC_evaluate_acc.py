import os, sys
sys.path.append(os.getcwd())
import re
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


def get_scores(result_file, data_file):
    # read result file
    # results = json.load(open(result_file))["results"] #TODO 绕过用json
    results_df = pd.read_csv(result_file, index_col=0)


    num = len(results_df)
    # assert num == 4241 
    #print("number of questions:", num)

    # read data file

    data_json = []
    with open(data_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                j_line = json.loads(line)
                data_json.append(j_line)
    # update data
    def _get_choice_text(choices, options):
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(options[i], c))
        choice_txt = " ".join(choice_list)
        #print(choice_txt)
        return choice_txt
    def _get_answer(answer, choices_list, options):
        if answer not in choices_list:
            raise ValueError(f"{answer} is not in choices_list")
        index_answer = choices_list.index(answer)
        answer = options[index_answer]
        return answer
    
    failures = []
    total = 0
    correct = 0
    for i, gt_data in enumerate(data_json):  
        image_name = gt_data['image']
        subject = gt_data['subj']
        object = gt_data['obj']
        gt_answer = gt_data['relation']
        problem = f"The {subject} (is) ___ the {object}."
        choice_list = gt_data['options']
        choice = _get_choice_text(choice_list, options=["A", "B", "C", "D", "E", "F"])
        gt_answer = _get_answer(gt_answer, choice_list, options=["A", "B", "C", "D", "E", "F"])
        problem = f"{problem}\nChoices: {choice}"
        qid_hash = problem + gt_answer
        hash_object = hashlib.sha256(qid_hash.encode())
        hex_dig = hash_object.hexdigest()
        unique_id = str(int(hex_dig, 16))
        unique_id = unique_id[:16]
        question_id = image_name.split('.')[0] + str(unique_id)
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
    parser.add_argument('--data_file', type=str, default="./dataspace/ScienceQA_DATA/problems.json")
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
            pred_answer = re.split(r'final answer\:|answer\:', pred_answer)[-1]      
            pred_answer.strip()
            for i in '#$%^&*()[].,;:{}':
                pred_answer = pred_answer.replace(i, '')
            pattern = re.compile(r'The answer is ([A-Z])')
            res = pattern.findall(pred_answer)
            try:
                if len(res[0]) == 1:
                    temp_pred_choice_list.append(res[0])
                else:
                    failed_answer.append({'question_id':temp_question_id_list[index], 'pred_answer':pred_answer})
                    temp_pred_choice_list.append(pred_answer)
            except:
                print(f"{pred_answer} is not in pattern")
                failed_answer.append({'question_id':temp_question_id_list[index], 'pred_answer':pred_answer})
                temp_pred_choice_list.append(pred_answer)
                
        results_df['pred_choice'] = temp_pred_choice_list
        results_df.to_csv(args.result_file, index=False)
        failed_answer_df = pd.DataFrame(failed_answer)
        failed_answer_df.to_csv(args.result_file.replace('.csv', '_failed_answer.csv'), index=False)
    scores, failures = get_scores(args.result_file, args.data_file)
    print(f"Scores: {scores}")
    # save failures
    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(args.result_file.replace('.csv', '_failures.csv'), index=False)