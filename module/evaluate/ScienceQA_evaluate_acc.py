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
import asyncio
from module.model.llm import openai_complete_if_cache, limit_async_func_call, vqa_model_func

def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    if len(total_pd) == 0:
        return -1
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc


def get_scores(result_file, data_file):
    # read result file
    # results = json.load(open(result_file))["results"] #TODO 绕过用json
    results_df = pd.read_csv(result_file)


    num = len(results_df)
    assert num == 4241 
    #print("number of questions:", num)

    # read data file
    sqa_data = json.load(open(data_file))

    # construct pandas data
    sqa_pd = pd.DataFrame(sqa_data).T
    res_pd = sqa_pd[sqa_pd['split'] == 'test']  # test set
    # update data
    failures = []
    for index, row in res_pd.iterrows():
        if int(index) not in results_df['question_id'].values:
            print(f'{index} not in results_df')
            continue
        res_pd.loc[index, 'no_context'] = True if (not row['hint'] and not row['image']) else False
        res_pd.loc[index, 'has_text'] = True if row['hint'] else False
        res_pd.loc[index, 'has_image'] = True if row['image'] else False
        res_pd.loc[index, 'has_text_image'] = True if (row['hint'] and row['image']) else False
        label = row['answer']
        pred_choices = results_df.loc[results_df['question_id'] == int(index), 'pred_choice'].values[0]

        options = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'Unknown':-1}
        try:
            pred = options[pred_choices]    
        except:
            print(f'{pred_choices} is not in options')
            pred = -1
        res_pd.loc[index, 'pred'] = pred
        res_pd.loc[index, 'true_false'] = (label == pred)
        if label != pred:
            failures.append({'question_id': index, 'label': label, 'pred': pred})
    # accuracy scores
    acc_average = len(res_pd[res_pd['true_false'] == True]) / num * 100
    #assert result_file.split('_')[-1] == "{:.3f}.json".format(acc_average)

    scores = {
        'acc_natural':
        get_acc_with_contion(res_pd, 'subject', 'natural science'),
        'acc_social':
        get_acc_with_contion(res_pd, 'subject', 'social science'),
        'acc_language':
        get_acc_with_contion(res_pd, 'subject', 'language science'),
        'acc_has_text':
        get_acc_with_contion(res_pd, 'has_text', True),
        'acc_has_image':
        get_acc_with_contion(res_pd, 'has_image', True),
        'acc_no_context':
        get_acc_with_contion(res_pd, 'no_context', True),
        'acc_grade_1_6':
        get_acc_with_contion(res_pd, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
        'acc_grade_7_12':
        get_acc_with_contion(res_pd, 'grade', ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
        'acc_average':
        "{:.2f}".format(acc_average),
    }
    topics = ['punctuation', 'literacy-in-science', 'verbs', 'pronouns', 'civics', 'culture', 'word-study', 'economics', 'physics', 'units-and-measurement', 'science-and-engineering-practices', 'reading-comprehension', 'global-studies', 'grammar', 'figurative-language', 'us-history', 'writing-strategies', 'world-history', 'reference-skills', 'biology', 'earth-science', 'phonological-awareness', 'capitalization', 'chemistry', 'vocabulary', 'geography']
    for t in topics:
        scores['acc_' + t] = get_acc_with_contion(res_pd, 'topic', t)

    return scores, failures


def print_scores(scores):
    latex_output = ""
    for key, score in scores.items():
        print(f"{key[4:]}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print(latex_output)


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
    
    if 'pred_choice' not in results_df.columns or True:
        temp_pred_choice_list = []
        # asyncio.run(get_choice(args.result_file))  # TODO 这里是用LLM来洗数据，可以按照论文的格式，直接用模板来洗数据
        pattern = re.compile(r'The answer is ([A-Z])')
        
        failed_answer = []
        
        for index, pred_answer in enumerate(temp_pred_answer_list):
            
            try:
                for i in '#$%^&*()[].,;:{}':
                    pred_answer = pred_answer.replace(i, '')
                pred_answer = pred_answer.strip()
                res = pattern.findall(pred_answer)
            except:
                res = []
            try:
                if len(res[0]) == 1:
                    temp_pred_choice_list.append(res[0])
                else:
                    failed_answer.append({'question_id':temp_question_id_list[index], 'pred_answer':pred_answer})
                    temp_pred_choice_list.append('Unknown')
            except:
                print(f"{pred_answer} is not in pattern")
                failed_answer.append({'question_id':temp_question_id_list[index], 'pred_answer':pred_answer})
                temp_pred_choice_list.append('Unknown')
        results_df['pred_choice'] = temp_pred_choice_list
        results_df.to_csv(args.result_file, index=False)
        failed_answer_df = pd.DataFrame(failed_answer)
        failed_answer_df.to_csv(args.result_file.replace('.csv', '_failed_answer.csv'), index=False)
    scores, failures = get_scores(args.result_file, args.data_file)
    print_scores(scores)
    # save failures
    with open(args.result_file.replace('.csv', '_failures.json'), 'w') as f:
        json.dump(failures, f)