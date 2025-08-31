import os
import json
from tqdm import tqdm
from latex2sympy2 import latex2sympy
import numpy as np
import re
from copy import deepcopy
import time  # 引入time模块
from math import *
import pandas as pd
import pickle
import json
from tqdm import tqdm  # Assuming tqdm is imported to show progress bar
import time
import string
import copy as cp
import os
import csv
def load(f, fmt=None):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split('.')[-1]
    return handlers[suffix](f)

# LOAD & DUMP
def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)

def can_infer_option(answer, choices):

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    match = re.search(r'\boxed\{(.*?)\}', answer_mod)

    if match:
        # 提取到的内容
        answer_mod = match.group(1)

    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3:
                print(f'A might be a quantifier in the string: {answer}.')
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False


# def can_infer_text(answer, choices):
#     answer = answer.lower()
#     assert isinstance(choices, dict)
#     for k in choices:
#         assert k in string.ascii_uppercase
#         choices[k] = str(choices[k]).lower()
#     cands = []
#     for k in choices:
#         if choices[k] in answer:
#             cands.append(k)
#     if len(cands) == 1:
#         return cands[0]
#     return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt
    # return copt if copt else can_infer_text(answer, choices)


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


# def get_gpt4_ICE():
#     example_1 = """
# Hint: Please answer the question and provide the final answer at the end.\n
# Question: Which number is missing?\n
# Model response: The number missing in the sequence is 14.\n
# Extracted answer: 14
# """

#     example_2 = """
# Hint: Please answer the question and provide the final answer at the end.\n
# Question: What is the fraction of females facing the camera?\n
# Model response: The fraction of females facing the camera is 0.6,
# which means that six out of ten females in the group are facing the camera.\n
# Extracted answer: 0.6
# """

#     example_3 = """
# Hint: Please answer the question and provide the final answer at the end.\n
# Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
# Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
# Extracted answer: 1.45
# """

#     example_4 = """
# Hint: Please answer the question and provide the final answer at the end.\n
# Question: Between which two years does the line graph saw its maximum peak?\n
# Model response: The line graph saw its maximum peak between 2007 and 2008.\n
# Extracted answer: [2007, 2008]
# """

#     example_5 = """
# Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
# Question: What fraction of the shape is blue?\n
# Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
# Model response: The correct answer is (B) 8/11.\n
# Extracted answer: B
# """

#     return [example_1, example_2, example_3, example_4, example_5]


def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Which number is missing?\n
Model response: The answer is \\boxed{14}.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The answer is \\boxed{0.6}.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: The answer is \\boxed{1.45}.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The answer is \\boxed{2007, 2008}.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The answer is \\boxed{B}.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]


def build_mathv_gpt4_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    question = line['question']
    prediction = str(line['pred_answer'])
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction + '\n'
    prompt += 'Extracted answer:'
    return prompt

def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}


def post_check(line, prefetch=False):
    res = None
    ans = line['answer']
    response = line['pred_answer'] if prefetch else line['extract']
    try:
        if len(eval(line['choices'])) > 0:
            ans = line['answer']
            choices = list_to_dict(eval(line['choices']))
            res = can_infer(response, choices)
            if prefetch:
                return res
        else:
            res = str(response)
            ans = str(ans)
    except ValueError:
        pass

    if is_equal(res, ans):
        return res if prefetch else True
    else:
        return False


async def MATH_V_auxeval(line, vqa_model, api_extra_body, retry=5):
    full_prompt = build_mathv_gpt4_prompt(line)
    log = ''
    temp_res = post_check(line, prefetch=True)
    if temp_res:
        res = temp_res
        return dict(log='Prefetch succeed', res=res)
    for i in range(retry):
        prediction = line['pred_answer']
        prompt= [{'role': 'user', 'content': f"{full_prompt}"}]
        res = await vqa_model(prompt=prompt, extra_body=api_extra_body)
        # res = model.generate(prompt, temperature=i * 0.5)
        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log=log, res=res)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='')

async def MATH_V_auxeval_for_MCTS(line, vqa_model, api_extra_body, retry=1):
    full_prompt = build_mathv_gpt4_prompt(line)
    log = ''
    # temp_res = post_check(line, prefetch=True)  #! forbidden
    # if temp_res:
    #     res = temp_res
    #     return dict(log='Prefetch succeed', res=res)
    
    prompt= [{'role': 'user', 'content': f"{full_prompt}"}]
    res = await vqa_model(prompt=prompt, extra_body=api_extra_body)
    return res

