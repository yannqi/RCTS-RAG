import os, sys
sys.path.append(os.getcwd())
from tqdm import tqdm
import time
import json
import asyncio
import copy
from module.evaluate.utils.MathV_utils import can_infer, load, dump, MATH_V_auxeval, post_check
import pandas as pd
import argparse
from latex2sympy2 import latex2sympy
from tqdm.asyncio import tqdm_asyncio
from module.model.llm import limit_async_func_call_with_multi_node, vqa_model_func_with_multi_node
from functools import partial
from collections import defaultdict
import re
import nest_asyncio
nest_asyncio.apply()

async def evaluate(result_file, data_file):
    
    vqa_model = limit_async_func_call_with_multi_node(10, max_node=len(os.environ.get("BASE_URL").split(';')))(
        partial(vqa_model_func_with_multi_node, hashing_kv=None)
    )
    api_extra_body = {
        "temperature": 0.1,
        "top_p": 0.001,
        "repetition_penalty": 1.05,
        "max_tokens": 512
    }
    storage_extract = result_file.replace('.csv', '_extract.xlsx')
    data = load(data_file, fmt='tsv')
    data.drop(columns=['image'], inplace=True)
    lt = len(data)
    lines = [data.iloc[i] for i in range(lt)]
    indices = [line['index'] for line in lines]
    # lines = load_jsonl(answer_file)
    results_df = pd.read_csv(result_file)
    qids_list = results_df['question_id'].tolist()
    pred_answer_list = results_df['pred_answer'].tolist()
    pred_lines = copy.deepcopy(lines)

    for index, qid in enumerate(qids_list):
        line = pred_lines[indices.index(qid)]
        pred_answer = pred_answer_list[index]
        # pred_answer = re.split(r'\*\*final answer:\*\*|\*\*FINAL ANSWER:\*\*|FINAL ANSWER:|Final Answer\:', pred_answer)[-1]
        # pred_answer = re.split(r':', pred_answer)[-1]
        pred_answer = re.split(r'BECASUE:|BECAUSE:|Because:|because:', pred_answer)[0]
        pred_answer = pred_answer.strip()
        pred_answer = pred_answer.strip('\n')
        pred_answer = pred_answer.strip('**')
        match = re.search(r'\\boxed\{([^{}]+)\}', pred_answer)
        if match:
            # 提取到的内容
            pred_answer = match.group(1)
            pred_answer = pred_answer.strip('$')
        line['pred_answer'] = pred_answer

    
    results_list = await tqdm_asyncio.gather(*[MATH_V_auxeval(line=pred_lines[indices.index(qid)], vqa_model=vqa_model, api_extra_body=api_extra_body) for qid in qids_list])
    assert os.path.exists(os.path.dirname(result_file.replace('.csv', '_extract.pkl'))) or os.path.dirname(result_file.replace('.csv', '_extract.pkl')) == ''
    if not os.path.exists(result_file.replace('.csv', '_extract.pkl')):
        dump({}, result_file.replace('.csv', '_extract.pkl'))
    ans = load(result_file.replace('.csv', '_extract.pkl'))

    for k, v in zip(qids_list, results_list):
        if k not in ans:
            ans[k] = {}
        ans[k]['log_extract'] = v['log']
        ans[k]['extract'] = v['res']
    
    ans_extract_list = []
    log_extract_list = []
    for idx in data['index']:
        try:
            ans_extract_list.append(ans[idx]['extract'])
            log_extract_list.append(ans[idx]['log_extract'])
        except: 
            ans_extract_list.append('Prefetch failed')
            log_extract_list.append('Prefetch failed')
            print(f"Warning: Prefetch failed for index {idx}")
    # data['extract'] = [ans[idx]['extract'] for idx in data['index']]
    # data['log_extract'] = [ans[idx]['log_extract'] for idx in data['index']]
    data['extract'] = ans_extract_list
    data['log_extract'] = log_extract_list
    dump(data, storage_extract)

    return storage_extract

def MATH_V_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    fetch = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        if item['log_extract'] == 'Prefetch failed':
            continue
        tot['Overall'] += 1
        tot[cate] += 1
        
        if item['log_extract'] == 'Prefetch succeed':
            fetch['Overall'] += 1
            fetch[cate] += 1
        if post_check(item, prefetch=False):
            hit['Overall'] += 1
            hit[cate] += 1  

    res = defaultdict(list)
    for k in tot.keys():
        res['Subject'].append(k)
        res['tot'].append(tot[k])
        res['prefetch'].append(fetch[k])
        res['hit'].append(hit[k])
        res['prefetch_rate'].append(fetch[k] / tot[k] * 100)
        res['acc'].append(hit[k] / tot[k] * 100)
    res = pd.DataFrame(res).sort_values('Subject', ignore_index=True)
    return res
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--result_file', type=str)
    args = parser.parse_args()
    storage_extract = asyncio.run(evaluate(result_file=args.result_file, data_file=args.data_file))
    # storage_extract = args.result_file.replace('.csv', '_extract.xlsx')
    score = MATH_V_acc(storage_extract)
    score_pth = storage_extract.replace('.xlsx', '_score.csv')
    dump(score, score_pth)
    print(score)
    # return score
