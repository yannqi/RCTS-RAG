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
from tqdm import tqdm



class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI
        copied from
        https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        'aint': "ain't",
        'arent': "aren't",
        'cant': "can't",
        'couldve': "could've",
        'couldnt': "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        'didnt': "didn't",
        'doesnt': "doesn't",
        'dont': "don't",
        'hadnt': "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        'hasnt': "hasn't",
        'havent': "haven't",
        'hed': "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        'hes': "he's",
        'howd': "how'd",
        'howll': "how'll",
        'hows': "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        'Im': "I'm",
        'Ive': "I've",
        'isnt': "isn't",
        'itd': "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        'itll': "it'll",
        "let's": "let's",
        'maam': "ma'am",
        'mightnt': "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        'mightve': "might've",
        'mustnt': "mustn't",
        'mustve': "must've",
        'neednt': "needn't",
        'notve': "not've",
        'oclock': "o'clock",
        'oughtnt': "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        'shant': "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        'shouldve': "should've",
        'shouldnt': "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": 'somebodyd',
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        'somebodyll': "somebody'll",
        'somebodys': "somebody's",
        'someoned': "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        'someonell': "someone'll",
        'someones': "someone's",
        'somethingd': "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        'somethingll': "something'll",
        'thats': "that's",
        'thered': "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        'therere': "there're",
        'theres': "there's",
        'theyd': "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        'theyll': "they'll",
        'theyre': "they're",
        'theyve': "they've",
        'twas': "'twas",
        'wasnt': "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        'weve': "we've",
        'werent': "weren't",
        'whatll': "what'll",
        'whatre': "what're",
        'whats': "what's",
        'whatve': "what've",
        'whens': "when's",
        'whered': "where'd",
        'wheres': "where's",
        'whereve': "where've",
        'whod': "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        'wholl': "who'll",
        'whos': "who's",
        'whove': "who've",
        'whyll': "why'll",
        'whyre': "why're",
        'whys': "why's",
        'wont': "won't",
        'wouldve': "would've",
        'wouldnt': "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        'yall': "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        'youd': "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        'youll': "you'll",
        'youre': "you're",
        'youve': "you've",
    }

    NUMBER_MAP = {
        'none': '0',
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
    }
    ARTICLES = ['a', 'an', 'the']
    PERIOD_STRIP = re.compile(r'(?!<=\d)(\.)(?!\d)')
    COMMA_STRIP = re.compile(r'(?<=\d)(\,)+(?=\d)')
    PUNCTUATIONS = [
        ';',
        r'/',
        '[',
        ']',
        '"',
        '{',
        '}',
        '(',
        ')',
        '=',
        '+',
        '\\',
        '_',
        '-',
        '>',
        '<',
        '@',
        '`',
        ',',
        '?',
        '!',
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(',', '').replace('?', '').replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + ' ' in in_text or ' ' + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, '')
            else:
                out_text = out_text.replace(p, ' ')
        out_text = self.PERIOD_STRIP.sub('', out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = ' '.join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace('\n', ' ').replace('\t', ' ').strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item





def eval_pred_list(pred_dict_list, disable_tqdm=False):
    pred_scores = []
    answer_processor = EvalAIAnswerProcessor()
    
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
    
    failures = []
    
    for entry in tqdm(pred_dict_list, disable=disable_tqdm):
        pred_answer = answer_processor(entry['pred_answer'])
        unique_answer_scores = _compute_answer_scores(entry['gt_answers'])
        score = unique_answer_scores.get(pred_answer, 0.0)
        pred_scores.append(score)
        if score == 0.0:
            failures.append(entry)
    accuracy = sum(pred_scores) / len(pred_scores)
    return accuracy, failures

# ref https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/eval/vqa/evaluate_vqa.py
def get_scores(result_file, data_file):
    # read result file

    results_df = pd.read_csv(result_file)

    num = len(results_df)
    results_df = pd.read_csv(result_file)
    qids_list = results_df['question_id'].tolist()
    pred_answer_list = results_df['pred_answer'].tolist()
    
    
    # read data file
    gt_annotation = json.load(open(data_file, 'r'))['annotations']
    question_id2gt_answers = {}
    for item in gt_annotation:
        question_id = item['question_id']
        answers = [answer['answer'] for answer in item['answers']]
        question_id2gt_answers[question_id] = answers
    

    # construct pandas data
    merged_dict_list = []
    for index, qid in enumerate(qids_list):
        pred_answer = pred_answer_list[index]
        pred_answer = re.split(r'\*\*final answer:\*\*|\*\*FINAL ANSWER:\*\*|FINAL ANSWER:|Answer\:|ANSWER\:', pred_answer)[-1]
        pred_answer = pred_answer.strip()
        pred_answer = pred_answer.strip('\n')
        pred_answer = pred_answer.strip('**')
        gt_answers = question_id2gt_answers[qid]
        merged_dict_list.append({'question_id': qid, 'pred_answer': pred_answer, 'gt_answers': gt_answers})
    print(len(merged_dict_list))

    scores, failures = eval_pred_list(merged_dict_list, disable_tqdm=False)    
    
    return scores, failures


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--result_file', type=str)
    args = parser.parse_args()
    
    print("GT_Data file: ", args.data_file)
    print("Pred_Result file: ", args.result_file)

    results_df = pd.read_csv(args.result_file)
    temp_pred_answer_list = results_df['pred_answer'].tolist()
    temp_question_id_list = results_df['question_id'].tolist()

    scores, failures = get_scores(args.result_file, args.data_file)
    print('The VizWiz accuracy is:', scores)
    # save failures
    with open(args.result_file.replace('.csv', '_failures.json'), 'w') as f:
        json.dump(failures, f)