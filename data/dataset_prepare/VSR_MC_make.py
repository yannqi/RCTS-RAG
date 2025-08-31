import os 
import numpy as np
import json
import copy

option_list = ['inside', 'away from', 'right of', 'over', 'surrounding', 'near', 'far away from', 'beside', 'under', 'contains', 'off', 'next to', 'in front of', 'in', 'touching', 'at the left side of', 'on', 'facing', 'below', 'left of', 'at the right side of', 'behind', 'attached to', 'on top of', 'above', 'at the side of', 'perpendicular to', 'far from', 'beneath', 'connected to', 'facing away from', 'parallel to', 'across from', 'at the edge of', 'part of', 'close to', 'alongside', 'at the back of', 'has as a part', 'consists of', 'opposite to', 'toward', 'adjacent to', 'into', 'in the middle of', 'beyond', 'within', 'around', 'outside', 'ahead of', 'against', 'across', 'out of', 'by', 'enclosed by', 'detached from', 'between', 'past', 'down from', 'with', 'along', 'at', 'down', 'among', 'congruent', 'through']

def main(data_root, data_type='zeroshot', num_choices=6):
    splits = ['train', 'dev', 'test']
    all_relations = []
    for split in splits:
        data_json = []
        json_path = os.path.join(data_root, 'splits', data_type, f'{split}.jsonl')
        with open(json_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                j_line = json.loads(line)
                data_json.append(j_line)
        data_json = [data for data in data_json if int(data['label']) != 0]
        print('parse the False data')
        for i, data in enumerate(data_json):
            if int(data['label']) == 0:
                raise ValueError('label is 0')

            
            question = data['caption']
        
            gt_answer = data['relation']
            others = [' is ', gt_answer, '.']
            
            for word in others:
                question = question.replace(word, '')
            object = question.split('the')[-1]
            object = object.strip()
            subject = question.split('the')[0]
            subject = subject.split('The')[-1]
            subject = subject.strip()
            data['subj'] = subject
            data['obj'] = object
            if gt_answer not in all_relations:
                all_relations.append(gt_answer)
            temp_option_list = copy.deepcopy(option_list)
            temp_option_list.remove(gt_answer)
            options = np.random.choice(temp_option_list, num_choices-1, replace=False)
            options = np.append(options, gt_answer)
            np.random.shuffle(options)
            data['options'] = options.tolist()
   
        # save to file
        with open(os.path.join(data_root, 'splits', data_type, f'{split}_options.jsonl'), 'w') as f:
            for item in data_json:
                f.write("%s\n" % json.dumps(item))


if __name__ == '__main__':
    data_root = 'dataspace/visual-spatial-reasoning/'
    main(data_root)