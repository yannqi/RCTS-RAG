import argparse
from omegaconf import OmegaConf
import os
import sys
sys.path.append(os.getcwd())
import json
import torch 
import faiss
from data.ok_vqa import OK_VQA 
from tqdm import tqdm
# https://zhuanlan.zhihu.com/p/530958094
def main(
        INDEX_TYPE:str,
        INDEX_SAVE_PATH:str,
        DATASET:str,
        DATASET_PATH: dict,
        EMBEDDING_MODEL: str,
        **kwargs,
):
  
    if DATASET == 'ok_vqa':
        train_dataset = OK_VQA(img_dir=DATASET_PATH['IMG_DIR'], annotation_path=DATASET_PATH['ANNOTATION_PATH'], split='train', feat_dir=DATASET_PATH['FEAT_DIR'])
        question_id, img_feat, text_feat, _, _ = train_dataset[0]
        img_dimension = img_feat.shape[0]
        text_dimension = text_feat.shape[0]
    index_2_question_id = []
    for index, (question_id, img_feat, text_feat, _, _) in enumerate(tqdm(train_dataset)):
        img_embedding = img_feat.unsqueeze(0)
        text_embedding = text_feat.unsqueeze(0)
        if index == 0:
            img_embedding_all = img_embedding
            text_embedding_all = text_embedding
        else:
            img_embedding_all = torch.cat((img_embedding_all, img_embedding), 0)
            text_embedding_all = torch.cat((text_embedding_all, text_embedding), 0)
        index_2_question_id.append(int(question_id))

    if INDEX_TYPE == "L2":
        cpu_img_index = faiss.IndexFlatL2(img_dimension)  
        cpu_text_index = faiss.IndexFlatL2(text_dimension)  
        # gpu_index = faiss.index_cpu_to_all_gpus(cpu_img_index)
    if INDEX_TYPE == "dot":
        cpu_img_index = faiss.IndexFlatIP(img_dimension) 
        cpu_text_index = faiss.IndexFlatIP(text_dimension)
        # gpu_index = faiss.index_cpu_to_all_gpus(cpu_img_index)
    if INDEX_TYPE == "cosine":
        # cosine = normalize & dot
        faiss.normalize_L2(img_embedding_all.numpy())
        faiss.normalize_L2(text_embedding_all.numpy())
        cpu_img_index = faiss.IndexFlatIP(img_dimension)  # 构建索引index
        cpu_text_index = faiss.IndexFlatIP(text_dimension)
        # gpu_index = faiss.index_cpu_to_all_gpus(cpu_img_index)
    assert cpu_img_index.is_trained
    cpu_img_index.add(img_embedding_all.numpy())  # 添加数据
    cpu_text_index.add(text_embedding_all.numpy())

    faiss.write_index(cpu_img_index, os.path.join(INDEX_SAVE_PATH, f'{EMBEDDING_MODEL}_embedding/{DATASET}/Faiss_{INDEX_TYPE}_image.index'))
    faiss.write_index(cpu_text_index, os.path.join(INDEX_SAVE_PATH, f'{EMBEDDING_MODEL}_embedding/{DATASET}/Faiss_{INDEX_TYPE}_text.index'))
    # save mapping id
    with open(os.path.join(INDEX_SAVE_PATH, f'{EMBEDDING_MODEL}_embedding/{DATASET}/Faiss_{INDEX_TYPE}_question_id_mapping.json'), 'w') as f:
        json.dump(index_2_question_id, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tool for extracting CLIP image features.')
    parser.add_argument('--config', dest='config', help='path to config file', type=str, required=True)
    args = parser.parse_args()
    args_dict = OmegaConf.load(args.config)
    for key in args_dict.keys():
        if key.endswith('_CONFIG'):
            temp_dict = OmegaConf.load(args_dict[key]) 
            args_dict = OmegaConf.merge(args_dict, temp_dict)

    main(**args_dict)
