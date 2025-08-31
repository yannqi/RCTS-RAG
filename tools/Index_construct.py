import os,sys
sys.path.append(os.getcwd())
import argparse
import asyncio
from data import ScienceQA, MathV, VSR_MC, VizWiz, MMMU
from misc import create_output_folders, always_get_an_event_loop
from module.storage import BaseKVStorage, FaissVectorDBStorage, JsonKVStorage, StorageNameSpace
from omegaconf import OmegaConf
from typing import Type, cast
import nest_asyncio
from misc.logger import logger

from module.RAG.RCTS_RAG import RCTS_RAG
nest_asyncio.apply()


async def main(
        LOG_DIR: str,
        DATASET_NAME:str,
        DATASET_PATH: dict,
        INDEX_TYPE: str,
        INDEX_MODEL: dict,
        NUM_GPUS: int,
        **kwargs    
        ):

    
    db_construct = RCTS_RAG(
        log_dir=LOG_DIR,
        index_type=INDEX_TYPE,
        index_model=INDEX_MODEL,
        num_gpus=NUM_GPUS
        )
   
    if DATASET_NAME == 'ScienceQA':
        dataset = ScienceQA(data_root=DATASET_PATH['DATA_ROOT'], split='trainval')
    elif DATASET_NAME == 'MathV':
        dataset = MathV(data_root=DATASET_PATH['DATA_ROOT'], split='test')
    elif DATASET_NAME == 'VSR_MC_zeroshot' or DATASET_NAME == 'VSR_MC_random':
        dataset = VSR_MC(data_root=DATASET_PATH['DATA_ROOT'], split='trainval', data_type=DATASET_NAME.split('_')[-1])
    elif DATASET_NAME == 'VizWiz':
        dataset = VizWiz(data_root=DATASET_PATH['DATA_ROOT'], split='train')
    elif DATASET_NAME == 'MMMU':
        dataset = MMMU(data_root=DATASET_PATH['DATA_ROOT'], split='validation') #TODO change
   
    VQA_doc_dict_list_wimage = []
    VQA_doc_dict_list = []
 
    for i, (sample_question_id, sample_image_path, sample_question, sample_answer) in enumerate(dataset):
        sample_question_id = DATASET_NAME + '_' + str(sample_question_id)
        # question_dict_list.append({'question': sample_question, 'question_id': sample_question_id})
        if isinstance(sample_image_path, list):
            sample_image_path = sample_image_path[0]
        if sample_image_path is not None:
            VQA_doc_dict_list_wimage.append({'question_id': sample_question_id, 'image_path': sample_image_path, 'question': sample_question, 'answer': sample_answer})
        VQA_doc_dict_list.append({'question_id': sample_question_id, 'image_path': sample_image_path, 'question': sample_question, 'answer': sample_answer})
                
    # TODO 从这里构造embedding
    new_VQA_doc_dict_list = await db_construct.docs_json_storage_construct(VQA_doc_dict_list)
    if 'faiss' in INDEX_TYPE:
        question_dict_list = []
        if new_VQA_doc_dict_list is not None:
            for k, v in new_VQA_doc_dict_list.items():
                question_dict_list.append({'question': v['question'], 'question_id': v['question_id']})
            await db_construct.question_vdb_construct(question_dict_list)
        # del db_construct.text_embedding_func
        # torch.cuda.empty_cache()
        # gc.collect()
        
    else:
        await db_construct._insert_done_json()

    if 'pre_flmr' in INDEX_TYPE:
        db_construct.hybrid_vdb_construct(VQA_doc_dict_list_wimage)
        

    
    logger.info("Finish constructing vectorDB")

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Tool for extracting vectorDB.')
    parser.add_argument('--config', dest='config', help='path to config file', type=str, required=True)
    parser.add_argument('--num_gpus', dest='num_gpus', help='number of gpus', type=int, default=1)
    args = parser.parse_args()
    args_dict = OmegaConf.load(args.config)
    for key in args_dict.keys():
        if key.endswith('_CONFIG'):
            temp_dict = OmegaConf.load(args_dict[key]) 
            args_dict = OmegaConf.merge(args_dict, temp_dict)
    LOG_DIR = create_output_folders(project_name=args_dict.PROJECT_NAME, output_dir=args_dict.LOG_DIR_ORI, config=args_dict, with_time=args_dict.WITH_TIME)
    asyncio.run(main(LOG_DIR=LOG_DIR, NUM_GPUS=args.num_gpus, **args_dict))
    sys.exit(0)
    
