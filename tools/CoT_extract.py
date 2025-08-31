import os, sys
sys.path.append(os.getcwd())
import argparse
import asyncio
from omegaconf import OmegaConf
from module.kg_construct.CoT_construct_for_draw import  Self_CoT_extract #TODO 有改进
from data import  ScienceQA, MathV, VSR_MC, VizWiz, MMMU
from misc import create_logging, create_output_folders
from openai import OpenAI
from module.storage import BaseKVStorage, JsonKVStorage, StorageNameSpace
from functools import partial
from typing import Type, cast
from module.model.llm import openai_complete_if_cache, limit_async_func_call_with_multi_node, vqa_model_func_with_multi_node

async def main(
        PROJECT_NAME: str,
        INFO_EXTRACT_TYPE: str,
        LOG_DIR: str,
        DATASET_NAME:str,
        DATASET_PATH: dict,
        ENABLE_LLM_CACHE: bool = False,
        LLM_MODEL_MAX_ASYNC: int = 10,
        CoT_LOOP_TIMES: int = 10,
        Reward_LOOP_TIMES: int = 3,
        api_extra_body: dict = {},
        **kwargs    
        ):

    if DATASET_NAME == 'ScienceQA':
        dataset = ScienceQA(data_root=DATASET_PATH['DATA_ROOT'], split='trainval')
        api_extra_body.update({'skip_special_tokens': False})
    elif DATASET_NAME == 'MathV':
        dataset = MathV(data_root=DATASET_PATH['DATA_ROOT'], split='test')
    elif DATASET_NAME == 'VSR_MC_zeroshot' or DATASET_NAME == 'VSR_MC_random':
        dataset = VSR_MC(data_root=DATASET_PATH['DATA_ROOT'], split='trainval', data_type=DATASET_NAME.split('_')[-1])
    elif DATASET_NAME == 'VizWiz':
        dataset = VizWiz(data_root=DATASET_PATH['DATA_ROOT'], split='train')
    elif DATASET_NAME == 'MMMU':
        dataset = MMMU(data_root=DATASET_PATH['DATA_ROOT'], split='validation')
    # Load VQA model
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    llm_response_cache = (
            key_string_value_json_storage_cls(
                namespace="llm_response_cache", save_dir=os.path.join(LOG_DIR, 'outputs')
            )
            if ENABLE_LLM_CACHE
            else None
        )
    vqa_model = limit_async_func_call_with_multi_node(LLM_MODEL_MAX_ASYNC, max_node=len(os.environ.get("BASE_URL").split(';')))(
            partial(vqa_model_func_with_multi_node, hashing_kv=llm_response_cache)
        )


    if INFO_EXTRACT_TYPE == 'Self_CoT':
        await Self_CoT_extract(
        vqa_model=vqa_model,   
        dataset=dataset, 
        dataset_name=DATASET_NAME,
        path_to_save_preds = os.path.join(LOG_DIR, 'outputs'),
        CoT_loop_times=CoT_LOOP_TIMES,
        Reward_loop_times=Reward_LOOP_TIMES,
        api_extra_body=api_extra_body
        )
    else: raise NotImplementedError




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tool for extracting CLIP image features.')
    parser.add_argument('--config', dest='config', help='path to config file', type=str, required=True)
    args = parser.parse_args()
    args_dict = OmegaConf.load(args.config)
    for key in args_dict.keys():
        if key.endswith('_CONFIG'):
            temp_dict = OmegaConf.load(args_dict[key]) 
            args_dict = OmegaConf.merge(args_dict, temp_dict)
    LOG_DIR = create_output_folders(project_name=args_dict.PROJECT_NAME, output_dir=args_dict.LOG_DIR_ORI, config=args_dict, with_time=args_dict.WITH_TIME)
    asyncio.run(main(LOG_DIR=LOG_DIR, **args_dict))
    
