import os,sys
sys.path.append(os.getcwd())
import asyncio
import json
from typing import Type, cast
from dataclasses import asdict, field, dataclass
import torch
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from module.model.embedding_model import EmbeddingFunc
from module.model.llm import limit_async_func_call
from module.storage import BaseKVStorage, FaissVectorDBStorage, JsonKVStorage, StorageNameSpace
from misc import create_output_folders, always_get_an_event_loop, compute_mdhash_id
from misc.logger import logger
from module.model.embedding_model import hf_embedding
from module.model.query import naive_text_query, random_query, hybrid_query

from flmr import FLMRModelForIndexing, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval
from flmr import index_custom_collection
from flmr import create_searcher, search_custom_collection

@dataclass
class RCTS_RAG:           
    log_dir: str
    index_type: str
    index_model: dict
    num_gpus: int = 1
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    vector_db_storage_cls: Type[BaseKVStorage] = FaissVectorDBStorage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
            
    def __post_init__(self):
        #TODO 目前我们仅支持文本的向量数据库构造 + PreFLMR

        self.vector_db_save_dir = os.path.join(self.log_dir, 'vector_db')
        os.makedirs(self.vector_db_save_dir, exist_ok=True)
        self.index_type = self.index_type.lower()
        # Load DB
        self.RAG_VQA_docs = self.key_string_value_json_storage_cls(
                namespace='RAG_VQA_docs',
                save_dir=self.vector_db_save_dir,
                ) 

        if 'faiss' in self.index_type:
            self.text_embedding_func = EmbeddingFunc(
                    embedding_dim=384,
                    max_token_size=5000,
                    func=lambda texts: hf_embedding(
                        texts,
                        tokenizer=AutoTokenizer.from_pretrained(
                                self.index_model['EMBEDDING_MODEL']['MODEL_PATH'], trust_remote_code=True
                        ),
                        embed_model=AutoModel.from_pretrained(
                            self.index_model['EMBEDDING_MODEL']['MODEL_PATH'], device_map="cuda",trust_remote_code=True
                        ),
                    )
                )

            self.text_embedding_func = limit_async_func_call(self.embedding_func_max_async)(
                self.text_embedding_func
            )

            
            self.question_vdb = self.vector_db_storage_cls(
                namespace='questions',
                save_dir=self.vector_db_save_dir,
                embedding_func=self.text_embedding_func,
                embedding_batch_num=self.embedding_batch_num,
                meta_fields={'question_id'},
            )
        if 'pre_flmr' in self.index_type:

            query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(self.index_model['COLBELT_MODEL']['MODEL_PATH'], subfolder="query_tokenizer")
            context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
                self.index_model['COLBELT_MODEL']['MODEL_PATH'], subfolder="context_tokenizer"
            )
            self.pre_flmr_model = FLMRModelForRetrieval.from_pretrained(
                self.index_model['COLBELT_MODEL']['MODEL_PATH'],
                query_tokenizer=query_tokenizer,
                context_tokenizer=context_tokenizer,
            )
            self.image_processor = AutoImageProcessor.from_pretrained(self.index_model['COLBELT_MODEL']['VIT_PATH'], trust_remote_code=True)
            self.searcher = None
   
    async def docs_json_storage_construct(self, VQA_dict_list):
        
        
        new_docs = {
                compute_mdhash_id(VQA_dict['question_id'], prefix="Qid_"): VQA_dict
                for VQA_dict in VQA_dict_list
            }
        
        _add_doc_keys = await self.RAG_VQA_docs.filter_keys(list(new_docs.keys()))  # Filter out the keys that already exist
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}

        if not len(new_docs):
            logger.warning("All docs are already in the storage")
            return None
        logger.info(f"[New Docs] inserting {len(new_docs)} docs")
        

        await self.RAG_VQA_docs.upsert(new_docs)
        return new_docs
        
    async def question_vdb_construct(self, question_dict_list):
        
        data_for_vdb = {
            compute_mdhash_id(q['question_id'], prefix='Qid_'): {
                'content': q['question'],
                'question_id': q['question_id'],
            } for q in question_dict_list
        }
        try:
            await self.question_vdb.upsert(data_for_vdb)
        except:
            raise NotImplementedError
        finally:
            await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [self.question_vdb, self.RAG_VQA_docs]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)


    def hybrid_vdb_construct(self, question_dict_list):
        # Use Pre-FLMR for hybrid search
        # TODO 目前没有引入增量检索
        question_dict_list = question_dict_list
        mapping_index2question_id = {}
        custom_collection = []
        for index, question_dict in enumerate(question_dict_list):
            question_id = question_dict['question_id']
            mapping_index2question_id[index] = compute_mdhash_id(question_id, prefix="Qid_")
            image_path = question_dict['image_path']
            custom_collection.append((question_dict['question'], None, image_path))
           
        index_custom_collection(
        custom_collection=custom_collection,  
        model=self.pre_flmr_model,
        index_root_path=self.vector_db_save_dir, 
        index_experiment_name="test_experiment",
        index_name="test_index",
        nbits=8, # number of bits in compression
        doc_maxlen=512, # maximum allowed document length
        overwrite=False, # whether to overwrite existing indices
        use_gpu=True, # whether to enable GPU indexing
        indexing_batch_size=64,
        model_temp_folder="tmp",
        nranks=self.num_gpus, # number of GPUs used in indexing
        )
        # Save mapping index2question_id
        mapping_dir = os.path.join(self.vector_db_save_dir, 'test_experiment')
        with open(os.path.join(mapping_dir, 'mapping_index2question_id.json'), 'w') as f:
            json.dump(mapping_index2question_id, f)
    async def _insert_done(self):
        tasks = []
        for storage_inst in [self.question_vdb, self.RAG_VQA_docs]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
        
    async def _insert_done_json(self):
        tasks = []
        for storage_inst in [self.RAG_VQA_docs]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def query(
        self,
        query: str,
        query_mode: str,
        top_k = 1,
        query_embedding_dict: dict = None
    ): 
        
        return await self._query(query, query_mode, top_k, query_embedding_dict)
    
    async def _query(self, query, query_mode, top_k, query_embedding_dict):
        img_path = query_embedding_dict['image_path']
        if  query_mode == 'hybrid':
            if img_path is None:
                query_mode = 'hybrid_woimg'
            else:
                query_mode = 'hybrid_wimg'

        if query_mode == 'naive_text' or query_mode == 'hybrid_woimg':
            context_dict = await naive_text_query(
                query=query,
                question_vdb=self.question_vdb,
                VQA_docs_db=self.RAG_VQA_docs,
                top_k=top_k,
                query_embedding_dict=query_embedding_dict,
            )
        
        elif query_mode == 'random':
            context_dict = await random_query(
                # query=query,
                # question_vdb=self.question_vdb,
                VQA_docs_db=self.RAG_VQA_docs,
                top_k=top_k,
                # query_embedding_dict=query_embedding_dict,
            )
        elif query_mode == 'hybrid_wimg':
            mapping_index2question_id_json_path = os.path.join(self.vector_db_save_dir, 'test_experiment', 'mapping_index2question_id.json')
            with open(mapping_index2question_id_json_path, 'r') as f:
                mapping_index2question_id_json = json.load(f)
            self.searcher = create_searcher(
                index_root_path=self.vector_db_save_dir,
                index_experiment_name="test_experiment",
                index_name="test_index",
                nbits=8, # number of bits in compression
                use_gpu=True, # whether to enable GPU searching
            ) if self.searcher is None else self.searcher
            context_dict = await hybrid_query(
                query=query,
                img_path=img_path,
                mapping_index2question_id_json=mapping_index2question_id_json,
                searcher=self.searcher,
                image_processor=self.image_processor,
                pre_flmr_model=self.pre_flmr_model,
                top_k=top_k,
                query_embedding_dict=query_embedding_dict,
                VQA_docs_db=self.RAG_VQA_docs,
            )
  
        else:
            raise NotImplementedError(f"query_mode {query_mode} is not implemented")
    
        return context_dict

    @staticmethod 
    def get_query_embdding_dict(question_id, query_content, image_path, dataset_name, embedding_dict, embedding_path):
        # keys COLBELT_MODEL, EMBEDDING_MODEL
        key = 'EMBEDDING_MODEL' if image_path is None else 'COLBELT_MODEL'
        embedding_name = embedding_dict[key]['MODEL_NAME']
        embedding_path = os.path.join(embedding_path, dataset_name, embedding_name)
        file_embedding_path = os.path.join(embedding_path, str(question_id) + '.pt')
        os.makedirs(embedding_path, exist_ok=True)
        query_embedding_dict = {
            'dataset_name': dataset_name,
            'query_id': question_id,
            'query_content': query_content,
            'image_path': image_path,
            'embedding_path': file_embedding_path
        }
        return query_embedding_dict
