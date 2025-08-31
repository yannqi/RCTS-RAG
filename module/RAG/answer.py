from module.mcts.mcts_reranking import MCTSr_Reranking
from .utils import make_interleave_content
async def answer_woMCTS(prompt_dict, dataset, CoT_df, vqa_model, api_extra_body, use_rag):
    sample_question = prompt_dict['question']
    sample_answer = prompt_dict['answer']
    sample_image_path = prompt_dict['image_path']
    sample_question_id = prompt_dict['question_id']
    context_dict_list = prompt_dict['context_dict_list']
    if use_rag and CoT_df is not None:
        use_CoT = True
        system_dtype = 'RAG_CoT'
    elif use_rag:
        use_CoT = False
        system_dtype = 'RAG'
    else:
        use_CoT = False
        system_dtype = 'woRAG'
    system_prompt = dataset.get_system_prompt(system_dtype=system_dtype)
    messages = [
                {
                    "role": "system", 'content': f'{system_prompt}',
                }
        ]
    if context_dict_list is not None:
        # Use RAG
        for context_dict in context_dict_list:
            context_image_path = context_dict['image_path']
            context_question = context_dict['question']
            context_answer = context_dict['answer']
            context_question_id = context_dict['question_id']
            context_question_id_ori = context_question_id.split('_')[-1]
            if hasattr(dataset, 'get_question_prompt'):
                context_question = dataset.get_question_prompt(context_question, use_CoT)
            if CoT_df is not None:
                # Use CoT
                CoT_df_context = CoT_df[CoT_df['question_id'].astype(str) == str(context_question_id_ori)]
                if len(CoT_df_context) != 0:
                    context_CoT = CoT_df_context['pred_CoT'].tolist()[0]
    
                    # context_answer = f'{context_CoT}\n**FINAL ANSWER:**:\n{context_answer}'
                    if dataset.name == 'MathV' or dataset.name == 'MMMU':
                        context_answer = f'{context_answer}\nBECAUSE: {context_CoT}'
                    elif dataset.name == 'ScienceQA':
                        # context_answer = f'**THOUGHT PROCESS:**\n\n{context_CoT}\n\n**FINAL ANSWER:**:\n\n{context_answer}'
                        context_answer = f'**FINAL ANSWER:**:\n\n{context_answer}\n\n**THOUGHT PROCESS:**\n\n{context_CoT}' #TODO yannqi For cot
                    # context_answer = f'{context_CoT}\nAnswer: {context_answer}'
            # context_answer = f'The answer is {context_answer}'
            context_content = make_interleave_content(context_question, context_image_path)
            messages.append({'role': 'user', 'content': context_content})
            messages.append({'role': 'assistant', 'content': f"{context_answer}"})
           
  
    if hasattr(dataset, 'get_question_prompt'):
        sample_question = dataset.get_question_prompt(sample_question, use_CoT)
    sample_content = make_interleave_content(sample_question, sample_image_path)    
    
    messages.append({'role': 'user', 'content': sample_content})

    pred_answer = await vqa_model(prompt=messages, extra_body=api_extra_body)
    return pred_answer



async def answer_wMCTS(prompt_dict, dataset, top_k, CoT_df, vqa_model, api_extra_body:dict, reward_vqa_model, reward_api_extra_body:dict, max_rollouts:int, reward_config_dict:dict):
    sample_question = prompt_dict['question']
    sample_image_path = prompt_dict['image_path']
    sample_question_id = prompt_dict['question_id']
    context_dict_list = prompt_dict['context_dict_list']
    reward_loop_times = int(reward_config_dict['reward_loop_times'])
    reward_api_extra_body.update({'n': reward_loop_times})
    MCTS_tree = MCTSr_Reranking(max_rollouts=max_rollouts, max_depth=top_k, vqa_model=vqa_model, api_extra_body=api_extra_body, reward_vqa_model=reward_vqa_model, reward_api_extra_body=reward_api_extra_body, dataset=dataset, reward_config_dict=reward_config_dict)
    pred_answer, reranking_context, ori_answer, api_count = await MCTS_tree.run(
        sample_image_path=sample_image_path,
        sample_question=sample_question, 
        sample_question_id=sample_question_id,
        context_dict_list=context_dict_list,
        CoT_df=CoT_df, 
        )
   
    return pred_answer, reranking_context, ori_answer, api_count