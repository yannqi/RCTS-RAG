from __future__ import annotations

"""

# 对话历史要不要加入？

"""
from dataclasses import dataclass, field
from typing import List, Optional
import random
import math
from collections import deque
from enum import Enum
from functools import partial
from module.model.llm import openai_complete_if_cache, limit_async_func_call, vqa_model_func
import tqdm
# from .prompt_configs import (
#     llama_3_8b_prompt_config,
#     gpt_4o_prompt_config,
#     RefineResponse,
# )
from module.RAG.utils import encode_image
import copy
import re
import numpy as np
import asyncio
from .mcts_prompt import MCTS_PROMPTS
from module.RAG.utils import make_interleave_content
ROOT_UCT_SCORE = 10_000

@dataclass
class MCTSNode:
    answer: str
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list) 
    visits: int = 0
    Q: float = 0
    level: int = 0
    reward_samples: List[int] = field(default_factory=list)
    retrieval_doc: dict = field(default_factory=dict) 
    retrieval_idx: int = 0
    
    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.answer}, Q={self.Q:.2f}, visits={self.visits}, level={self.level} ,retrieval_doc={self.retrieval_doc})"

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)
        # Average worst-case and average outcomes
        self.Q = (min_reward + avg_reward) / 2

    def add_context(self, retrieval_doc: dict):
        self.retrieval_doc = retrieval_doc

class SelectionPolicy(Enum):
    GREEDY = 1
    IMPORTANCE_SAMPLING = 2
    PAIRWISE_IMPORTANCE_SAMPLING = 3


class InitializeStrategy(Enum):
    ZERO_SHOT = 1
    DUMMY_ANSWER = 2

@dataclass
class MCTSr_Reranking:
    max_rollouts: int
    vqa_model: function
    api_extra_body: dict
    reward_vqa_model: function
    reward_api_extra_body: dict
    exploration_constant: float = 0.5 # Q 区间为 0-1 我们设置为0.5 瞅瞅
    max_children: int = 3 # 对应一个节点可以有几个子节点
    max_depth: int = 3 # 对应树的最大深度
    epsilon: float = 1e-10
    selection_policy: SelectionPolicy = SelectionPolicy.IMPORTANCE_SAMPLING
    initialize_strategy: InitializeStrategy = InitializeStrategy.ZERO_SHOT 
    root: MCTSNode = MCTSNode(answer="I don't know.")
    reward_config_dict: dict = field(default_factory=dict)
    use_reference_answer: bool = True
    dataset: function = None
 
    async def run(self, sample_image_path, 
                        sample_question, 
                        sample_question_id,
                        context_dict_list,
                        CoT_df):
        self.api_count = 0
        self.use_CoT = True if CoT_df is not None else False
        self.sample_question = sample_question
        self.sample_image_path = sample_image_path
        if hasattr(self.dataset, 'get_question_prompt'):
            zero_sample_question = self.dataset.get_question_prompt(sample_question, use_CoT=False)
            RAG_sample_question = self.dataset.get_question_prompt(sample_question, use_CoT=self.use_CoT)
        else:
            zero_sample_question = sample_question
            RAG_sample_question = sample_question
        self.sample_content = make_interleave_content(RAG_sample_question, sample_image_path)
        self.zero_sample_content = make_interleave_content(zero_sample_question, sample_image_path)

        self.zero_shot_prompt = self.dataset.get_system_prompt('woRAG')
        if self.use_CoT:
            self.rag_prompt = self.dataset.get_system_prompt('RAG_CoT')
            self.rag_prompt_refine = self.dataset.get_system_prompt('RAG_CoT')
        else:
            self.rag_prompt = self.dataset.get_system_prompt('RAG')
            self.rag_prompt_refine = self.dataset.get_system_prompt('RAG') 
        self.context_dict_list = context_dict_list
        # self.context_dict_action_reward_list = [0] * len(context_dict_list)
        # self.context_dict_action_visits_list = [0] * len(context_dict_list)
        
        self.CoT_df = CoT_df
        
        # Reward config
        self.reward_type = self.reward_config_dict.get('reward_type', 'mutual') # mutual or reverse or self
        if self.reward_type == 'reverse' or self.reward_type == 'mutual':
            reverse_num = self.reward_config_dict.get('reverse_num', 1)
            reverse_type = self.reward_config_dict.get('reverse_type', 'greedy')
            if reverse_type == 'greedy':
                self.eval_context_dict_list = self.context_dict_list[:reverse_num]
            elif reverse_type == 'random':
                self.eval_context_dict_list = random.sample(self.context_dict_list, reverse_num)     
            else:
                self.eval_context_dict_list = None    
                
        await self.initialize()
        history_answer = [self.root.answer]
        if self.use_reference_answer:  
            node = self.root
            reference_answer_dtype = ['greedy' , 'prob_sample'] # , 'random_sample'  
            temp_uneval_child_list = []
            for sample_type in reference_answer_dtype:
                node = self.root
                for _ in range(self.max_depth):
                    child = await self._RAG_refine(node, sample_type=sample_type) 
                    if sample_type != 'random_sample': 
                        node.add_child(child)
                    if child.level == self.max_depth:
                        history_answer.append(child.answer)
                        if sample_type != 'random_sample':
                            temp_uneval_child_list.append(child)
                        if sample_type == 'greedy':
                            greedy_answer = child.answer            
                    node = child

            if self.dataset.name == 'ScienceQA' or self.dataset.name == 'VSR_MC' or self.dataset.name == 'VizWiz' or self.dataset.name == 'MMMU':
                es_flag =  self.dataset.judge_early_stop_for_mcts(history_answer) 
            elif self.dataset.name == 'MathV':
                es_flag = await self.dataset.judge_early_stop_for_mcts(history_answer, sample_question_id, self.vqa_model, self.api_extra_body)
                self.api_count += len(history_answer)
            else:
                raise NotImplementedError(f"dataset: {self.dataset.name} is not supported.")
            if es_flag:
                print(f"Early stop triggered.")
                return greedy_answer, ['None'], self.save_tree(), self.api_count
            else:
                for child in temp_uneval_child_list:
                    await self.mutual_evaluate(child)
                    self.backpropagate(child)
         
         

        for _ in range(self.max_rollouts):
            node = self.select_node()
            for _ in range(self.max_depth):
                if node.level < self.max_depth:
                    # await self.mutual_evaluate(node) 
                    child = await self._RAG_refine(node, sample_type='prob_sample') 
                    node.add_child(child)
                    if child.level == self.max_depth:
                        await self.mutual_evaluate(child)
                        self.backpropagate(child)
                    node = child
                else:
                    break
        # self.print_tree(self.root)
        best_answer, best_path = self.get_best_answer_branch()
        return best_answer, best_path, self.save_tree(), self.api_count
    

    async def _RAG_refine(self, node: MCTSNode, sample_type: str) -> MCTSNode:
        
          
        exist_retrieval_doc_list = []
        if len(node.retrieval_doc) != 0:
            exist_retrieval_doc_list.append(node.retrieval_doc)
        parent = node.parent
        level = 1 
        while parent:
            level += 1
            if len(parent.retrieval_doc) != 0:
                exist_retrieval_doc_list.append(parent.retrieval_doc)
            parent = parent.parent
            
        current_context_dict_list = [x for x in self.context_dict_list if x not in exist_retrieval_doc_list]
     
        confidence_score_list = [x['confidence'] for x in current_context_dict_list]


        if self.sample_image_path is not None: 
            # confidence_score bigger is better
            pass
            
        elif self.sample_image_path is None:
            confidence_score_list = [- x for x in confidence_score_list]
            
        # Min-Max normal
        def _min_max_normalize(data, epsilon=1e-3):
            min_val = min(data)
            max_val = max(data)
            normalized_data = [(x - min_val) / ((max_val - min_val)+ epsilon) for x in data] 
            normalized_data = [x + epsilon for x in normalized_data]
            return normalized_data
        sample_score_list = _min_max_normalize(confidence_score_list)
        if sample_type == 'prob_sample':
            selected_retrieval_idx = random.choices(range(len(sample_score_list)), weights=sample_score_list, k=1)[0]
        elif sample_type == 'greedy':
            selected_retrieval_idx = sample_score_list.index(max(sample_score_list))
        elif sample_type == 'random_sample':
            selected_retrieval_idx = random.randint(0, len(sample_score_list)-1)
        selected_retrieval_context = current_context_dict_list[selected_retrieval_idx]

        if len(exist_retrieval_doc_list) != 0:
            exist_retrieval_doc_list.reverse()
        exist_retrieval_doc_list.append(selected_retrieval_context)
        if len(exist_retrieval_doc_list) > self.max_depth:            
            raise AttributeError("Max retrieval doc reached")
        
        elif len(exist_retrieval_doc_list) == self.max_depth:
            messages = [
                        {
                            "role": "system", 'content': f'{self.rag_prompt_refine}',
                        }
                ]
            for idx, context_dict in enumerate(exist_retrieval_doc_list):
                context_image_path = context_dict['image_path']
                context_question = copy.deepcopy(context_dict['question'])
                context_answer = copy.deepcopy(context_dict['answer'])
                context_question_id = context_dict['question_id']
                context_question_id_ori = context_question_id.split('_')[-1]
                
                if self.CoT_df is not None:
                    # Use CoT
                    CoT_df_context = self.CoT_df[self.CoT_df['question_id'].astype(str) == str(context_question_id_ori)]
                    context_CoT = CoT_df_context['pred_CoT'].tolist()[0]
                    if self.dataset.name == 'ScienceQA':
                        context_answer = f'**THOUGHT PROCESS:**\n\n{context_CoT}\n\n**FINAL ANSWER:**:\n\n{context_answer}\n'
                    elif self.dataset.name == 'MMMU' or self.dataset.name == 'MathV':
                        context_answer = f'{context_answer}\nBECAUSE: {context_CoT}'    
                    
                if hasattr(self.dataset, 'get_question_prompt'):
                    context_question = self.dataset.get_question_prompt(context_question, use_CoT=self.use_CoT)
                context_content = make_interleave_content(context_question, context_image_path)
                messages.append({'role': 'user', 'content':  context_content})
                messages.append({'role': 'assistant', 'content': f"{context_answer}"})
            
            messages.append({'role': 'user', 'content':  self.sample_content})
          
            
            refined_answer = await self.vqa_model(prompt=messages, extra_body=self.api_extra_body)
            self.api_count += 1
        else: 
            refined_answer = 'This is not the end of the node'
        return MCTSNode(answer=refined_answer, parent=node, retrieval_doc=selected_retrieval_context, retrieval_idx=selected_retrieval_idx, level=level)




    async def _self_evaluate_answer(self, node: MCTSNode) -> int:
        # Self-consistency evaluation. TODO only support wCoT
        consistency_prompt = self.dataset.get_system_prompt('Self_Consistency')
        pred_answer = copy.deepcopy(node.answer)
        if self.dataset.name == 'ScienceQA':
            pred_cot_answer_list = re.split(r'\*\*final answer:\*\*|\*\*FINAL ANSWER:\*\*|FINAL ANSWER:|Final Answer\:', pred_answer)
            if len(pred_cot_answer_list) == 1:
                return 0
            pred_answer_as_gt = pred_cot_answer_list[-1]
            pred_answer_as_gt = pred_answer_as_gt.strip()
            pred_answer_as_gt = pred_answer_as_gt.strip('\n')
            pred_answer_as_gt = pred_answer_as_gt.strip('**')
            pred_CoT = pred_cot_answer_list[0]
            pred_CoT = pred_CoT.strip()
            pred_CoT = pred_CoT.strip('\n')
        elif  self.dataset.name == 'MMMU' or self.dataset.name == 'MathV':
            pred_cot_answer_list = re.split(r'BECAUSE:|because:|Because:', pred_answer)
            if len(pred_cot_answer_list) == 1:
                return 0
            pred_answer_as_gt = pred_cot_answer_list[0]
            pred_answer_as_gt = pred_answer_as_gt.strip()
            pred_answer_as_gt = pred_answer_as_gt.strip('\n')
            pred_answer_as_gt = pred_answer_as_gt.strip('**')
            pred_CoT = pred_cot_answer_list[-1]
            pred_CoT = pred_CoT.strip()
            pred_CoT = pred_CoT.strip('\n')
            
        
        reward = []
        messages = [
                {
                    "role": "system", 'content': f'{consistency_prompt}',
                }
        ]
        if hasattr(self.dataset, 'get_question_prompt'):
            self_eval_question = self.dataset.get_question_prompt(self.sample_question, use_CoT=False)
        else:
            self_eval_question = self.sample_question
        self_eval_sample_content = make_interleave_content(f"{self_eval_question}\n\n**THOUGHT PROCESS**{pred_CoT}", self.sample_image_path)
      
        messages.append({'role': 'user', 'content': self_eval_sample_content })
        
        pred_answer_list = await self.reward_vqa_model(prompt=messages, extra_body=self.reward_api_extra_body)
      
        self.api_count += 1
        pred_answer_list = [pred_answer_list] if isinstance(pred_answer_list, str) else pred_answer_list
        for pred_answer in pred_answer_list:
            if self.dataset.name == 'ScienceQA' or self.dataset.name == 'VSR_MC' or self.dataset.name == 'MMMU':
                reward_answer = self.dataset.judge_answer_for_mcts(pred_answer_as_gt, pred_answer)
            elif self.dataset.name == 'MathV' or self.dataset.name == 'VizWiz':
                reward_answer = self.dataset.judge_answer_for_mcts_self_eval(pred_answer_as_gt, pred_answer)
            reward.append(reward_answer)
        return sum(reward) / len(reward)
    
    
    async def _reverse_evaluate_answer(self, node: MCTSNode) -> int:
        #* Reverse evaluate
        exist_retrieval_doc_list = []
        exist_retrieval_doc_list.append(node.retrieval_doc)
        parent = node.parent
        while parent:
            if len(parent.retrieval_doc) != 0:
                exist_retrieval_doc_list.append(parent.retrieval_doc)
            parent = parent.parent       
        reward = []

        reverse_messages = [
                {
                    "role": "system", 'content': f'{self.rag_prompt}',
                }
        ]
        
    
        reverse_messages.append({'role': 'user', 'content':  self.sample_content})
        reverse_messages.append({'role': 'assistant', 'content': f"{node.answer}"})
        
        reverse_type = self.reward_config_dict.get('reverse_type', 'greedy')
        if reverse_type == 'retrieval':
            cur_eval_context_dict_list = exist_retrieval_doc_list
        else:
            cur_eval_context_dict_list = self.eval_context_dict_list
            
        for idx, exist_retrieval_doc in enumerate(cur_eval_context_dict_list):
            reverse_messages_context = copy.deepcopy(reverse_messages)
            context_image_path = exist_retrieval_doc['image_path']
            context_question = copy.deepcopy(exist_retrieval_doc['question'])
            context_answer = copy.deepcopy(exist_retrieval_doc['answer'])
            context_question_id = exist_retrieval_doc['question_id']
            
            if hasattr(self.dataset, 'get_question_prompt'):
                context_question = self.dataset.get_question_prompt(context_question, use_CoT=self.use_CoT)
            reverse_context_content = make_interleave_content(context_question, context_image_path)
            reverse_messages_context.append({'role': 'user', 'content':  reverse_context_content })
         
            pred_context_answer_list = await self.reward_vqa_model(prompt=reverse_messages_context, extra_body=self.reward_api_extra_body)
            self.api_count += 1
            pred_context_answer_list = [pred_context_answer_list] if isinstance(pred_context_answer_list, str) else pred_context_answer_list
          
            for pred_context_answer in pred_context_answer_list:
                if self.dataset.name == 'ScienceQA' or self.dataset.name == 'VSR_MC' or self.dataset.name == 'MMMU':
                    reward_answer = self.dataset.judge_answer_for_mcts(context_answer, pred_context_answer)
                elif self.dataset.name == 'MathV' or self.dataset.name == 'VizWiz':
                    reward_answer = self.dataset.judge_answer_for_mcts(pred_context_answer, context_question_id)
         
                reward.append(reward_answer)
        
        return sum(reward) / len(reward)
    
    
    async def _reverse_evaluate_answer_woCoT(self, node: MCTSNode) -> int:
        #* Reverse evaluate
        exist_retrieval_doc_list = []
        exist_retrieval_doc_list.append(node.retrieval_doc)
        parent = node.parent
        while parent:
            if len(parent.retrieval_doc) != 0:
                exist_retrieval_doc_list.append(parent.retrieval_doc)
            parent = parent.parent       
        reward = []
        system_prompt = self.dataset.get_system_prompt('RAG')
        reverse_messages = [
                {
                    "role": "system", 'content': f'{system_prompt}',
                }
        ]
        if self.dataset.name == 'ScienceQA':
            pred_answer = copy.deepcopy(node.answer)
            pred_answer = re.split(r'\*\*final answer:\*\*|\*\*FINAL ANSWER:\*\*|FINAL ANSWER:|Final Answer\:', pred_answer)[-1]
            pred_answer = pred_answer.strip()
            pred_answer = pred_answer.strip('\n')
            pred_answer = pred_answer.strip('**')
            pred_answer_woCoT = pred_answer
        elif self.dataset.name == 'MMMU' or self.dataset.name == 'MathV':
            pred_answer = copy.deepcopy(node.answer)
            pred_answer = re.split(r'BECAUSE:|because:|Because:', pred_answer)[0]
            pred_answer = pred_answer.strip()
            pred_answer_woCoT = pred_answer
        reverse_messages.append({'role': 'user', 'content':  self.sample_content})
        reverse_messages.append({'role': 'assistant', 'content': f"{pred_answer_woCoT}"})
     
        
        # for idx, exist_retrieval_doc in enumerate(exist_retrieval_doc_list): 
        for idx, exist_retrieval_doc in enumerate(self.eval_context_dict_list):
            reverse_messages_context = copy.deepcopy(reverse_messages)
            context_image_path = exist_retrieval_doc['image_path']
            context_question = copy.deepcopy(exist_retrieval_doc['question'])
            context_answer = copy.deepcopy(exist_retrieval_doc['answer'])
            context_question_id = exist_retrieval_doc['question_id']
            if hasattr(self.dataset, 'get_question_prompt'):
                context_question = self.dataset.get_question_prompt(context_question, use_CoT=self.use_CoT)
            reverse_context_content = make_interleave_content(context_question, context_image_path)
            reverse_messages_context.append({'role': 'user', 'content':  reverse_context_content })
         
            
            pred_context_answer_list = await self.reward_vqa_model(prompt=reverse_messages_context, extra_body=self.reward_api_extra_body)
            self.api_count += 1
            pred_context_answer_list = [pred_context_answer_list] if isinstance(pred_context_answer_list, str) else pred_context_answer_list
            for pred_context_answer in pred_context_answer_list:
                if self.dataset.name == 'ScienceQA' or self.dataset.name == 'VSR_MC':
                    reward_answer = self.dataset.judge_answer_for_mcts(context_answer, pred_context_answer)
                elif self.dataset.name == 'MathV' or self.dataset.name == 'VizWiz':
                    reward_answer = self.dataset.judge_answer_for_mcts(pred_context_answer, context_question_id)
                    # self.api_count += 1
                reward.append(reward_answer)
            # print(f"Len of reward: {len(reward)}, reward: {reward}")
        return sum(reward) / len(reward)
    
    async def mutual_evaluate(self, node: MCTSNode):
        """Evaluate the quality of the answer. Sample `num_samples` times and average the results."""
       
        if self.reward_type == 'self':
            self_reward = await self._self_evaluate_answer(node)
            reward = self_reward
        elif self.reward_type == 'reverse':
            reverse_reward = await self._reverse_evaluate_answer(node)  
            reward = reverse_reward
        elif self.reward_type == 'mutual':
            self_reward = await self._self_evaluate_answer(node)
            reverse_reward = await self._reverse_evaluate_answer(node)  
            reward = float(self.reward_config_dict['self_weight']) * self_reward + float(self.reward_config_dict['reverse_weight']) * reverse_reward
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
        retrieval_idx = node.retrieval_idx

        node.add_reward(reward)

    def backpropagate(self, node: MCTSNode):
        
        parent = node.parent
        current_node_Q = node.Q
        
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
     
            parent.Q = ((parent.Q * parent.visits + current_node_Q) / (parent.visits + 1)  + best_child_Q) / 2 
            parent.visits += 1
            retrieval_idx = parent.retrieval_idx

            parent = parent.parent

    def uct(self, node: MCTSNode):
        if not node.parent:
            # Using an arbitrarily high UCT score for the root node.
            # helps to prioritize breadth.
            return ROOT_UCT_SCORE
        return node.Q + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
        )

    def is_fully_expanded_old(self, node: MCTSNode):
        return len(node.children) >= self.max_children or any(
            child.Q > node.Q for child in node.children
        )
    def is_fully_expanded(self, node: MCTSNode):
        return (len(node.children) >= self.max_children) or (node.level >= self.max_depth)
    def select_node(self):
        """Select a non-fully expanded node with the highest UCT value.

        A node is fully expanded if either:
        1. It has reached the max number of children
        2. Any of its children have a Q value greater than its own
        """
        candidates: list[MCTSNode] = []
        to_consider = deque([self.root])  #  双端队列，可以在两端添加或删除元素。

        # judge the candidates
        while to_consider:
            current_node = to_consider.popleft() # 弹出队首元素
            if not self.is_fully_expanded(current_node) and current_node.level < self.max_depth:
                candidates.append(current_node)
            to_consider.extend(current_node.children)

        if not candidates:
            return self.root


        # TODO 这里可以扩展一个加权采样
        if self.selection_policy == SelectionPolicy.GREEDY:
            return max(candidates, key=self.uct)
        elif self.selection_policy == SelectionPolicy.IMPORTANCE_SAMPLING:
            # Sample, weighted by UCT score
            uct_scores = [self.uct(node) for node in candidates]
            selected_pair_idx = random.choices(
                range(len(candidates)), weights=uct_scores, k=1
            )[0]  
            return candidates[selected_pair_idx]
        elif self.selection_policy == SelectionPolicy.PAIRWISE_IMPORTANCE_SAMPLING:
            # Sample, weighted by the difference in UCT scores between pairs
            uct_scores = [self.uct(node) for node in candidates]
            pairs = [
                (i, j) for i in range(len(candidates)) for j in range(len(candidates))
            ]
            pair_weights = [
                max(uct_scores[i], uct_scores[j]) - min(uct_scores[i], uct_scores[j])
                for i, j in pairs
            ]
            selected_pair_idx = random.choices(
                range(len(pairs)), weights=pair_weights, k=1
            )[0]
            selected_candidate_idx = max(
                pairs[selected_pair_idx], key=lambda x: uct_scores[x]
            )
            return candidates[selected_candidate_idx]
        else:
            raise ValueError(f"Invalid selection policy: {self.selection_policy}")

    async def zero_shot(self):
        
        messages=[
            {
                "role": "system",
                "content": f"{self.zero_shot_prompt}"
            },
            ]

       
        messages.append({'role': 'user', 'content': self.zero_sample_content})

        zero_shot_answer = await self.vqa_model(prompt=messages, extra_body=self.api_extra_body)
        self.api_count += 1
        return zero_shot_answer

    async def initialize(self):
        """Generate a zero-shot answer."""
        if self.initialize_strategy == InitializeStrategy.ZERO_SHOT:
            self.root =  MCTSNode(answer= await self.zero_shot(), level=0) # 调用MCTSNode的init类
        elif self.initialize_strategy == InitializeStrategy.DUMMY_ANSWER:
            self.root = MCTSNode(answer="I don't know.")
        else:
            raise ValueError(f"Invalid initialize strategy: {self.initialize_strategy}")
        # Use the reference answer
        
    

    def get_best_answer_node(self):
        from collections import deque
        to_visit = deque([self.root])
        best_node = self.root

        while to_visit:
            current_node = to_visit.popleft()
            if current_node.Q > best_node.Q:
                best_node = current_node
            to_visit.extend(current_node.children)

        return best_node.answer



    def get_best_answer_branch(self):
        from collections import deque

        def evaluate_branch(node):
            """递归计算从当前节点到叶节点的分支总得分"""
            if not node.children:  # 如果是叶节点
                return node.Q, [node]
            max_score = float('-inf')
            best_path = []
            for child in node.children:
                child_score, child_path = evaluate_branch(child)
                if child_score > max_score:
                    max_score = child_score
                    best_path = child_path
            return node.Q + max_score, [node] + best_path

        # 从根节点开始评估所有分支
        best_score, best_path = evaluate_branch(self.root)
        # 返回最佳分支的最后一个叶节点的 answer
        best_reranking_examples = [node.retrieval_doc for node in best_path]
        return best_path[-1].answer, best_reranking_examples[1:]

    def print_tree(self, node: MCTSNode | None, level: int = 0 ):
        if node is None:
            return
        indent = " " * level * 2
        node_str = repr(node)
        for line in node_str.split("\n"):
            print(indent + line)
        for child in node.children:
            self.print_tree(child, level + 1)
    def save_tree(self):
        # Save the tree in dict format
        def node_to_dict(node: MCTSNode):
            return {
                "answer": node.answer,
                "Q": node.Q,
                "visits": node.visits,
                "level": node.level,
                "retrieval_doc": node.retrieval_doc,
                "retrieval_idx": node.retrieval_idx,
                "children": [node_to_dict(child) for child in node.children]
            }

        return node_to_dict(self.root)
    



    
