
from __future__ import annotations

"""

Implements the MCTS + Self-Refine algorithm from
`Accessing GPT-4 level Mathematical Olympiad Solutions via Monte
Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report`
by Zhang et. al.

The authors' [repo](https://github.com/trotsky1997/MathBlackBox) uses critiques,
refinements, and parent nodes' answers as conversation history.
I haven't tried it yet.

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
import numpy as np
import asyncio
from misc import encode_image
ROOT_UCT_SCORE = 10_000

@dataclass
class MCTSNode:
    answer: str
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    Q: float = 0
    reward_samples: List[int] = field(default_factory=list)

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.answer}, Q={self.Q:.2f}, visits={self.visits})"

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)
        # Average worst-case and average outcomes
        self.Q = (min_reward + avg_reward) / 2


class SelectionPolicy(Enum):
    GREEDY = 1
    IMPORTANCE_SAMPLING = 2
    PAIRWISE_IMPORTANCE_SAMPLING = 3


class InitializeStrategy(Enum):
    ZERO_SHOT = 1
    DUMMY_ANSWER = 2

@dataclass
class MCTSr:
    problem: str
    max_rollouts: int
    exploration_constant: float = 1.0
    max_children: int = 2
    epsilon: float = 1e-10
    reward_limit: int = 95
    excess_reward_penalty: int = 5
    selection_policy: SelectionPolicy = SelectionPolicy.IMPORTANCE_SAMPLING
    initialize_strategy: InitializeStrategy = InitializeStrategy.ZERO_SHOT

    root: MCTSNode = MCTSNode(answer="I don't know.")

    # Logs
    critiques: List[str] = field(default_factory=list)
    refinements: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    selected_nodes: List[MCTSNode] = field(default_factory=list)

    def self_refine(self, node: MCTSNode) -> MCTSNode:
        raise NotImplementedError()

    def _evaluate_answer(self, node: MCTSNode) -> int:
        raise NotImplementedError()

    async def self_evaluate(self, node: MCTSNode):
        """Evaluate the quality of the answer. Sample `num_samples` times and average the results."""
        reward = await self._evaluate_answer(node)

        if reward > self.reward_limit:
            reward -= self.excess_reward_penalty

        node.add_reward(reward)

    def backpropagate(self, node: MCTSNode):
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = (parent.Q + best_child_Q) / 2
            parent.visits += 1
            parent = parent.parent

    def uct(self, node: MCTSNode):
        if not node.parent:
            # Using an arbitrarily high UCT score for the root node.
            # helps to prioritize breadth.
            return ROOT_UCT_SCORE

        return node.Q + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
        )

    def is_fully_expanded(self, node: MCTSNode):
        return len(node.children) >= self.max_children or any(
            child.Q > node.Q for child in node.children
        )

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
            if not self.is_fully_expanded(current_node):
                candidates.append(current_node)
            to_consider.extend(current_node.children)

        if not candidates:
            return self.root

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

    def zero_shot(self) -> str:
        """Generate a zero-shot answer."""
        raise NotImplementedError()

    async def initialize(self):
        """Generate a zero-shot answer."""
        if self.initialize_strategy == InitializeStrategy.ZERO_SHOT:
            self.root =  MCTSNode(answer= await self.zero_shot()) # 调用MCTSNode的init类
        elif self.initialize_strategy == InitializeStrategy.DUMMY_ANSWER:
            self.root = MCTSNode(answer="I don't know.")
        else:
            raise ValueError(f"Invalid initialize strategy: {self.initialize_strategy}")

    async def run(self):
        
        await self.initialize()
        for _ in tqdm.tqdm(range(self.max_rollouts)):
            node = self.select_node()
            await self.self_evaluate(node)
            child = await self.self_refine(node)
            node.add_child(child)
            await self.self_evaluate(child)
            self.backpropagate(child)
        self.print()
        return self.get_best_answer()

    def get_best_answer(self):
        from collections import deque
        to_visit = deque([self.root])
        best_node = self.root

        while to_visit:
            current_node = to_visit.popleft()
            if current_node.Q > best_node.Q:
                best_node = current_node
            to_visit.extend(current_node.children)

        return best_node.answer

    def print(self):
        print_tree(self.root)



class MCTSrReranking(MCTSr):
    def __init__(self, problem: str, max_rollouts: int = 5):
        super().__init__(problem, max_rollouts)
        self.vqa_model = limit_async_func_call(1)(
            partial(vqa_model_func, hashing_kv=None)
        )
        # pred_answer = await vqa_model(prompt=messages, extra_body=api_extra_body)
    async def zero_shot(self, 
                        sample_image_path: str = None,
                        sample_question: str = None,
                        system_prompt: str = "The user will provide a problem. Solve the problem. Think step by step.",
                        ):
        
        messages=[
            {
                "role": "system",
                "content": f"{system_prompt}",
            },
            ]
        if sample_image_path is not None:
            sample_image, image_dtype = encode_image(sample_image_path)
            messages.append({'role': 'user', 'content':  [
                {
                "type": "image_url",
                "image_url": {"url": f"data:{image_dtype};base64,{sample_image}"},
                },
                {"type": "text", "text": f"{sample_question}"}]
            })
        else:
            messages.append({'role': 'user', 'content':  f"{sample_question}"})

        zero_shot_answer = await self.vqa_model(prompt=messages, extra_body={})
        return zero_shot_answer

    async def self_refine(self, node: MCTSNode) -> MCTSNode:
        
        messages=[
            {
                "role": "system",
                "content": (
                    "Provide a detailed and constructive critique to improve the answer. "
                    "Highlight specific areas that need refinement or correction."
                    )
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<current_answer>\n{node.answer}\n</current_answer>",
                    ]
                ),
            },
        ]
           
        critique = await self.vqa_model(prompt=messages, extra_body={})
        assert critique is not None
        self.critiques.append(critique)

  
        messages=[
            {
                "role": "system",
                "content": """# Instruction
Refine the answer based on the critique. Your refined answer should be a direct and concise solution to the problem.

## Additional guidelines
- Your response should not refer to or discuss the criticisms.
- Do not repeat the problem statement.
- Respond with only the answer.
""",
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<current_answer>\n{node.answer}\n</current_answer>",
                        f"<critique>\n{critique}\n</critique>",
                    ]
                ),
            },
        ]
         
        refined_answer = await self.vqa_model(prompt=messages, extra_body={})
        assert refined_answer is not None
        self.refinements.append(refined_answer)

        return MCTSNode(answer=refined_answer, parent=node)

    async def _evaluate_answer(self, node: MCTSNode) -> int:
        messages = [
            {
                "role": "system",
                "content": ("Provide a reward score between -100 and 100 for the answer quality, using very strict standards. "
                        "Do not give a full score above 95. Make sure the reward score is an integer. "
                        "Return *ONLY* the score.")
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<answer>\n{node.answer}\n</answer>",
                    ]
                ),
            },
        ]
        for attempt in range(3):
            try:
                response = await self.vqa_model(prompt=messages, extra_body={})
                assert response is not None
                return int(response) # 这个int在这里会有一次判定
            except ValueError:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": response,
                        },
                        {
                            "role": "user",
                            "content": "Failed to parse reward as an integer.",
                        },
                    ]
                )
                if attempt == 2:
                    raise

def print_tree(node: MCTSNode | None, level: int = 0):
    if node is None:
        return
    indent = " " * level * 2
    node_str = repr(node)
    for line in node_str.split("\n"):
        print(indent + line)
    for child in node.children:
        print_tree(child, level + 1)



class MCTSrLlama38B(MCTSr):
    def __init__(self, problem: str, max_rollouts: int = 5):
        super().__init__(problem, max_rollouts)
        self.vqa_model = limit_async_func_call(1)(
            partial(vqa_model_func, hashing_kv=None)
        )
        # pred_answer = await vqa_model(prompt=messages, extra_body=api_extra_body)
    async def zero_shot(self) -> str:
        
        messages=[
            {
                "role": "system",
                "content": "The user will provide a problem. Solve the problem. Think step by step.",
            },
            {
                "role": "user",
                "content": f"{self.problem}",
            }
            ]


        answer = await self.vqa_model(prompt=messages, extra_body={})
        return answer

    async def self_refine(self, node: MCTSNode) -> MCTSNode:
        
        messages=[
            {
                "role": "system",
                "content": (
                    "Provide a detailed and constructive critique to improve the answer. "
                    "Highlight specific areas that need refinement or correction."
                    )
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<current_answer>\n{node.answer}\n</current_answer>",
                    ]
                ),
            },
        ]
           
        critique = await self.vqa_model(prompt=messages, extra_body={})
        assert critique is not None
        self.critiques.append(critique)

  
        messages=[
            {
                "role": "system",
                "content": """# Instruction
Refine the answer based on the critique. Your refined answer should be a direct and concise solution to the problem.

## Additional guidelines
- Your response should not refer to or discuss the criticisms.
- Do not repeat the problem statement.
- Respond with only the answer.
""",
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<current_answer>\n{node.answer}\n</current_answer>",
                        f"<critique>\n{critique}\n</critique>",
                    ]
                ),
            },
        ]
         
        refined_answer = await self.vqa_model(prompt=messages, extra_body={})
        assert refined_answer is not None
        self.refinements.append(refined_answer)

        return MCTSNode(answer=refined_answer, parent=node)

    async def _evaluate_answer(self, node: MCTSNode) -> int:
        messages = [
            {
                "role": "system",
                "content": ("Provide a reward score between -100 and 100 for the answer quality, using very strict standards. "
                        "Do not give a full score above 95. Make sure the reward score is an integer. "
                        "Return *ONLY* the score.")
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<answer>\n{node.answer}\n</answer>",
                    ]
                ),
            },
        ]
        for attempt in range(3):
            try:
                response = await self.vqa_model(prompt=messages, extra_body={})
                assert response is not None
                return int(response) # 这个int在这里会有一次判定
            except ValueError:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": response,
                        },
                        {
                            "role": "user",
                            "content": "Failed to parse reward as an integer.",
                        },
                    ]
                )
                if attempt == 2:
                    raise

def print_tree(node: MCTSNode | None, level: int = 0):
    if node is None:
        return
    indent = " " * level * 2
    node_str = repr(node)
    for line in node_str.split("\n"):
        print(indent + line)
    for child in node.children:
        print_tree(child, level + 1)
