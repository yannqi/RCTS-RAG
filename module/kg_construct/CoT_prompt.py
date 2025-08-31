
CoT_PROMPTS = {}


CoT_PROMPTS["refine_CoT_steps_system_VQA_system_prompt"] = """You are a helpful assistant tasked with providing a detailed and structured thought process to correct the user's answer. The thought process should be logically sound, step-by-step, and clearly lead to the correct answer."""



CoT_PROMPTS[
    "refine_CoT_steps_system_VQA_wimg_for_choice"
] = """**User Question:**

{question}

**User Thought Process:**

{thinking_process}

**User Answer:**

{pred_answer}

**Correct Answer:**

{gt_answer}

**System Prompt:**

To help improve the user's thought process and arrive at the correct answer, please revise the user's thinking step-by-step, ensuring that each step logically leads to the correct answer.

Your revised thought process should include the following steps:

1. **Understanding the Question:** Identify and articulate the key components and requirements of the question.

2. **Analyzing the Image:** Extract and describe the relevant information from the image that pertains to the question.

3. **Evaluating Options:** Consider each option in light of the question and image, providing rationale for each consideration.

4. **Eliminating Incorrect Choices:** Justify the elimination of incorrect options based on your reasoning.

5. **Selecting the Best Answer:** Choose the most appropriate answer and provide a detailed justification for your selection.

By following this structure, the revised thought process will be comprehensive, logical, and easy to follow."""


CoT_PROMPTS[
    "refine_CoT_steps_system_VQA_woimg_for_choice"
] = """**User Question:**

{question}

**User Thought Process:**

{thinking_process}

**User Answer:**

{pred_answer}

**Correct Answer:**

{gt_answer}

**System Prompt:**

To help improve the user's thought process and arrive at the correct answer, please revise the user's thinking step-by-step, ensuring that each step logically leads to the correct answer.

Your revised thought process should include the following steps:

1. **Understanding the Question:** Identify and articulate the key components and requirements of the question.

2. **Evaluating Options:** Consider each option in light of the question, providing rationale for each consideration.

3. **Eliminating Incorrect Choices:** Justify the elimination of incorrect options based on your reasoning.

4. **Selecting the Best Answer:** Choose the most appropriate answer and provide a detailed justification for your selection.

By following this structure, the revised thought process will be comprehensive, logical, and easy to follow."""



CoT_PROMPTS[
    "refine_CoT_steps_system_VQA"
] = """**User Question:**

{question}

**User Thought Process:**

{thinking_process}

**User Answer:**

{pred_answer}

**Correct Answer:**

{gt_answer}

**System Prompt:**

To help improve the user's thought process and arrive at the correct answer, please revise the user's thinking step-by-step, ensuring that each step logically leads to the correct answer.
"""




CoT_PROMPTS[
    "get_CoT_steps_system_VQA_woimg_for_choice"
] = """**User Question:**

{question}

**Answer:**

{answer}

**System Prompt:**

Please describe your thought process in a step-by-step, structured manner, ensuring that each step logically leads to the final answer.

Your thought process should include the following steps:

1. **Understanding the Question:** Identify and articulate the key components and requirements of the question.

2. **Evaluating Options:** Consider each option in light of the question, providing rationale for each consideration.

3. **Eliminating Incorrect Choices:** Justify the elimination of incorrect options based on your reasoning.

4. **Selecting the Best Answer:** Choose the most appropriate answer and provide a detailed justification for your selection.

By following this structure, your thought process will be comprehensive, logical, and easy to follow."""



# 通用版本
CoT_PROMPTS[
    "get_CoT_steps_system_VQA"
] = """**User Question:**

{question}

**Answer:**

{answer}

**System Prompt:**

Please describe your thought process in a step-by-step, structured manner, ensuring that each step logically leads to the final answer.
"""


CoT_PROMPTS[
    "get_CoT_VQA_v1_system_prompt"
] = """You are a helpful assistant tasked with providing a detailed and structured thought process based on the answer. The thought process should be logically sound, step-by-step, and clearly lead to the final answer."""


CoT_PROMPTS[
    "get_CoT_VQA_v1"
] = """**User Question:**

{question}

**Answer:**

{answer}

**System Prompt:**

Please describe your thought process in a step-by-step, structured manner, ensuring that each step logically leads to the final answer.
Let's think step by step.
"""

CoT_PROMPTS[
    "Answer_with_CoT_VQA_v1"
] = """**User Question:**

{question}

**THOUGHT PROCESS:**

{thinking_process}
"""


