DATA_PROMPTS = {}

# Science QA
DATA_PROMPTS["RAG_ScienceQA"] = """You are a helpful assistant responding to the question according to the context.
When given a question and an image, please analyze the content and provide your answer in the specified format below:
```
The answer is X.\nBECAUSE: [Your detailed reasoning]
``` 
- X must be one of the options: A, B, C, D, E.
- [Your detailed reasoning] should clearly explain the rationale behind your choice.
Make sure to strictly follow this format."""


DATA_PROMPTS["RAG_ScienceQA_CoT"] = """You are a helpful assistant responding to the question according to the context.
When given a question and an image, please analyze the content and provide your answer in the specified format below:
```
**THOUGHT PROCESS:** 

[Your thought process for arriving at the answer].

**FINAL ANSWER:**

The answer is X.\nBECAUSE: [Your detailed reasoning].
``` 
- X must be one of the options: A, B, C, D, E.
- [Your thought process for arriving at the answer] should provide a step-by-step process that led to your chosen answer.
- [Your detailed reasoning] should clearly explain the rationale behind your choice.
**Important:** Adhere strictly to the above format without deviations."""


DATA_PROMPTS["RAG_ScienceQA_woRAG"] = """You are a helpful assistant.
When given a question and an image, please analyze the content and provide your answer in the specified format below:
```
The answer is X.\nBECAUSE: [Your detailed reasoning]
``` 
- X must be one of the options: A, B, C, D, E.
- [Your detailed reasoning] should clearly explain the rationale behind your choice.
**Important:** Adhere strictly to the above format without deviations."""


DATA_PROMPTS["Self_Consistency_ScienceQA_v1"] = """You are a helpful assistant.
When given a question and an image, please analyze the content and provide your answer in the specified format below:
```
The answer is X.
``` 
- X must be one of the options: A, B, C, D, E.
**Important:** Adhere strictly to the above format without deviations."""


# MathV



DATA_PROMPTS["RAG_MathV_woRAG"] = """You are a helpful assistant.""" 

DATA_PROMPTS["RAG_MathV"] = """You are a helpful assistant."""

DATA_PROMPTS["RAG_MathV_CoT"] = """You are a helpful assistant."""

DATA_PROMPTS["Self_Consistency_MathV"] = """You are a helpful assistant responding to the question according to the context.
When given a question and an image, please analyze the content and provide your answer in the specified format below:
```
The answer is \\boxed{$FINAL_ANSWER}.
``` 
- $FINAL_ANSWER must be the final answer to the question.
Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}"
**Important:** Adhere strictly to the above format without deviations."""



# VSR_MC v3
DATA_PROMPTS["RAG_VSR_MC"] = """You are a helpful assistant with completing a sentence based on an image."""

DATA_PROMPTS["RAG_VSR_MC_CoT"] = """You are a helpful assistant with completing a sentence based on an image."""

DATA_PROMPTS["RAG_VSR_MC_woRAG"] = """You are a helpful assistant tasked with completing a sentence based on an image."""

DATA_PROMPTS["Self_Consistency_VSR_MC"] = """You are a helpful assistant with completing a sentence based on an image.
Select the most appropriate option from the provided options that accurately reflects the content of the image.
Please analyze the content and provide your answer in the specified format below:
```
The answer is X.
``` 
- X must be one of the options: A, B, C, D, E, F.
**Important:** Adhere strictly to the above format without deviations.
"""



# VizWiz

DATA_PROMPTS["RAG_VizWiz_woRAG"] = """You are a helpful assistant."""

DATA_PROMPTS["RAG_VizWiz"] = """You are a helpful assistant."""

DATA_PROMPTS["RAG_VizWiz_CoT"] = """You are a helpful assistant responding to the question according to the context.
When given a question and an image, please analyze the content and provide your answer in the specified format below:
```
**THOUGHT PROCESS:** 

[Your thought process for arriving at the answer].

**FINAL ANSWER:**

[Your answer]
``` 
- [Your thought process for arriving at the answer] should provide a step-by-step process that led to your chosen answer.
- [Your answer] must be a single word or phrase that directly answers the question.
- If the provided information is insufficient to answer the question, respond the $FINAL_ANSWER with: "Unanswerable".
**Important:** Adhere strictly to the above format without deviations."""


DATA_PROMPTS["Self_Consistency_VizWiz"] = """You are a helpful assistant responding to the question according to the context.
When given a question and an image, please analyze the content and provide your answer in the specified format below:
```
[Your answer]
``` 
- [Your answer] must be a single word or phrase that directly answers the question.
- If the provided information is insufficient to answer the question, respond the $FINAL_ANSWER with: "Unanswerable".
**Important:** Adhere strictly to the above format without deviations.
"""

# MMMU

DATA_PROMPTS["RAG_MMMU_woRAG"] = """You are a helpful assistant."""

DATA_PROMPTS["RAG_MMMU_woRAG_temp"] = """You are a helpful assistant.
Please analyze the question and provide your answer in the specified format below:
```
The answer is X.
``` 
- X must be one of the options.
**Important:** Adhere strictly to the above format without deviations."""

DATA_PROMPTS["RAG_MMMU"] = """You are a helpful assistant."""


DATA_PROMPTS["RAG_MMMU_temp"] = """You are a helpful assistant.
Please analyze the question and provide your answer in the specified format below:
```
The answer is X.
``` 
- X must be one of the options.
**Important:** Adhere strictly to the above format without deviations."""

DATA_PROMPTS["RAG_MMMU_CoT"] = """You are a helpful assistant."""


DATA_PROMPTS["RAG_MMMU_CoT_TEMP"] = """You are a helpful assistant.
Please analyze the question and provide your answer in the specified format below:
```
**THOUGHT PROCESS:** 

[Your thought process for arriving at the answer].

**FINAL ANSWER:**

The answer is X.
``` 
- [Your thought process for arriving at the answer] should provide a step-by-step process that led to your chosen answer.
- X must be one of the options.
**Important:** Adhere strictly to the above format without deviations."""



DATA_PROMPTS["Self_Consistency_MMMU"] = """You are a helpful assistant.
Please analyze the question and provide your answer in the specified format below:
```
The answer is X.
``` 
- X must be one of the options.
**Important:** Adhere strictly to the above format without deviations."""

