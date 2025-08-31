MCTS_PROMPTS = {}



MCTS_PROMPTS["MCTS_Self_Reward"] = """You are a helpful assistant.
Task: Evaluate the provided answer to the given question based on the criteria. Assign an appropriate score from 0 to 10.
Scoring Criteria:
10: The answer is perfect. It is completely correct and accurately addresses the user's question. Details are comprehensive, understanding is evident, with no omissions or errors.
9: The answer is excellent, nearly perfect with minimal room for improvement. Information is complete and logically clear, with just slight lack of detail.
8: The answer is very good. Most of the content is correct and accurate, adequately answers the question, but contains minor inaccuracies or could use additional information.
7: The answer is good, essentially correct but misses some important details or includes some imprecise elements, requiring further refinement.
6: The answer is adequate, with some correct content, but misses key information or has a certain misunderstanding, needing clear improvements.
5: The answer is barely passable, addressing some aspects of the question but generally lacking in accuracy and completeness, requiring significant improvement.
4: The answer is subpar, with most information incorrect or incomplete, partially off-topic, needing major adjustments and additional content.
3: The answer is poor, failing to correctly address the question, with seriously insufficient or frequently incorrect information, showing a basic misunderstanding.
2: The answer is very poor, barely touches on correctness, with content mostly unrelated to the user's question, requiring a reevaluation and proper response.
1: The answer is extremely poor, failing to answer the user's question, with all information irrelevant and unrelated to the question.
0: The answer is entirely off-topic, failing to provide any relevant information, with no connection to the question at all.
Instructions: Based on the criteria above, evaluate the provided answer to the question and assign a score. 
Make sure the reward score is an integer. Return *ONLY* the score.
"""
#  Clearly state the reasons for your scoring, citing specific areas where the answer excels or needs improvement.


MCTS_PROMPTS["MCTS_Reverse_Reward"] = """You are a helpful assistant.
Task: Evaluate the predicted answer to the given question and compare it with the reference answer using the criteria below. Assign an appropriate score from 0 to 10.
Scoring Criteria:
10: The answer is perfect. It is completely correct and accurately addresses the user's question. Details are comprehensive, understanding is evident, with no omissions or errors.
9: The answer is excellent, nearly perfect with minimal room for improvement. Information is complete and logically clear, with just slight lack of detail.
8: The answer is very good. Most of the content is correct and accurate, adequately answers the question, but contains minor inaccuracies or could use additional information.
7: The answer is good, essentially correct but misses some important details or includes some imprecise elements, requiring further refinement.
6: The answer is adequate, with some correct content, but misses key information or has a certain misunderstanding, needing clear improvements.
5: The answer is barely passable, addressing some aspects of the question but generally lacking in accuracy and completeness, requiring significant improvement.
4: The answer is subpar, with most information incorrect or incomplete, partially off-topic, needing major adjustments and additional content.
3: The answer is poor, failing to correctly address the question, with seriously insufficient or frequently incorrect information, showing a basic misunderstanding.
2: The answer is very poor, barely touches on correctness, with content mostly unrelated to the user's question, requiring a reevaluation and proper response.
1: The answer is extremely poor, failing to answer the user's question, with all information irrelevant and unrelated to the question.
0: The answer is entirely off-topic, failing to provide any relevant information, with no connection to the question at all.
Instructions: Based on the criteria above, evaluate the provided answer to the question and assign a score. 
Make sure the reward score is an integer. Return *ONLY* the score.
"""