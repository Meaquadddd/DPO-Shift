from email import message
from email.mime import base
import json
from weakref import ref
from datasets import load_dataset
import random
import os
from git import Tree
import ollama
from ollama import chat
from ollama import ChatResponse
import test
from tqdm import tqdm
import os
import torch
# os.system("ollama serve")

result_list = []

dataset_names=['capybara','ultrafeedback','alpacafarm']
model_names=['meta-llama','mistral','Qwen']
real_names=['Llama-3-8b','Zephyr','Qwen-7b']
strategy = ['fixed','increase_linear','decrease_linear']
test_dataset_choice=1
train_dataset_choice=1
model_choice=0
strtegy_choice=0
dpo_lambda = 0.75
use_short_list = False
sft_baseline = False
# for now

answer_home = '../FastChat/fastchat/llm_judge/data'
answer_path = os.path.join(answer_home,dataset_names[test_dataset_choice],'model_answer',model_names[model_choice])

if test_dataset_choice == 0:
    order_questions = load_dataset('json', data_files='../FastChat/fastchat/llm_judge/data/capybara/question.jsonl')['train']
    if use_short_list:
        random_indices = torch.randperm(len(order_questions))[:500]
    
elif test_dataset_choice == 1:
    order_questions = load_dataset('json', data_files='../FastChat/fastchat/llm_judge/data/ultrafeedback/question.jsonl')['train']
    if use_short_list:
        if not os.path.exists('../FastChat/fastchat/llm_judge/data/ultrafeedback/question_500.pt'):
            order_questions = load_dataset('json', data_files='../FastChat/fastchat/llm_judge/data/ultrafeedback/question.jsonl')['train']
            random_indices = torch.randperm(len(order_questions))[:500]
            random_indices = random_indices.numpy()
            order_questions = order_questions.select(random_indices)
            torch.save(order_questions,'../FastChat/fastchat/llm_judge/data/ultrafeedback/question_500.pt')
        else:
            order_questions = torch.load('../FastChat/fastchat/llm_judge/data/ultrafeedback/question_500.pt')
elif test_dataset_choice == 2:
    if not os.path.exists('../FastChat/fastchat/llm_judge/data/alpacafarm/question_500.pt'):
        order_questions = load_dataset('json', data_files='../FastChat/fastchat/llm_judge/data/alpacafarm/question.jsonl')['train']
        random_indices = torch.randperm(len(order_questions))[:500]
        random_indices = random_indices.numpy()
        order_questions = order_questions.select(random_indices)
        torch.save(order_questions,'../FastChat/fastchat/llm_judge/data/alpacafarm/question_500.pt')
    else:
        order_questions = torch.load('../FastChat/fastchat/llm_judge/data/alpacafarm/question_500.pt')

if sft_baseline:
    baseline_answer = os.path.join(answer_path,f'{real_names[model_choice]}-sft-ultrachat_200k-{dataset_names[test_dataset_choice]}.jsonl')
else:
    baseline_answer = os.path.join(answer_path,f'{real_names[model_choice]}-dpo-{dataset_names[train_dataset_choice]}-{strategy[strtegy_choice]}-1.0-{dataset_names[test_dataset_choice]}.jsonl')
baseline_answer = load_dataset('json', data_files=baseline_answer)['train']

if test_dataset_choice == 1:
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")['test_prefs']
    map_dict = {item['prompt']:item['chosen'][1]['content'] for item in ds}


# prompt = """
# You are tasked with comparing the responses of two assistants, Assistant A and Assistant B, to a user’s question. Additionally, you will be provided a reference answer to evaluate the quality of both responses of the two assistants. 

# User’s Question:
# {question}

# Reference Answer:
# {ref_answer}

# Assistant A’s Response:
# {response_compare}

# Assistant B’s Response:
# {response_baseline}

# FIRST output 'A', 'B', or "Tie" to indicate your judgement on the these two responses. Then, provide a one-sentence explanation on your choice. 

# The principle of your judgement should focus on the following criteria:

# 1. Do not judge the quality of the answers based on their length.  
# 2. If two answers have no essential difference in meaning and only differ in wording, format or length, please output "Tie"  
# 3. Determine which answer's meaning is closer to the reference answer.  
# 4. The reference answer might not always represent the best response, hence you can take potential improvements to the reference answer into account. 
# 5. The reference answer might not be the unique correct answer for open-ended questions, hence you can take correct and factual alternative answers into account for these type of questions.
# """
prompt1="""
You are tasked with comparing the responses of two assistants, Assistant A and Assistant B, to a user’s question. Additionally, you will be provided with a reference answer to evaluate the quality of the responses from both assistants.

User’s Question:
{question}

Reference Answer:
{ref_answer}

Assistant A’s Response:
{response_compare}

Assistant B’s Response:
{response_baseline}

First, output 'A', 'B', or 'Tie' to indicate your judgment of these two responses. Then, provide a one-sentence explanation for your choice.

The principles for your judgment should consider the following criteria:

1. Do not judge the quality of the two responses based on their length.  
2. Determine which response’s meaning is essentially closer to the reference answer.
3. For open-ended questions, the reference answer might not be the unique correct answer, and you can take correct and factual alternative responses into account for these type of questions. 
4. If the two responses have no essential difference in meaning and correctness, and only differ in wording, format, or length, output ‘Tie’.
"""

prompt2 ='''
You are tasked with comparing the responses of two assistants, Assistant A and Assistant B, to a user’s question. Additionally, you will be provided with a reference answer to evaluate the quality of the responses from both assistants.

User’s Question:
{question}

Reference Answer:
{ref_answer}

Assistant A’s Response:
{response_compare}

Assistant B’s Response:
{response_baseline}

First, output 'A', 'B', or 'Tie' to indicate your judgment of these two responses. Then, provide a one-sentence explanation for your choice.

The principles for your judgment should consider the following criteria:

1. Do not judge the quality of the two responses based on their length.
2. Determine which response’s meaning is essentially closer to the reference answer.
3. Evaluate the responses based on their helpfulness, relevance, accuracy, depth, and level of detail.
4. For open-ended questions, the reference answer might not be the unique correct answer, and you can take correct and factual alternative responses into account for these type of questions. 
5. If the two responses have no essential difference in meaning and correctness, and only differ in wording, format, or length, output ‘Tie’.
'''

# 4. The reference answer might not always represent the best response, hence you can take potential improvements to the reference answer into account. 
# 5. The reference answer might not be the unique correct answer for open-ended questions, hence you can take correct and factual alternative answers into account for these type of questions.
# prompt = "Please act as an impartial judge and compare the quality of the response provided by two AI assistants to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistants' answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. \nQuery: {question}\nResponse A:{response_baseline}\nResponse B:{response_compare}\nFIRST provide a one-sentence comparison of the two responses and explain which you feel is more helpful. SECOND, on a new line, state only 'A' or 'B' to indicate which response is more helpful. Your response should use the format: Comparison: <one-sentence comparison and explanation> More helpful: <'A' or 'B'>"

for strategy_choice in [0]:
    if strategy_choice == 0:
        # dpo_lambda_list = [0.55,0.65,0.75,0.85,0.95,1.0]
        # dpo_lambda_list = [0.96,0.97,0.98,0.99]
        dpo_lambda_list = [0.999,0.9999]
        # dpo_lambda_list = [0.95]
    elif (strategy_choice == 1) or (strategy_choice == 2):
        dpo_lambda_list = [0.95]
    for dpo_lambda in dpo_lambda_list:
        if dpo_lambda == "sft":
            compare_answer = os.path.join(answer_path,f'{real_names[model_choice]}-sft-ultrachat_200k-{dataset_names[test_dataset_choice]}.jsonl')
            output_file_path = f'{real_names[model_choice]}-sft-ultrachat_200k-{dataset_names[test_dataset_choice]}.jsonl'
        else:
            if strategy_choice == 0:
                compare_answer = os.path.join(answer_path,f'{real_names[model_choice]}-dpo-{dataset_names[train_dataset_choice]}-{strategy[strategy_choice]}-{dpo_lambda}-{dataset_names[test_dataset_choice]}.jsonl')
                output_file_path = f'{real_names[model_choice]}-dpo-{dataset_names[train_dataset_choice]}-{strategy[strategy_choice]}-{dpo_lambda}-{dataset_names[test_dataset_choice]}.jsonl'
                if sft_baseline == True:
                    output_file_path = f'{real_names[model_choice]}-dpo-{dataset_names[train_dataset_choice]}-{strategy[strategy_choice]}-{dpo_lambda}-{dataset_names[test_dataset_choice]}-vssft.jsonl'

            elif strategy_choice == 1:
                compare_answer = os.path.join(answer_path,f'{real_names[model_choice]}-dpo-{dataset_names[train_dataset_choice]}-{strategy[strategy_choice]}-{dpo_lambda}to1.0-{dataset_names[test_dataset_choice]}.jsonl')
                output_file_path = f'{real_names[model_choice]}-dpo-{dataset_names[train_dataset_choice]}-{strategy[strategy_choice]}-{dpo_lambda}to1.0-{dataset_names[test_dataset_choice]}.jsonl'
                if sft_baseline == True:
                    output_file_path = f'{real_names[model_choice]}-dpo-{dataset_names[train_dataset_choice]}-{strategy[strategy_choice]}-{dpo_lambda}to1.0-{dataset_names[test_dataset_choice]}-vssft.jsonl'
            elif strategy_choice == 2:
                compare_answer = os.path.join(answer_path,f'{real_names[model_choice]}-dpo-{dataset_names[train_dataset_choice]}-{strategy[strategy_choice]}-1.0to{dpo_lambda}-{dataset_names[test_dataset_choice]}.jsonl')
                output_file_path = f'{real_names[model_choice]}-dpo-{dataset_names[train_dataset_choice]}-{strategy[strategy_choice]}-1.0to{dpo_lambda}-{dataset_names[test_dataset_choice]}.jsonl'
                if sft_baseline == True:
                    output_file_path = f'{real_names[model_choice]}-dpo-{dataset_names[train_dataset_choice]}-{strategy[strategy_choice]}-1.0to{dpo_lambda}-{dataset_names[test_dataset_choice]}-vssft.jsonl'

        # breakpoint()
        compare_answer = load_dataset('json', data_files=compare_answer)['train']
        # if use_old == True:
        output_file_path = os.path.join('prompts', output_file_path)
        # else:
        #     output_file_path = os.path.join('prompts_new', output_file_path)

        # 文件路径，这里以当前目录下的test.txt为例，你可以替换成实际想要检测的文件路径
        if not os.path.exists(output_file_path):
            # 使用open函数以写入模式打开文件，若文件不存在会自动创建，这里只是创建了空文件，若要写入内容可在open后进行相应操作
            with open(output_file_path, 'w') as f:
                pass
        else:
            print(f"{output_file_path} 文件已存在,自动清空")
            open(output_file_path, 'w').close()
        for _,qs in enumerate(order_questions):
            question = qs['turns'][0]
            qs_index = qs['question_id']-1
            if test_dataset_choice == 1:
                ref_answer = map_dict[question]
            else:
                ref_answer = qs['answer']
            response_baseline = baseline_answer[qs_index]['choices'][0]['turns'][0]
            response_compare = compare_answer[qs_index]['choices'][0]['turns'][0]

            prompt = prompt2
            input = prompt.format(
                question = question,
                ref_answer = ref_answer,
                response_baseline = response_baseline,
                response_compare = response_compare,
            )
            messages ={
                'question_id': qs_index+1,
                'role': 'user',
                'content': input
            },


        # output_file_path = f'{real_names[model_choice]}-dpo-{dataset_names[dataset_choice]}-{strategy[strtegy_choice]}-{dpo_lambda}-winrate.jsonl'
            
            with open(output_file_path, 'a') as f:
                f.write(json.dumps(messages[0]) + '\n')
            # for item in result_list:
            #     # 将每个字典对象转换为JSON字符串并写入文件，每行一个JSON对象
            #     f.write(json.dumps(item) + '\n')

        print(f'------------------{output_file_path} is done!!!-------------------')