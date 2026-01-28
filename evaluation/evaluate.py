USE_Q8_QUANTIZATION = False
"""
    A very simple demo to load ChatTS model and use it.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoProcessor
import torch
import matplotlib.pyplot as plt
import numpy as np
import os,sys
from tqdm import tqdm
if USE_Q8_QUANTIZATION:
    from transformers import BitsAndBytesConfig 
import json
import argparse
parser = argparse.ArgumentParser(description="ChatTS评估脚本")


parser.add_argument('--model_name', type=str, required=True, help='模型名称，如 chatts-base')
parser.add_argument('--model_path', type=str, default='NO', help='模型checkpoint路径')
parser.add_argument('--save_folder', type=str, default='', help='模型checkpoint路径')
parser.add_argument('--lab_name', type=str, required=True, help='数据集标签，如 state_sp')
args = parser.parse_args()
save_folder = args.save_folder
data = []
paths = [
    f'./datasets/{args.lab_name}_chatts.json',
]
input_file = next((p for p in paths if os.path.exists(p)), None)

if input_file is None:
    raise FileNotFoundError(f'Cannot find {args.lab_name}_chatts.json in result/ or timeeval/')

data = json.load(open(input_file, 'r'))
result = []
model_name = args.model_name
model_path = model_name
model_alias = save_folder.split('/')[-1].lower()


if args.model_path != 'NO':
    model_path = args.model_path
print(f'Find model {model_name} at {model_path}')
folder_name = save_folder.split('/')[-1]
output_file_path = f"{save_folder}/data/{folder_name}_{args.lab_name}.json"
if os.path.exists(output_file_path):
    print(f"文件 {output_file_path} 已存在，程序退出")
    sys.exit(0)  
output_file_folder = f"{save_folder}/data/"
os.makedirs(output_file_folder, exist_ok=True)
if 'chatts' in model_alias or 'feelts' in model_alias or 'chats' in model_alias:
    if USE_Q8_QUANTIZATION:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='cuda:0', quantization_config=quantization_config, torch_dtype="float16")
    elif 'grpo' in model_alias:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='cuda:0', torch_dtype='bfloat16')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='cuda:0', torch_dtype='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, add_generation_prompt=True,
        enable_thinking=False)
    from tqdm import tqdm

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, tokenizer=tokenizer)
    

    for i in tqdm(range(len(data))):
        if 'llm' not in args.lab_name:
            ...
        else :
            data[i]['timeseries'] = []
        try :
            ts = data[i]['timeseries']
        except:
            data[i]['timeseries'] = []
            ts = []
        import numpy as np
        prompt = data[i]['question']
        if 'grpo' in  model_alias:
            print(f'in gpro model -- tokenize')
            messages = [
                {"role": "system", "content": "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format, with the answer included between the <answer> and </answer> tags: <think>\n...\n</think>\n<answer>\nYour answer\n</answer>"},
                {"role": "user", "content": prompt}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                chat_template="{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {%- if messages[0]['content'] is string %}{{- messages[0]['content'] }}{%- else %}{%- for item in messages[0]['content'] %}{%- if item.text is defined and item.text %}{{- item.text }}{%- endif %}{%- endfor %}{%- endif %}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\n' }}{%- if messages[0]['content'] is string %}{{- messages[0]['content'] }}{%- else %}{%- for item in messages[0]['content'] %}{%- if item.text is defined and item.text %}{{- item.text }}{%- endif %}{%- endfor %}{%- endif %}{{- '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' }}{%- if message.content is string %}{{- message.content }}{%- else %}{%- for item in message.content %}{%- if item.text is defined and item.text %}{{- item.text }}{%- endif %}{%- endfor %}{%- endif %}{{- '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {%- if message.content is string %}{{- '\\n' + message.content }}{%- else %}{{- '\\n' }}{%- for item in message.content %}{%- if item.text is defined and item.text %}{{- item.text }}{%- endif %}{%- endfor %}{%- endif %}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n<think>\n' }}\n{%- endif %}\n",
                tokenize=False,
                add_generation_prompt=True
            )
        else :
            prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n"
        def get_dimension(lst):
            if not isinstance(lst, list):
                return 0
            if not lst:  # 空列表视为 1 维
                return 1
            return 1 + max(get_dimension(item) for item in lst)
        if ts != [] and get_dimension(ts) == 1:
            ts = [ts]
        if ts != []:
            inputs = processor(text=[prompt], timeseries=ts, padding=True, return_tensors="pt")
        else :
            inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        inputs = dict((k, v.to(0)) for k, v in inputs.items())
        outputs = model.generate(
                        **inputs,
                        max_new_tokens=5000,
                        temperature=0.1,
                        use_cache=False,
                        max_time=60.0
                    )

        input_len = inputs['attention_mask'][0].sum().item()
        output = outputs[0][input_len:]
        text_out = tokenizer.decode(output, skip_special_tokens=True)
        result.append(text_out)
else :
    ctx_length = 16240
    num_gpus = 1
    gpu_per_model = 1
    batch_size = 1
    ENGINE = 'vllm'
    MULTIPROCESS = True
    from vllm import LLM, SamplingParams
    sampling_params = SamplingParams(temperature=0.01, top_p=0.95, max_tokens=ctx_length, stop_token_ids=[151643, 151645], stop=['<|endoftext|>', '<|im_end|>'])
    llm = LLM(model=model_name, trust_remote_code=True, max_model_len=ctx_length, tensor_parallel_size=gpu_per_model, gpu_memory_utilization=0.95)
    for d in tqdm(data, desc='generating'):
        if 'timeseries' not in d:
            d['timeseries'] = []
        ts = d['timeseries']
        if ts != [] and type(ts[0]) != list: ts = [ts]
        for _ts in ts:
            ts_str = str(_ts)
            d['question'] = d['question'].replace('<ts><ts/>', ts_str, 1)
        q = d['question']
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{q}<|im_end|><|im_start|>assistant\n"
        tokenizer = llm.get_tokenizer()
        input_ids = tokenizer(question, return_tensors="pt")["input_ids"]

        model_max_length = llm.llm_engine.model_config.max_model_len
        safe_max_tokens = min(ctx_length, model_max_length)
        if input_ids.shape[1] > model_max_length:
            input_ids = input_ids[:, -model_max_length:]
            question = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        question = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        sampling_params = SamplingParams(
            temperature=0.01,
            top_p=0.95,
            max_tokens=safe_max_tokens,
            stop_token_ids=[151643, 151645],
            stop=['<|endoftext|>', '<|im_end|>']
        )
        a = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        result.append(a)
json.dump(result, open(output_file_path, 'w'), ensure_ascii=False, indent=4)