import json
import re
import multiprocessing
from tqdm import tqdm
import json
import os
import re
import numpy as np
import random
import time
import traceback
from typing import *

EVA_MODEL_PATH = "./models/Qwen2.5-14B-Instruct"
ctx_length = 16240
num_gpus = 1
gpu_per_model = 1
batch_size = 1
ENGINE = 'vllm'
MULTIPROCESS = True
from vllm import LLM, SamplingParams
sampling_params = SamplingParams(temperature=0.01, top_p=0.95, max_tokens=ctx_length, stop_token_ids=[151643, 151645], stop=['<|endoftext|>', '<|im_end|>'])

from tqdm import tqdm
def try_fix_json(json_string, special_words=['question', 'answer', 'success', 'reference', ',', ':', '\n', '}', '{']):
    quotes_indices = [m.start() for m in re.finditer(r'"', json_string)]
    fixed_json = list(json_string)
    for i in quotes_indices:
        for special in special_words:
            if json_string[i + 1:].startswith(special) or json_string[:i].endswith(': '):
                break
        else:
            fixed_json[i] = r'\"'

    result = ''.join(fixed_json)
    result = result.replace('True', 'true').replace('False', 'false')

    result = re.sub(r'"\s*\n\s*"', '",\n"', result)

    return result

def escape_newlines_in_quotes(json_string):
    matches = list(re.finditer(r'(?<!\\)"([^"\\]*(?:\\.[^"\\]*)*)"', json_string, re.DOTALL))
    fixed_json = []
    last_end = 0
    
    for match in matches:
        start, end = match.span()
        text_between_quotes = json_string[start:end]
        escaped_text = text_between_quotes.replace('\n', '\\n')
        fixed_json.append(json_string[last_end:start])
        fixed_json.append(escaped_text)
        last_end = end
    
    fixed_json.append(json_string[last_end:])
    
    return ''.join(fixed_json)

def parse_llm_json(json_string, special_words=['question', 'answer', 'success', 'reference', ',', ':', '\n', '}', '{']):
    json_string = json_string.replace('```json', '').replace('```', '')
    try:
        json.loads(json_string)
    except Exception as err:
        json_string = try_fix_json(json_string, special_words)
        json_string = escape_newlines_in_quotes(json_string)
    try:
        ans = json.loads(json_string)
    except Exception as e:
        print(e)
        ans = {}
    return ans
def culculate_base_extreme_acc(answer,target,info):
    def parse_answer(answer):
        prompt = """
        你是一个文本分析大师，接下来我会给你一段文本请你帮我提取出我想要的信息，这段文本是：{answer}。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json格式的提取内容
        """
        json_template = """
        {"extreme": "提取的极值信息，只包含数字","index": "提取的位置信息，索引从0开始，如果没有则是-1"} """ # 注意json——template要独立出来，防止prompt中有多个 {} 造成干扰
        question = prompt.format(answer=answer,json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['extreme', 'index'])
        data = json_str
        ans_ext = data['extreme']
        ans_index = data['index']
        return ans_ext, ans_index
    def parse_target(target):
        ext,ind = target.split('，位置 ')
        return float(ext), int(ind)
    count = 0
    corr = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        count+=2
        aex, ain = parse_answer(ans)
        tex, tin = parse_target(tar)
        print(f'TRUE : {tex} {tin}')
        print(f'GEN : {aex} {ain}')
        if aex == tex:
            corr+=1
        if ain == tin:
            corr+=1
        infoi['score'] = f'{corr/count}'
        infoi['meta'] = f'[Extreme] TRUE : {tex} {tin} | GEN : {aex} {ain}'
    return corr/count

def culculate_base_changepoint_acc(answer,target,info):
    def parse_answer(answer):
        prompt = """
        你是一个文本分析大师，接下来我会给你一段文本请你帮我提取出我想要的信息，这段文本是：{answer}。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json
        """
        json_template = """{"is_exist": true or false,"index": "提取的位置信息，索引从0开始，如果没有则是-1"}"""
        question = prompt.format(answer=answer,json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['is_exist', 'index'])
        data = json_str
        ans_ext = data['is_exist']
        ans_index = data['index']
        return ans_ext, ans_index
    def parse_target(target):
        ext,ind = target.split('，变点大致出现在位置')
        if '是' in ext:
            return True, int(ind)
        return False, int(ind)
    count = 0
    corr = 0
    for ans, tar, infoi in tqdm(zip(answer, target,info)):
        count+=2
        aex, ain = parse_answer(ans)
        tex, tin = parse_target(tar)
        if aex == tex:
            corr+=1
        if ain == tin:
            corr+=1
        infoi['score'] = f'{corr/count}'
        infoi['meta'] = f'[ChangePoint] TRUE : {tex} {tin} | GEN : {aex} {ain}'
    return corr/count
def culculate_ts_index_acc(answer, target, info):
    def parse_answer_target(answer, target):
        prompt = """
        你是一个文本分析大师，接下来我会给你两段文，都是在描述一个函数曲线中同一个点的值，第一段是我的答案文本，会描述得比较详实：{answer}
        第二段是目标文本，直接给出了对应点值的真实值：{target}
        请上判断，我的答案文本是否是正确的，如果一致则回答为‘是’，否则为‘否’。（输出只包括“是”或“否”）
        """
        question = prompt.format(answer=answer,target=target)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        if '是' in model_output:
            return True
        return False
    count = 0
    corr = 0
    for ans, tar, infoi in tqdm(zip(answer, target,info)):
        count +=1
        flag = parse_answer_target(ans,tar)
        print(f'ans {ans} tar {tar} flag {flag}')
        if flag:
            corr+=1
        infoi['score'] = f'{corr/count}'
        infoi['meta'] = f'[Trend] {flag}'
    return corr/count
def culculate_base_trend_acc(answer,target,info):
    def parse_answer_target(answer, target):
        prompt = """
        你是一个文本分析大师，接下来我会给你两段文本，第一段是我的答案文本，会描述得比较详实：{answer}
        第二段是目标文本，是对时间序列的简单描述：{target}
        请你从语义上判断，这两段文本对时间序列的趋势描述是否基本一致，如果一致则你的回答为‘是’，否则为‘否’。（输出只包括“是”或“否”）
        """
        question = prompt.format(answer=answer,target=target)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        if '是' in model_output:
            return True
        return False
    count = 0
    corr = 0
    for ans, tar, infoi in tqdm(zip(answer, target,info)):
        count +=1
        flag = parse_answer_target(ans,tar)
        print(f'ans {ans} tar {tar} flag {flag}')
        if flag:
            corr+=1
        infoi['score'] = f'{corr/count}'
        infoi['meta'] = f'[Trend] {flag}'
    return corr/count


def culculate_range_f1(answer,target,info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段答案文本请你帮我提取出我想要的信息。答案文本是关于判断一条时间序列是否存在异常，如果存在则指出异常的区间。对区间的表述通过描述起点、终点的位置实现（如点24到点35，或第24个点到第35个点，指的都是区间（24~35））。需要判断的答案文本是：{answer}。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，同时希望你不要给出过多的解释，回答中只包含一段json代码。如果你发现整段区间都是异常，则说明区间中不存在异常。json中只可以包含一个dict。
        """
        json_template = """{"is_exist": "bool类型，true or false" 如果不存在异常则为false，存在则为true,"start": "int类型，表示异常区间的起点，仅有一个数字","end": "int类型，表示异常区间的终点，仅有一个数字"}"""
        question = prompt.format(answer=answer,json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['is_exist', 'start', 'end'])
        data = json_str
        ans_exist = data['is_exist'] if data['start'] is not None else 'true'
        ans_start = data['start'] if type(data['start']) == int else 0
        ans_end = data['end'] if type(data['end']) == int else 0
        return ans_exist, ans_start, ans_end
    def parse_target(target):
        if '异常' in target:
            return False, -1, -1
        start, end = target.split('-')
        return True, int(start), int(end)

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target,info)):
        aex, ast, aen = parse_answer(ans)
        tex, tst, ten = parse_target(tar)
        count +=1
        if tex == False and aex==False:
            total += 1
            continue
        if tex ==False and aex==True:
            total += 0
            continue
        overlap_start = max(ast, tst)
        overlap_end = min(aen, ten)
        overlap_length = max(0, overlap_end-overlap_start)
        recall = overlap_length/(ten-tst) if ten>tst else 1
        precison = overlap_length/(aen-ast) if aen>ast else 1
        total += 2*recall*precison/(recall+precison) if recall*precison!=0 else 0
        infoi['score'] = f'{2*recall*precison/(recall+precison) if recall*precison!=0 else 0}'
        infoi['meta'] = f'[Range] TRUE : {tex} {tst} {ten} | GEN : {aex} {ast} {aen}'
    return total/count

def culculate_ts_index2(answer,target,info):
    def parse_answer(answer):
        prompt = """你是一个文本提取大师，接下来我会给你一段文本请你帮我提取出我想要的信息，这段文本是：{answer}。
        这段文本是回答一个曲线中某个点的值是多少，我需要你提取出答案中的这个点值
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，同时希望你不要给出过多的解释，回答中只包含一段json代码。json中只可以包含一个dict。如果没有找到的对应值的话，返回值0.
        """
        json_template = """{"point_values": "答案找出的点的值大小，仅有一个数字，保留二位小数"}"""
        question = prompt.format(answer=answer,json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['point_values'])
        data = json_str
        try : 
            ans = float(data['point_values'])
        except Exception as e:
            print(e)
            exit()
        return ans
    def parse_target(target):
        if '异常' in target:
            return False, -1, -1
        start, end = target.split('-')
        return True, int(start), int(end)

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target,info)):
        aex = float(parse_answer(ans))
        tex = float(tar)
        count +=1
        if abs(aex-tex) < 5: total+=1
        infoi['score'] = f'{total}'
        infoi['meta'] = f'[Range] TRUE : {tex} | GEN : {aex}'
    return total/count

def culculate_multi_f1(answer,target,info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本请你帮我提取出我想要的信息，这段文本是：{answer}。答案指出了哪几条时间序列存在异常情况，请提取出答案文本指出的时间序列标号。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中只包含一个值为列表的dict
        """
        json_template = """{ "indexes":[...] }//发生异常的时间序列序号，用list的形式给出 """ # 注意要将含有 {} 的内容单独出来，防止对prompt.format造成干扰
        question = prompt.format(answer=answer,json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['indexes'])
        data = json_str
        indexes = data['indexes']
        return indexes
    def parse_target(target):
        indexes = []
        for i in target:
            indexes.append(i['index']+1)
        return indexes
    count = 0
    total = 0
    for ans, tar,infoi in tqdm(zip(answer, target,info)):
        try :
            ains = parse_answer(ans)
            tins = parse_target(tar)
            intersize = len(set(ains) & set(tins))
            recall = intersize/len(tins) if len(tins)>0 else 1
            precision = intersize/len(ains) if len(ains)>0 else 1
            count+=1
            total+= 2*precision*recall/(precision+recall) if precision*recall>0 else 0
            infoi['score'] = f'{2*recall*precision/(recall+precision) if recall*precision!=0 else 0}'
            infoi['meta'] = f'[Multi] TRUE : {tins} | GEN : {ains}'
        except Exception as e:
            count += 1
            infoi['score'] = 0
            infoi['meta'] = f'[ERROR] {e}'
    return total/count


def culculate_state_score(answer,target,info):
    score_table = {
        'normal':{
            'normal':3,
            'low':2,
            'mid':1,
            'high':0
        },
        'low':{
            'normal':1,
            'low':3,
            'mid':2,
            'high':1
        },
        'mid':{
            'normal':0,
            'low':1,
            'mid':3,
            'high':2
        },
        'high':{
            'normal':0,
            'low':0,
            'mid':1,
            'high':3
        },
    }
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本请你帮我提取出我想要的信息，这段文本是：{answer}。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中只包含一个dict
        """
        json_template = """{"state": "..."} 文本中提取的状态信息，答案从['normal','low','mid','high']中选择(只包含一个答案)}"""
        question = prompt.format(answer=answer,json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['state'])
        data = json_str
        state = data['state']
        if 'low' in state:
            return 'low'
        if 'mid' in state:
            return 'mid'
        if 'high' in state:
            return 'high'
        return 'normal'

    def parse_target(target):
        if '正常' in target:
            return 'normal'
        elif 'nor' in target:
            return 'normal'
        elif '低' in target:
            return 'low'
        elif 'low' in target:
            return 'low'
        elif 'mid' in target:
            return 'mid'
        elif '中' in target:
            return 'mid'
        elif '高' in target:
            return 'high'
        elif 'high' in target:
            return 'high'

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target,info)):
        count+=1
        a = parse_answer(ans)
        t = parse_target(tar)
        total += score_table[t][a]/3
        infoi['score'] = f'{score_table[t][a]/3}'
        infoi['meta'] = f'[State] TRUE : {t} | GEN : {a}'
    return total/count
def stage1_base_acc(answer, target, info):
    def parse_answer(answer):
        prompt = """
        你是一个精确的文本分析专家。我将给你一段文本，这是模型对问题的回答："""+answer+"""。
        请从文本中提取以下信息：
        - value: 回答中提到的数值（如最小值或均值），只包含数字，可能带有小数点。如果没有明确的数值，则为空字符串。
        - index: 回答中提到的位置索引（从0开始计数），如果是整数。如果没有提到位置，则为-1。如果有多个，取最靠前的。
        输出必须是严格的JSON格式，不要添加任何额外文本、解释或代码块。
        示例输出：
        {"value": " -1.04", "index": "194"}
        或
        {"value": "0.45", "index": "-1"}
        """
        json_template = ""  # No need for separate template, integrated into prompt
        question = prompt
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['value', 'index'])
        data = json_str
        ans_value = data['value']
        ans_index = data['index']
        return ans_value, ans_index

    def parse_target(target):
        tv = target['value']
        ti = target.get('index', -1)
        return float(tv), int(ti)

    total_score = 0.0
    num_items = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        av_str, ai_str = parse_answer(ans)
        try:
            av = float(av_str) if av_str else float('nan')
            ai = int(ai_str) if ai_str != '-1' else -1
        except ValueError:
            av = float('nan')
            ai = -1
        tv, ti = parse_target(tar)

        score = 0.0
        value_correct = abs(av - tv) <= 0.1
        has_index = ti != -1

        if value_correct:
            score += 0.5 if has_index else 1.0

        if has_index:
            index_correct = abs(ai - ti) <= 1
            if index_correct:
                score += 0.5

        infoi['score'] = f'{score}'
        infoi['meta'] = f'[Value] TRUE : {tv} {ti} | GEN : {av} {ai}'
        print(f'TRUE : {tv} {ti}')
        print(f'GEN : {av} {ai}')

        total_score += score
        num_items += 1

    if num_items == 0:
        return 0.0
    return total_score / num_items
def stage2_calc_shape(answer, target, info):
    def parse_answer(answer):
        prompt = """
        你是一个精确的文本分析专家。我将给你一段文本，这是模型对问题的回答："""+answer+"""。
        请从文本中提取以下信息：
        - start: 回答中提到的异常区间的起始点索引（整数，从0开始计数）。如果没有提到，则为-1。
        - end: 回答中提到的异常区间的结束点索引（整数，从0开始计数）。如果没有提到，则为-1。如果有多个区间，取最主要的或第一个提到的。
        - segtype: 回答中提到的异常类型，标准化为以下英文之一："up"（上升、rising、ascending、increase）、"down"（下降、descending、decrease、fall）、"oscillate"（振荡、vibrate、fluctuate、wave）、"platform"（平台、flat、constant、steady、plateau）。如果不匹配任何类型，则为空字符串。
        输出必须是严格的JSON格式，不要添加任何额外文本、解释或代码块。
        示例输出：
        {"start": "1", "end": "23", "segtype": "platform"}
        或
        {"start": "-1", "end": "-1", "segtype": ""}
        """
        question = prompt
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['start', 'end', 'segtype'])
        data = json_str
        ans_start = data['start']
        ans_end = data['end']
        ans_segtype = data['segtype']
        return ans_start, ans_end, ans_segtype

    def parse_target(target):
        ts = target['start']
        te = target['end']
        tt = target['segtype']
        return int(ts), int(te), tt

    def compute_interval_f1(true_start, true_end, pred_start, pred_end):
        if true_start > true_end or pred_start > pred_end:
            return 0.0
        intersection_start = max(true_start, pred_start)
        intersection_end = min(true_end, pred_end)
        if intersection_start > intersection_end:
            return 0.0
        intersection = intersection_end - intersection_start + 1
        true_len = true_end - true_start + 1
        pred_len = pred_end - pred_start + 1
        precision = intersection / pred_len
        recall = intersection / true_len
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    total_score = 0.0
    num_items = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        as_str, ae_str, at_str = parse_answer(ans)
        try:
            a_start = int(as_str) if as_str != '-1' else -1
            a_end = int(ae_str) if ae_str != '-1' else -1
            a_segtype = at_str if at_str else ''
        except ValueError:
            a_start = -1
            a_end = -1
            a_segtype = ''
        t_start, t_end, t_segtype = parse_target(tar)

        score = 0.0
        segtype_correct = (a_segtype == t_segtype)
        if segtype_correct:
            score += 0.5

        f1 = compute_interval_f1(t_start, t_end, a_start, a_end)
        score += 0.5 * f1

        infoi['score'] = f'{score}'
        infoi['meta'] = f'[Shape] TRUE : {t_start}-{t_end} {t_segtype} | GEN : {a_start}-{a_end} {a_segtype}'
        print(f'TRUE : {t_start} {t_end} {t_segtype}')
        print(f'GEN : {a_start} {a_end} {a_segtype}')

        total_score += score
        num_items += 1

    if num_items == 0:
        return 0.0
    return total_score / num_items
def stage2_calc_trend(answer, target, info):
    def parse_answer(answer):
        prompt = """
        你是一个精确的文本分析专家。我将给你一段文本，这是模型对问题的回答："""+answer+"""。
        请从文本中提取以下信息：
        - trend_type: 回答中提到的趋势类型，标准化为以下英文之一："up"（上升、rising、ascending、increase）、"down"（下降、descending、decrease、fall）、"stable"（平稳、steady、flat、constant、plateau）、"shake"（振荡、oscillate、vibrate、fluctuate、wave）。如果不匹配任何类型或没有提到，则为空字符串""。
        输出必须是严格的JSON格式，不要添加任何额外文本、解释或代码块。
        示例输出：
        {"trend_type": "down"}
        或
        {"trend_type": ""}
        """
        question = prompt
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['trend_type'])
        data = json_str
        ans_trend_type = data['trend_type']
        return ans_trend_type

    def parse_target(target):
        tt = target['trend_type']
        return tt

    total_score = 0.0
    num_items = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        a_type = parse_answer(ans)
        if not a_type:
            a_type = ''

        t_type = parse_target(tar)

        score = 1.0 if a_type == t_type else 0.0

        infoi['score'] = f'{score}'
        infoi['meta'] = f'[Trend] TRUE : {t_type} | GEN : {a_type}'
        print(f'TRUE : {t_type}')
        print(f'GEN : {a_type}')

        total_score += score
        num_items += 1

    if num_items == 0:
        return 0.0
    return total_score / num_items    
def stage2_calc_period(answer, target, info):
    def parse_answer(answer):
        prompt = """
        你是一个精确的文本分析专家。我将给你一段文本，这是模型对问题的回答："""+answer+"""。
        请从文本中提取以下信息：
        - period_type: 回答中提到的周期类型，标准化为以下英文之一："sine"（正弦波、sine wave、sinusoidal、sine-like、波状）、"square"（方波、square wave、rectangular、pulse、step）。如果不匹配任何类型或没有提到，则为空字符串""。
        输出必须是严格的JSON格式，不要添加任何额外文本、解释或代码块。
        示例输出：
        {"period_type": "sine"}
        或
        {"period_type": ""}
        """
        question = prompt
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['period_type'])
        data = json_str
        ans_period_type = data['period_type']
        return ans_period_type

    def parse_target(target):
        pt = target['period_type']
        return pt

    total_score = 0.0
    num_items = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        a_type = parse_answer(ans)
        if not a_type:
            a_type = ''

        t_type = parse_target(tar)

        score = 1.0 if a_type == t_type else 0.0

        infoi['score'] = f'{score}'
        infoi['meta'] = f'[Period] TRUE : {t_type} | GEN : {a_type}'
        print(f'TRUE : {t_type}')
        print(f'GEN : {a_type}')

        total_score += score
        num_items += 1

    if num_items == 0:
        return 0.0
    return total_score / num_items
def stage2_calc_point(answer, target, info):
    def parse_answer(answer):
        prompt = """
        你是一个精确的文本分析专家。我将给你一段文本，这是模型对问题的回答："""+answer+"""。
        请从文本中提取以下信息：
        - spike_type: 回答中提到的异常点类型，标准化为以下英文之一："up"（上突、upward spike、upper outlier、positive anomaly）、"down"（下突、downward spike、lower outlier、negative anomaly）。如果不匹配任何类型或没有提到，则为空字符串""。
        - point: 回答中提到的异常点的位置索引（整数，从0开始计数）。如果没有提到，则为-1。如果有多个，取最主要的或第一个提到的。
        输出必须是严格的JSON格式，不要添加任何额外文本、解释或代码块。
        示例输出：
        {"spike_type": "down", "point": "43"}
        或
        {"spike_type": "", "point": "-1"}
        """
        question = prompt
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['spike_type', 'point'])
        data = json_str
        ans_spike_type = data['spike_type']
        ans_point = data['point']
        return ans_spike_type, ans_point
    def parse_target(target):
        st = target['spike_type']
        p = target['point']
        return st, int(p)
    total_score = 0.0
    num_items = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        a_type, a_point_str = parse_answer(ans)
        try:
            a_point = int(a_point_str) if a_point_str != '-1' else -1
        except ValueError:
            a_point = -1
        if not a_type:
            a_type = ''

        t_type, t_point = parse_target(tar)

        type_score = 0.5 if a_type == t_type else 0.0
        point_score = 0.5 if (a_point != -1 and abs(a_point - t_point) <= 1) else 0.0

        score = type_score + point_score

        infoi['score'] = f'{score}'
        infoi['meta'] = f'[Point] TRUE : {t_type} {t_point} | GEN : {a_type} {a_point}'
        print(f'TRUE : {t_type} {t_point}')
        print(f'GEN : {a_type} {a_point}')

        total_score += score
        num_items += 1

    if num_items == 0:
        return 0.0
    return total_score / num_items
def generated_change_trend_qa(answer, target, info):
    def parse_answer(answer):
        prompt = """
        你是一个精确的文本分析专家。我将给你一段文本，这是模型对问题的回答："""+answer+"""。
        请从文本中提取以下信息：
        - consistent: 回答中提到的前后趋势是否一致的判断，标准化为字符串"True"或"False"。如果没有提到，则为"False"。
        - pre_trend: 回答中提到的变点前趋势类型，标准化为以下英文之一："up"（上升、rising、ascending、increase）、"down"（下降、descending、decrease、fall）、"stable"（平稳、steady、flat、constant、plateau）。如果不匹配任何类型或没有提到，则为空字符串""。
        - post_trend: 回答中提到的变点后趋势类型，标准化为以下英文之一："up"（上升、rising、ascending、increase）、"down"（下降、descending、decrease、fall）、"stable"（平稳、steady、flat、constant、plateau）。如果不匹配任何类型或没有提到，则为空字符串""。
        输出必须是严格的JSON格式，不要添加任何额外文本、解释或代码块。
        示例输出：
        {"consistent": "False", "pre_trend": "up", "post_trend": "down"}
        或
        {"consistent": "", "pre_trend": "", "post_trend": ""}
        """
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['consistent', 'pre_trend', 'post_trend'])
        data = json_str
        ans_consistent = data['consistent']
        ans_pre_trend = data['pre_trend']
        ans_post_trend = data['post_trend']
        return ans_consistent, ans_pre_trend, ans_post_trend
    def parse_target(target):
        parts = target.split(',')
        if len(parts) != 3:
            return False, '', ''
        consistent_str = parts[0].strip()
        pre = parts[1].strip()
        post = parts[2].strip()
        t_consistent = True if consistent_str == 'True' else False
        return t_consistent, pre, post
    total_score = 0.0
    num_items = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        a_cons_str, a_pre, a_post = parse_answer(ans)
        a_consistent = True if a_cons_str == 'True' else (False if a_cons_str == 'False' else None)
        if a_consistent is None:
            a_consistent = False  # Default to False if unclear
        if not a_pre:
            a_pre = ''
        if not a_post:
            a_post = ''

        t_consistent, t_pre, t_post = parse_target(tar)
        score = 0.0
        if a_consistent == t_consistent:
            score += 1/3
        if a_pre == t_pre:
            score += 1/3
        if a_post == t_post:
            score += 1/3
        infoi['score'] = f'{score:.3f}'
        infoi['meta'] = f'[Change Trend] TRUE : {t_consistent} {t_pre} {t_post} | GEN : {a_consistent} {a_pre} {a_post}'
        print(f'TRUE : {t_consistent} {t_pre} {t_post}')
        print(f'GEN : {a_consistent} {a_pre} {a_post}')
        total_score += score
        num_items += 1
    if num_items == 0:
        return 0.0
    return total_score / num_items
  
def multi(answer,target,info):
    return culculate_multi_f1(answer, target,info)
def base(answer, target,info):
    return (2*culculate_base_extreme_acc(answer[:10],target[:10],info[:10])+culculate_base_changepoint_acc(answer[10:15],target[10:15],info[10:15])+culculate_base_trend_acc(answer[-5:],target[-5:],info[-5:]))/4
def state(answer,target,info):
    return culculate_state_score(answer,target,info)
def double(answer,target,info):
    return culculate_range_f1(answer,target,info)
def ts_index(answer, target, info):
    return culculate_ts_index2(answer, target, info)
def double_check_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答时间序列B中是否存在明显的和时间序列A不同的地方，以及如果存在，请指出不同的区间是从第几个点到第几个点。请你帮我提取出我想要的信息，这段文本是：{answer}。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中只包含两个键的dict：has_difference为布尔值，differences_interval为列表（如果存在差异则为[start, end]两个整数的列表，否则为空列表）
        """
        json_template = """{ "has_difference": true/false, "differences_interval": [...] }//has_difference表示是否存在差异，differences_interval表示差异区间，用[start, end]列表给出或空列表"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['has_difference', 'differences_interval'])
        data = json_str
        has_difference = data['has_difference']
        differences_interval = data['differences_interval']
        return has_difference, differences_interval

    def parse_target(target):
        has_difference = target['has_difference']
        differences_interval = target['differences_interval'] if has_difference else []
        return has_difference, differences_interval

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        try:
            a_has, a_interval = parse_answer(ans)
            t_has, t_interval = parse_target(tar)
            if a_has != t_has:
                score = 0.0
            else:
                if not t_has:
                    score = 1.0
                else:
                    if len(a_interval) != 2 or len(t_interval) != 2:
                        score = 0.0
                    else:
                        a_start, a_end = sorted(a_interval)
                        t_start, t_end = sorted(t_interval)
                        inter_start = max(a_start, t_start)
                        inter_end = min(a_end, t_end)
                        inter_len = max(0, inter_end - inter_start + 1)
                        a_len = a_end - a_start + 1
                        t_len = t_end - t_start + 1
                        precision = inter_len / a_len if a_len > 0 else 0
                        recall = inter_len / t_len if t_len > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        score = f1
            infoi['score'] = f'{score}'
            infoi['meta'] = f'[DoubleCheck] TRUE: has={t_has}, interval={t_interval} | GEN: has={a_has}, interval={a_interval}'
        except Exception as e:
            score = 0.0
            infoi['score'] = f'{score}'
            infoi['meta'] = f'[ERROR] TRUE: {t_has}, {t_interval} | GEN: {a_has}, {a_interval}'
        count += 1
        total += score
    return total / count if count > 0 else 0
def relative_trend(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答时间序列中第一个趋势以及第二个趋势相较于第一个趋势的变化情况。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中第一个趋势（上升、下降或平稳）以及第二个趋势相较于第一个趋势的变化（上升、下降、新稳态或加速/减速+上升/下降）。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中只包含两个键的dict：first_trend为字符串，relative_trend为字符串
        """
        json_template = """{ "first_trend": "上升/下降/平稳", "relative_trend": "上升/下降/新稳态/加速上升/加速下降/减速上升/减速下降" }//first_trend表示第一个趋势，relative_trend表示第二个趋势相较于第一个趋势的变化"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['first_trend', 'relative_trend'])
        data = json_str
        first_trend = data['first_trend']
        relative_trend = data['relative_trend']
        return first_trend, relative_trend

    def parse_target(target):
        first_trend = target['first_trend']
        relative_trend = target['relative_trend']
        return first_trend, relative_trend

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        a_first, a_relative = parse_answer(ans)
        t_first, t_relative = parse_target(tar)
        score = 0.0
        if a_first == t_first:
            score += 0.3
        if a_relative == t_relative:
            score += 0.7
        else:
            if t_relative in ['新稳态'] and a_relative in ['平稳']:
                score += 0.1
            elif t_relative.startswith('加速') or t_relative.startswith('减速'):
                base_trend = t_relative[2:]  # 提取加速/减速后的趋势
                if a_relative == base_trend:
                    score += 0.1
            elif a_relative in ['上升', '下降'] and t_relative in ['上升', '下降']:
                if a_relative == t_relative:
                    score += 0.1
        infoi['score'] = f'{score}'
        infoi['meta'] = f'[RelativeTrend] TRUE: first={t_first}, relative={t_relative} | GEN: first={a_first}, relative={a_relative}'
        count += 1
        total += score
    return total / count if count > 0 else 0
def segment_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答时间序列中趋势变化点(分割点)的位置。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中列出的所有变化点，按从小到大的顺序列出。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中只包含一个键的dict：split_points为整数列表，标答回答中提到的分割点的位置
        """
        json_template = """{ "split_points": [...] }//变化点列表，按从小到大顺序的整数列表"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['split_points'])
        data = json_str
        try :
            data['change_points'] = data['split_points']
            change_points = sorted(data['change_points'])
        except :
            change_points = []
        return change_points
    
    def parse_target(target):
        change_points = sorted(target['change_points'])
        return change_points
    def compute_tp_with_tolerance(a, t, tol=5):
        if not a and not t:
            return 0
        i, j = 0, 0
        tp = 0
        while i < len(a) and j < len(t):
            if abs(a[i] - t[j]) <= tol:
                tp += 1
                i += 1
                j += 1
            elif a[i] < t[j]:
                i += 1
            else:
                j += 1
        return tp
    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        a_points = parse_answer(ans)
        t_points = parse_target(tar)
        one_shot = 0.0
        if len(a_points) == len(t_points):
            all_match = all(abs(a_points[i] - t_points[i]) <= 5 for i in range(len(a_points)))
            if all_match:
                one_shot = 1.0
        tp = compute_tp_with_tolerance(a_points, t_points, 5)
        precision = tp / len(a_points) if len(a_points) > 0 else 0.0
        recall = tp / len(t_points) if len(t_points) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        score = max(one_shot, f1)
        infoi['score'] = f'{score}'
        infoi['meta'] = f'[ChangePoints] TRUE: {t_points} | GEN: {a_points} | 1-shot: {one_shot}, F1: {f1}'
        count += 1
        total += score
    return total / count if count > 0 else 0
def multi_anomaly_rca(answer, target, info):
    def normalize_rank(rank):
        """
        强制 rank 变成 List[int]
        """
        if not isinstance(rank, list):
            return []

        out = []
        for x in rank:
            if isinstance(x, list):
                out.extend(x)
            elif isinstance(x, (int, str)):
                try:
                    out.append(int(x))
                except:
                    pass
        return out
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答时间序列中的异常根因和传播路径。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中指出的异常根因时序（一个整数）和传播路径（按顺序的整数列表，从根因开始）。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中只包含两个键的dict：root_cause为整数，rank为整数列表
        """
        json_template = """{ "root_cause": 整数, "rank": [...] }//root_cause表示异常根因的时序整数标号，rank表示传播路径的有序数字列表"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['root_cause', 'rank'])
        data = json_str
        try :
            root_cause = data['root_cause']
        except : 
            root_cause = -1
        try:
            rank = normalize_rank(data['rank'])
        except :
            rank = []
        return root_cause, rank

    def parse_target(target):
        root_cause = target['root_cause']
        rank = target['rank']
        return root_cause, rank

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        try :
            a_root, a_rank = parse_answer(ans)
            t_root, t_rank = parse_target(tar)
            root_score = 0.5 if a_root == t_root else 0.0
            inter_size = len(set(a_rank) & set(t_rank))
            precision = inter_size / len(a_rank) if len(a_rank) > 0 else 0.0
            recall = inter_size / len(t_rank) if len(t_rank) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if len(a_rank) == len(t_rank):
                correct_count = sum(a_rank[i] == t_rank[i] for i in range(len(t_rank)))
                percentage = correct_count / len(t_rank) if len(t_rank) > 0 else 0.0
            else:
                percentage = 0.0
            rank_score = max(f1, percentage)
            score = root_score + rank_score * 0.5
            infoi['score'] = f'{score}'
            infoi['meta'] = f'[RootCauseRank] TRUE: root={t_root}, rank={t_rank} | GEN: root={a_root}, rank={a_rank} | F1: {f1}, Percentage: {percentage}'
            count += 1
            total += score
        except Exception as e:
            infoi['score'] = '0.0'
            infoi['meta'] = f'[RootCauseRank] ERROR in parsing or scoring: {str(e)} when {ans=} and {a_root=} {a_rank=}'
            count += 1
    return total / count if count > 0 else 0
def multi_change_point_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答多条时间序列中是否存在变化点以及变化点的位置。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中每条时间序列（以ts_开头命名，如ts_1, ts_2等）是否存在变化点以及变化点的位置（如果存在，最多只有一个变化点）。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中包含若干键的dict，每个键（ts_1, ts_2等）对应一个dict，包含has_change_point（布尔值）和change_points（整数列表，可能为空）
        """
        json_template = """{ "ts_1": {"has_change_point": true/false, "change_points": [...]}, "ts_2": {"has_change_point": true/false, "change_points": [...]}, ... }//每个时序的has_change_point表示是否存在变化点，change_points表示变化点位置（最多一个或空列表）"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip())  # 动态解析所有ts_开头的键
        data = json_str
        return data

    def parse_target(target):
        return target

    def compute_match_with_tolerance(a_point, t_point, tol=5):
        if not a_point and not t_point:
            return True
        if len(a_point) == 1 and len(t_point) == 1:
            return abs(a_point[0] - t_point[0]) <= tol
        return False

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        a_data = parse_answer(ans)
        t_data = parse_target(tar)
        score = 0.0
        meta = []
        ts_keys = [key for key in t_data.keys() if key.startswith('ts_')]
        if not ts_keys:
            infoi['score'] = '0.0'
            infoi['meta'] = '[MultiChangePoints] No valid time series found'
            count += 1
            continue
        ts_weight = 1.0 / len(ts_keys)
        for ts in ts_keys:
            a_has = a_data.get(ts, {'has_change_point': False, 'change_points': []})['has_change_point']
            t_has = t_data[ts]['has_change_point']
            a_points = a_data.get(ts, {'has_change_point': False, 'change_points': []})['change_points']
            t_points = t_data[ts]['change_points']
            if a_has == t_has:
                if not t_has:
                    score += ts_weight  # one-shot for no change point
                    meta.append(f'{ts}: No change, correct')
                else:
                    if compute_match_with_tolerance(a_points, t_points, 5):
                        score += ts_weight
                        meta.append(f'{ts}: Change at {t_points}, correct within ±5')
                    else:
                        meta.append(f'{ts}: Change at {t_points}, incorrect (GEN: {a_points})')
            else:
                meta.append(f'{ts}: Incorrect has_change_point (TRUE: {t_has}, GEN: {a_has})')
        infoi['score'] = f'{score:.3f}'
        infoi['meta'] = f'[MultiChangePoints] TRUE: {t_data} | GEN: {a_data} | Details: {"; ".join(meta)}'
        count += 1
        total += score
    return total / count if count > 0 else 0
def multi_change_trend_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答多条时间序列的趋势识别，包括变点位置和每个片段的趋势描述。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中指出的变点位置（共享的整数列表，按从小到大排序），以及每个时间序列（以数字键命名，如"0", "1"等）的趋势列表（字符串列表，可用的词为"up", "stable", "down", "reverse_up", "reverse_down", "accelerate_up", "accelerate_down"等）。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中包含change_points键为整数列表，trends键为dict（内含数字键如"0", "1", "2",,,,,等，每个为字符串列表， 表示对应时序0,1,2,,, 中每个区间的具体趋势， 使用规定的英文单词）
        """
        json_template = """{ "change_points": [...], "trends": {"0": [...], "1": [...], ... } }//change_points为变点列表，trends为每个序列的趋势列表"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip())  # 动态解析
        data = json_str
        try :
            change_points = sorted(data['change_points'])
            trends = {k: [t.lower().strip() for t in data['trends'][k]] for k in data['trends']}
        except :
            change_points = []
            trends = {k: [t.lower().strip() for t in data['trends'][k]] for k in data['trends']}
        return change_points, trends

    def parse_target(target):
        change_points = sorted(target['change_points'])
        trends = {k: [t.lower().strip() for t in target[k]] for k in target if k != 'change_points'}
        return change_points, trends

    def compute_tp_with_tolerance(a, t, tol=5):
        if not a and not t:
            return 0
        i, j = 0, 0
        tp = 0
        while i < len(a) and j < len(t):
            if abs(a[i] - t[j]) <= tol:
                tp += 1
                i += 1
                j += 1
            elif a[i] < t[j]:
                i += 1
            else:
                j += 1
        return tp

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        try :
            a_points, a_trends = parse_answer(ans)
            t_points, t_trends = parse_target(tar)
            ts_keys = [k for k in t_trends.keys() if k.isdigit()]
            if not ts_keys:
                infoi['score'] = '0.0'
                infoi['meta'] = '[TrendRecognition] No valid time series found'
                count += 1
                continue
            one_shot_points = 0.0
            if len(a_points) == len(t_points) and all(a_points[i] == t_points[i] for i in range(len(a_points))):
                one_shot_points = 0.4
            tp = compute_tp_with_tolerance(a_points, t_points, 5)
            precision = tp / len(a_points) if len(a_points) > 0 else 0.0
            recall = tp / len(t_points) if len(t_points) > 0 else 0.0
            f1_points = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_points *= 0.4
            points_score = max(one_shot_points, f1_points)
            trend_score = 0.0
            num_ts = len(ts_keys)
            ts_weight = 0.6 / num_ts if num_ts > 0 else 0.0
            trend_meta = []
            for ts in ts_keys:
                a_tr = a_trends.get(ts, [])
                t_tr = t_trends[ts]
                if len(a_tr) == len(t_tr):
                    matches = sum(a == b for a, b in zip(a_tr, t_tr))
                    partial_score = (matches / len(t_tr)) * ts_weight if len(t_tr) > 0 else ts_weight
                    trend_score += partial_score
                    trend_meta.append(f'ts_{int(ts)+1}: {matches}/{len(t_tr)}')
                else:
                    trend_meta.append(f'ts_{int(ts)+1}: len mismatch ({len(a_tr)} vs {len(t_tr)})')
            score = points_score + trend_score
            infoi['score'] = f'{score:.3f}'
            infoi['meta'] = f'[TrendRecognition] TRUE points: {t_points}, trends: {t_trends} | GEN points: {a_points}, trends: {a_trends} | Points 1-shot: {one_shot_points}, F1: {f1_points/0.4 if f1_points > 0 else 0:.3f} | Trends: {"; ".join(trend_meta)}'
            count += 1
            total += score
        except :
            count += 1
    return total / count if count > 0 else 0
def multi_extreme_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答多条时间序列的最小值和最大值。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中每条时间序列（以ts_开头命名，如ts_1, ts_2等）的最小值和最大值。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中包含若干键的dict，每个键（ts_1, ts_2等）对应一个dict，包含min（浮点数）和max（浮点数）
        """
        json_template = """{ "ts_1": {"min": 浮点数, "max": 浮点数}, "ts_2": {"min": 浮点数, "max": 浮点数}, ... }//每个时序的min和max表示最小值和最大值"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip())  # 动态解析所有ts_开头的键
        data = json_str
        return data

    def parse_target(target):
        return target

    def compute_value_match(a_val, t_val, tol=1e-3):
        try :
            return abs(a_val - t_val) <= tol
        except :
            return False

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        a_data = parse_answer(ans)
        t_data = parse_target(tar)
        ts_keys = [key for key in t_data.keys() if key.startswith('ts_')]
        if not ts_keys:
            infoi['score'] = '0.0'
            infoi['meta'] = '[MinMaxMulti] No valid time series found'
            count += 1
            continue
        ts_weight = 1.0 / len(ts_keys) if ts_keys else 0.0
        value_weight = ts_weight / 2  # min和max各占一半权重
        score = 0.0
        meta = []
        for ts in ts_keys:
            a_vals = a_data.get(ts, {'min': 0.0, 'max': 0.0})
            t_vals = t_data[ts]
            min_match = compute_value_match(a_vals['min'], t_vals['min'], 3)
            max_match = compute_value_match(a_vals['max'], t_vals['max'], 3)
            if min_match:
                score += value_weight
                meta.append(f'{ts}: min correct ({a_vals["min"]})')
            else:
                meta.append(f'{ts}: min incorrect (TRUE: {t_vals["min"]}, GEN: {a_vals["min"]})')
            if max_match:
                score += value_weight
                meta.append(f'{ts}: max correct ({a_vals["max"]})')
            else:
                meta.append(f'{ts}: max incorrect (TRUE: {t_vals["max"]}, GEN: {a_vals["max"]})')
        infoi['score'] = f'{score:.3f}'
        infoi['meta'] = f'[MinMaxMulti] TRUE: {t_data} | GEN: {a_data} | Details: {"; ".join(meta)}'
        count += 1
        total += score
    return total / count if count > 0 else 0
def multi_index_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答多条时间序列中指定点的点值。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中每条时间序列（以ts_开头命名，如ts_1, ts_2等）在指定点的点值。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中包含若干键的dict，每个键（ts_1, ts_2等）对应一个浮点数，表示指定点的点值
        """
        json_template = """{ "ts_1": 浮点数, "ts_2": 浮点数, ... }//每个时序的键表示指定点的点值"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip())  # 动态解析所有ts_开头的键
        data = json_str
        return data

    def parse_target(target):
        return target

    def compute_value_match(a_val, t_val, tol=5.0):
        return abs(a_val - t_val) <= tol

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        a_data = parse_answer(ans)
        t_data = parse_target(tar)
        ts_keys = [key for key in t_data.keys() if key.startswith('ts_')]
        if not ts_keys:
            infoi['score'] = '0.0'
            infoi['meta'] = '[PointValueMulti] No valid time series found'
            count += 1
            continue
        ts_weight = 1.0 / len(ts_keys) if ts_keys else 0.0
        score = 0.0
        meta = []
        for ts in ts_keys:
            a_val = a_data.get(ts, 0.0)
            t_val = t_data[ts]
            if compute_value_match(a_val, t_val, 7.0):
                score += ts_weight
                meta.append(f'{ts}: value correct ({a_val})')
            else:
                meta.append(f'{ts}: value incorrect (TRUE: {t_val}, GEN: {a_val})')
        infoi['score'] = f'{score:.3f}'
        infoi['meta'] = f'[PointValueMulti] TRUE: {t_data} | GEN: {a_data} | Details: {"; ".join(meta)}'
        count += 1
        total += score
    return total / count if count > 0 else 0
def multi_spike_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答多条时间序列中是否存在突刺以及突刺的位置。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中每条时间序列（以ts_开头命名，如ts_1, ts_2等）是否存在突刺以及突刺的位置（如果存在，包含突刺点的下标）。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中包含若干键的dict，每个键（ts_1, ts_2等）对应一个dict，包含has_spike（布尔值）和spike_points（整数列表，可能为空）
        """
        json_template = """{ "ts_1": {"has_spike": true/false, "spike_points": [...]}, "ts_2": {"has_spike": true/false, "spike_points": [...]}, ... }//每个时序的has_spike表示是否存在突刺，spike_points表示突刺位置（列表，可能为空）"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip())  # 动态解析所有ts_开头的键
        data = json_str
        return data

    def parse_target(target):
        return target

    def compute_spike_match(a_points, t_points, tol=3):
        if not a_points and not t_points:
            return True
        if len(a_points) == len(t_points):
            return all(abs(a - t) <= tol for a, t in zip(sorted(a_points), sorted(t_points)))
        return False

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        try :
            a_data = parse_answer(ans)
            t_data = parse_target(tar)
            ts_keys = [key for key in t_data.keys() if key.startswith('ts_')]
            if not ts_keys:
                infoi['score'] = '0.0'
                infoi['meta'] = '[spikePointsMulti] No valid time series found'
                count += 1
                continue
            ts_weight = 1.0 / len(ts_keys) if ts_keys else 0.0
            score = 0.0
            meta = []
            for ts in ts_keys:
                a_has = a_data.get(ts, {'has_spike': False, 'spike_points': []})['has_spike']
                t_has = t_data[ts]['has_spike']
                a_points = a_data.get(ts, {'has_spike': False, 'spike_points': []})['spike_points']
                t_points = t_data[ts]['spike_points']
                if a_has == t_has:
                    if not t_has:
                        score += ts_weight  # one-shot for no spike
                        meta.append(f'{ts}: No spike, correct')
                    else:
                        if compute_spike_match(a_points, t_points, 3):
                            score += ts_weight
                            meta.append(f'{ts}: spike at {t_points}, correct within ±3')
                        else:
                            meta.append(f'{ts}: spike at {t_points}, incorrect (GEN: {a_points})')
                else:
                    meta.append(f'{ts}: Incorrect has_spike (TRUE: {t_has}, GEN: {a_has})')
            infoi['score'] = f'{score:.3f}'
            infoi['meta'] = f'[spikePointsMulti] TRUE: {t_data} | GEN: {a_data} | Details: {"; ".join(meta)}'
            count += 1
            total += score
        except Exception as e:
            infoi['meata'] = f'[ERROR] {e}'
            count += 1
            infoi['score'] = 0
    return total / count if count > 0 else 0
def multi_period_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答多条时间序列中是否存在季节性变化以及其周期长度。
            请你帮我提取出我想要的信息，这段文本是：{answer}。
            请提取出答案文本中每条时间序列（以ts_开头命名，如ts_1, ts_2等）是否存在季节性以及周期长度。

            我希望你的总结如下
            {json_template}

            我希望你的答案中有且仅有这段json，json中包含若干键的dict，
            每个键（ts_1, ts_2等）对应一个dict，包含：
            - has_season（布尔值）
            - period（整数或null）
            """
        json_template = """{
            "ts_1": {"has_season": true/false, "period": number or null},
            "ts_2": {"has_season": true/false, "period": number or null},
            ...
            } // has_season表示是否存在季节性，period表示周期长度（无则为null）"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
            "<|im_start|>user\n" + question + "<|im_end|>"
            "<|im_start|>assistant\n"
        )

        model_output = llm.generate(
            question, sampling_params, use_tqdm=False
        )[0].outputs[0].text

        data = parse_llm_json(model_output.strip())
        return data

    def parse_target(target):
        return target

    def compute_period_match(a_period, t_period, tol=2):
        if a_period is None or t_period is None:
            return False
        return abs(int(a_period) - int(t_period)) <= tol

    count = 0
    total = 0.0

    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        try:
            a_data = parse_answer(ans)
            t_data = parse_target(tar)

            ts_keys = [k for k in t_data.keys() if k.startswith("ts_")]
            if not ts_keys:
                infoi['score'] = '0.0'
                infoi['meta'] = '[seasonalPeriodMulti] No valid time series found'
                count += 1
                continue

            ts_weight = 1.0 / len(ts_keys)
            score = 0.0
            meta = []

            for ts in ts_keys:
                a_ts = a_data.get(ts, {'has_season': False, 'period': None})
                t_ts = t_data[ts]

                a_has = bool(a_ts.get('has_season', False))
                t_has = bool(t_ts.get('has_season', False))
                a_period = a_ts.get('period', None)
                t_period = t_ts.get('period', None)

                if a_has == t_has:
                    if not t_has:
                        score += ts_weight
                        meta.append(f'{ts}: no seasonality, correct')
                    else:
                        if compute_period_match(a_period, t_period, tol=2):
                            score += ts_weight
                            meta.append(
                                f'{ts}: season period {t_period}, correct within ±2'
                            )
                        else:
                            meta.append(
                                f'{ts}: season period {t_period}, incorrect (GEN: {a_period})'
                            )
                else:
                    meta.append(
                        f'{ts}: incorrect has_season (TRUE: {t_has}, GEN: {a_has})'
                    )

            infoi['score'] = f'{score:.3f}'
            infoi['meta'] = (
                f'[seasonalPeriodMulti] '
                f'TRUE: {t_data} | GEN: {a_data} | '
                f'Details: {"; ".join(meta)}'
            )

            count += 1
            total += score

        except Exception as e:
            infoi['meta'] = f'[ERROR] {e}'
            infoi['score'] = 0.0
            count += 1

    return total / count if count > 0 else 0.0

def multi_trend_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答多条时间序列的整体趋势。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中每条时间序列（以ts_开头命名，如ts_1, ts_2等）的整体趋势（字符串，值为"up", "down"或"stable"）。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中包含若干键的dict，每个键（ts_1, ts_2等）对应一个dict，包含trend（字符串，值为"up", "down"或"stable"）
        """
        json_template = """{ "ts_1": {"trend": "up/down/stable"}, "ts_2": {"trend": "up/down/stable"}, ... }//每个时序的trend表示整体趋势"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip())  # 动态解析所有ts_开头的键
        data = json_str
        for ts in data:
            data[ts]['trend'] = data[ts]['trend'].lower().strip()  # 规范化趋势值
        return data

    def parse_target(target):
        for ts in target:
            target[ts]['trend'] = target[ts]['trend'].lower().strip()  # 规范化趋势值
        return target

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        try :
            a_data = parse_answer(ans)
            t_data = parse_target(tar)
            ts_keys = [key for key in t_data.keys() if key.startswith('ts_')]
            if not ts_keys:
                infoi['score'] = '0.0'
                infoi['meta'] = '[TrendMulti] No valid time series found'
                count += 1
                continue
            ts_weight = 1.0 / len(ts_keys) if ts_keys else 0.0
            score = 0.0
            meta = []
            for ts in ts_keys:
                a_trend = a_data.get(ts, {'trend': ''})['trend']
                t_trend = t_data[ts]['trend']
                if a_trend == t_trend:
                    score += ts_weight
                    meta.append(f'{ts}: trend correct ({t_trend})')
                else:
                    meta.append(f'{ts}: trend incorrect (TRUE: {t_trend}, GEN: {a_trend})')
            infoi['score'] = f'{score:.3f}'
            infoi['meta'] = f'[TrendMulti] TRUE: {t_data} | GEN: {a_data} | Details: {"; ".join(meta)}'
            count += 1
            total += score
        except Exception as e:
            infoi['meata'] = f'[ERROR] {e}'
            count += 1
            infoi['score'] = 0            
    return total / count if count > 0 else 0
def uni_change_point_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答一条时间序列中是否存在变化点以及变化点的位置。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中是否存在变化点以及变化点的位置（可能包含多个变化点的下标，按从小到大排序）。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中包含两个键：has_change_point（布尔值）和change_points（整数列表，可能为空）
        """
        json_template = """{ "has_change_point": true/false, "change_points": [...] }//has_change_point表示是否存在变化点，change_points表示变化点位置（列表，可能为空）"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['has_change_point', 'change_points'])
        data = json_str
        has_change_point = data['has_change_point']
        change_points = sorted(data['change_points'])
        return has_change_point, change_points

    def parse_target(target):
        has_change_point = target['has_change_point']
        change_points = sorted(target['change_points'])
        return has_change_point, change_points

    def compute_change_point_match(a_points, t_points, tol=5):
        if not t_points and not a_points:
            return True
        if len(t_points) <= 1:  # 标答最多一个变化点
            for a_point in a_points:  # 依次考量答案中的变点
                if len(t_points) == 1 and abs(a_point - t_points[0]) <= tol:
                    return True
            return False
        return False

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        try:
            a_has, a_points = parse_answer(ans)
            t_has, t_points = parse_target(tar)
            score = 0.0
            if a_has == t_has:
                if not t_has:
                    score = 1.0  # one-shot for no change point
                else:
                    if compute_change_point_match(a_points, t_points, 5):
                        score = 1.0
            infoi['score'] = f'{score:.3f}'
            infoi['meta'] = f'[SingleChangePoint] TRUE: has={t_has}, points={t_points} | GEN: has={a_has}, points={a_points}'
            count += 1
            total += score
        except Exception as e:
            infoi['meata'] = f'[ERROR] {e}'
            count += 1
            infoi['score'] = 0
    return total / count if count > 0 else 0
def uni_extreme_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答一条时间序列的最小值和最大值。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中时间序列的最小值和最大值。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中包含两个键：min（浮点数）和max（浮点数）
        """
        json_template = """{ "min": 浮点数, "max": 浮点数 }//min和max表示时间序列的最小值和最大值"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['min', 'max'])
        data = json_str
        min_val = data['min']
        max_val = data['max']
        return min_val, max_val

    def parse_target(target):
        min_val = target['min']
        max_val = target['max']
        return min_val, max_val

    def compute_value_match(a_val, t_val, tol=1e-3):
        return abs(a_val - t_val) <= tol

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        try:
            a_min, a_max = parse_answer(ans)
            t_min, t_max = parse_target(tar)
            score = 0.0
            min_match = compute_value_match(a_min, t_min, 3)
            max_match = compute_value_match(a_max, t_max, 3)
            if min_match:
                score += 0.5
            if max_match:
                score += 0.5
            infoi['score'] = f'{score:.3f}'
            infoi['meta'] = f'[MinMaxSingle] TRUE: min={t_min}, max={t_max} | GEN: min={a_min}, max={a_max} | Details: min {"correct" if min_match else "incorrect"}, max {"correct" if max_match else "incorrect"}'
            count += 1
            total += score
        except Exception as e:
            infoi['meata'] = f'[ERROR] {e}'
            count += 1
            infoi['score'] = 0        
    return total / count if count > 0 else 0
def uni_index_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答一条时间序列中选定点的点值。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中时间序列选定点的点值。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中包含一个键：value（浮点数）
        """
        json_template = """{ "value": 浮点数 }//value表示时间序列选定点的点值"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['value'])
        data = json_str
        value = data['value']
        return value

    def parse_target(target):
        return target

    def compute_value_match(a_val, t_val, tol=1e-3):
        return abs(a_val - t_val) <= tol

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        try :
            a_value = parse_answer(ans)
            t_value = parse_target(tar)
            score = 1.0 if compute_value_match(a_value, t_value, 3) else 0.0
            infoi['score'] = f'{score:.3f}'
            infoi['meta'] = f'[SinglePointValue] TRUE: value={t_value} | GEN: value={a_value}'
            count += 1
            total += score
        except Exception as e:
            infoi['meata'] = f'[ERROR] {e}'
            count += 1
            infoi['score'] = 0            
    return total / count if count > 0 else 0
def uni_spike_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答一条时间序列中是否存在突刺以及突刺的位置。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中是否存在突刺以及突刺的位置（如果存在，包含突刺点的下标）。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中包含两个键：has_spike（布尔值）和spike_points（整数列表，可能为空）
        """
        json_template = """{ "has_spike": true/false, "spike_points": [...] }//has_spike表示是否存在突刺，spike_points表示突刺位置（列表，可能为空）"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['has_spike', 'spike_points'])
        data = json_str
        has_spike = data['has_spike']
        spike_points = sorted(data['spike_points'])
        return has_spike, spike_points

    def parse_target(target):
        has_spike = target['has_spike']
        spike_points = sorted(target['spike_points'])
        return has_spike, spike_points

    def compute_spike_match(a_points, t_points, tol=3):
        if not a_points and not t_points:
            return True
        if len(t_points) <= 1:  # 标答最多一个突刺点
            for a_point in a_points:  # 依次考量答案中的突刺点
                if len(t_points) == 1 and abs(a_point - t_points[0]) <= tol:
                    return True
            return False
        return False

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        try :
            a_has, a_points = parse_answer(ans)
            t_has, t_points = parse_target(tar)
            score = 0.0
            if a_has == t_has:
                if not t_has:
                    score = 1.0  # one-shot for no spike
                else:
                    if compute_spike_match(a_points, t_points, 3):
                        score = 1.0
            infoi['score'] = f'{score:.3f}'
            infoi['meta'] = f'[SinglespikePoint] TRUE: has={t_has}, points={t_points} | GEN: has={a_has}, points={a_points}'
            count += 1
            total += score
        except Exception as e:
            infoi['meata'] = f'[ERROR] {e}'
            count += 1
            infoi['score'] = 0
    return total / count if count > 0 else 0
def uni_period_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答一条时间序列中是否存在季节性变化以及其周期长度。
                    请你帮我提取出我想要的信息，这段文本是：{answer}。
                    请提取出答案文本中是否存在季节性（seasonal variation）以及周期长度（如果存在）。

                    我希望你的总结如下
                    {json_template}

                    我希望你的答案中有且仅有这段json，json中包含两个键：
                    - has_season（布尔值）
                    - period（整数或null）
                    """
        json_template = """{ "has_season": true/false, "period": number or null } // has_season表示是否存在季节性，period表示周期长度"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
            "<|im_start|>user\n" + question + "<|im_end|>"
            "<|im_start|>assistant\n"
        )

        model_output = llm.generate(
            question, sampling_params, use_tqdm=False
        )[0].outputs[0].text

        data = parse_llm_json(model_output.strip(), ['has_season', 'period'])

        has_season = bool(data['has_season'])
        period = data['period']
        if period is not None:
            period = int(period)

        return has_season, period

    def parse_target(target):
        has_season = bool(target['has_season'])
        period = target.get('period', None)
        if period is not None:
            period = int(period)
        return has_season, period

    def compute_period_match(a_period, t_period, tol=2):
        """
        判断周期是否匹配，允许一定误差
        """
        if a_period is None or t_period is None:
            return False
        return abs(a_period - t_period) <= tol

    count = 0
    total = 0.0

    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        try:
            a_has, a_period = parse_answer(ans)
            t_has, t_period = parse_target(tar)

            score = 0.0

            if a_has == t_has:
                if not t_has:
                    score = 1.0
                else:
                    if compute_period_match(a_period, t_period, tol=2):
                        score = 1.0

            infoi['score'] = f'{score:.3f}'
            infoi['meta'] = (
                f'[SeasonalPeriod] '
                f'TRUE: has={t_has}, period={t_period} | '
                f'GEN: has={a_has}, period={a_period}'
            )

            count += 1
            total += score

        except Exception as e:
            infoi['meta'] = f'[ERROR] {e}'
            infoi['score'] = 0.0
            count += 1

    return total / count if count > 0 else 0.0

def uni_trend_task(answer, target, info):
    def parse_answer(answer):
        prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答一条时间序列的整体趋势。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中时间序列的整体趋势（字符串，值为"up", "down"或"stable"）。
        我希望你的总结如下
        {json_template}
        我希望你的答案中有且仅有这段json，json中包含一个键：trend（字符串，值为"up", "down"或"stable"）
        """
        json_template = """{ "trend": "up/down/stable" }//trend表示时间序列的整体趋势"""
        question = prompt.format(answer=answer, json_template=json_template)
        question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
        model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
        json_str = parse_llm_json(model_output.strip(), ['trend'])
        data = json_str
        trend = data['trend'].lower().strip()  # 规范化趋势值
        return trend

    def parse_target(target):
        trend = target['trend'].lower().strip()  # 规范化趋势值
        return trend

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        a_trend = parse_answer(ans)
        t_trend = parse_target(tar)
        score = 1.0 if a_trend == t_trend else 0.0
        infoi['score'] = f'{score:.3f}'
        infoi['meta'] = f'[SingleTrend] TRUE: trend={t_trend} | GEN: trend={a_trend}'
        count += 1
        total += score
    return total / count if count > 0 else 0
def choices_task(answer, target, info):
    def parse_answer(answer):
        try :
            prompt = """你是一个文本分析大师，接下来我会给你一段文本，这段文本的目的是回答一个选择题。请你帮我提取出我想要的信息，这段文本是：{answer}。请提取出答案文本中选择的选项（小写字母a, b, c等）。
            我希望你的总结如下
            {json_template}
            我希望你的答案中有且仅有这段json，json中包含一个键：choice（字符串，小写字母a, b, c），如果没有识别出选项，请默认返回"a"
            """
            json_template = """{ "choice": "a/b/c" }//choice表示选择的选项（小写字母）"""
            question = prompt.format(answer=answer, json_template=json_template)
            question = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
            model_output = llm.generate(question, sampling_params, use_tqdm=False)[0].outputs[0].text
            json_str = parse_llm_json(model_output.strip(), ['choice'])
            data = json_str
            choice = data['choice'].lower().strip()  # 规范化选择值
            return choice
        except: 
            return 'a'

    def parse_target(target):
        target = target.strip()
        if target.startswith('(') and target.endswith(')') and len(target) == 3:
            return target[1].lower()
        return target.lower()

    count = 0
    total = 0
    for ans, tar, infoi in tqdm(zip(answer, target, info)):
        try :
            a_choice = parse_answer(ans)
            t_choice = parse_target(tar)
            score = 1.0 if a_choice == t_choice else 0.0
            infoi['score'] = f'{score:.3f}'
            infoi['meta'] = f'[Choices] TRUE: choice={t_choice} | GEN: choice={a_choice}'
            count += 1
            total += score
        except Exception as e:
            infoi['meata'] = f'[ERROR] {e}'
            count += 1
            infoi['score'] = 0
    return total / count if count > 0 else 0
def ift_task(answer, target, info):
    """
    answer: List[str]            # 模型生成的文本（含 A1/A2/...）
    target: List[List[dict]]     # 每个样本是一个子问题列表
    info:   List[dict]
    """

    evaluate_mapper = {
        'tms_uni_extreme_task': uni_extreme_task,
        'tms_uni_change_point_task': uni_change_point_task,
        'tms_uni_trend_task': uni_trend_task,
        'tms_multi_trend_task': multi_trend_task,
        'tms_uni_spike_task': uni_spike_task,
        'tms_multi_spike_task': multi_spike_task,
        'tms_uni_period_task': uni_period_task,
        'tms_multi_period_task': multi_period_task,
        'tms_comparison_task': double_check_task,
        'tms_segment_task': segment_task,
        'tms_relative_task': relative_trend,
    }

    count = 0
    total = 0.0

    for ans_text, tar_list, infoi in tqdm(zip(answer, target, info)):
        try:
            parsed_answers = parse_multi_answer_text(ans_text)

            sub_scores = []
            sub_meta = []

            for i, q in enumerate(tar_list, start=1):
                answer_i = parsed_answers.get(i, "")

                qtype = q["question_type"]
                if not qtype.startswith("tms_"):
                    qtype = "tms_uni_" + qtype

                if qtype not in evaluate_mapper:
                    sub_scores.append(0.0)
                    sub_meta.append(f'Q{i}: Unsupported task {qtype}')
                    continue

                eval_fn = evaluate_mapper[qtype]

                sub_answer = [answer_i]
                sub_target = [q["answer_meta"]]
                sub_info = [{}]

                s = eval_fn(sub_answer, sub_target, sub_info)
                sub_scores.append(s)

                sub_meta.append(
                    f'Q{i} ({qtype}): score={s:.3f}, meta={sub_info[0].get("meta", "")}'
                )

            final_score = sum(sub_scores) / len(sub_scores) if sub_scores else 0.0
            infoi['score'] = f'{final_score:.3f}'
            infoi['meta'] = '[IFT] ' + ' | '.join(sub_meta)

            total += final_score
            count += 1

        except Exception as e:
            infoi['meta'] = f'[ERROR] {e}'
            infoi['score'] = 0.0
            count += 1

    return total / count if count > 0 else 0.0

metrix_tools = {
    'double': double,
    'double_cnn':double,
    'double_enh':double,
    'multi_result':multi,
    'state':state,
    'state_sp':state,
    'base':base,
    'ts_index':ts_index,
    'stage1_base_global':stage1_base_acc,
    'stage1_base_local':stage1_base_acc,
    'stage2_calc_shape':stage2_calc_shape,
    'stage2_calc_trend': stage2_calc_trend,
    'stage2_calc_period': stage2_calc_period,
    'stage2_calc_point':  stage2_calc_point,
    'generated_change_trend_qa': generated_change_trend_qa,
    'multi_anomaly_detection': multi,
    'double_check_task': double_check_task,
    'relative_trend': relative_trend,
    'segment_task': segment_task,
    'multi_anomaly_rca_task': multi_anomaly_rca,
    'gen_double_check_different_length': double_check_task,
    'gen_multi_change_trend_task_different_length': multi_change_trend_task,
    'gen_relative_trend_different_length': relative_trend,
    'gen_segment_task_different_length': segment_task,
    'multi_anomaly_detection_different_length': multi,
    'multi_anomaly_rca_different_length': multi_anomaly_rca,
    'multi_change_point_task': multi_change_point_task,
    'multi_change_trend_task': multi_change_trend_task,
    'multi_extreme_task': multi_extreme_task,
    'multi_index_task': multi_index_task,
    'multi_spike_task': multi_spike_task,
    'multi_trend_task': multi_trend_task,
    'uni_change_point_task': uni_change_point_task,
    'uni_extreme_task': uni_extreme_task,
    'uni_index_task': uni_index_task,
    'uni_spike_task': uni_spike_task,
    'uni_trend_task':uni_trend_task,
    'TSQA_Trend':choices_task,
    'TSQA_Volatility':choices_task,
    'TSQA_Seasonality':choices_task,
    'TSQA_Outliers':choices_task,
    'ceval':choices_task,
    'describe_task':multi_change_trend_task,
    'multi_anomaly_prediction_task':multi,
    'tms_uni_extreme_task': uni_extreme_task,
    'tms_multi_extreme_task': multi_extreme_task,
    'tms_uni_change_point_task': uni_change_point_task,
    'tms_multi_change_point_task': multi_change_point_task,
    'tms_uni_trend_task': uni_trend_task,
    'tms_multi_trend_task': multi_trend_task,
    'tms_uni_spike_task': uni_spike_task,
    'tms_multi_spike_task': multi_spike_task,
    'tms_uni_period_task': uni_period_task,
    'tms_multi_period_task': multi_period_task,
    'tms_comparison_task': double_check_task,
    'tms_segment_task': segment_task,
    'tms_relative_task': relative_trend,
    'tms_anomaly_detection_task' : multi,
    'tms_root_cause_analysis_task': multi_anomaly_rca,
    'tms_describe_task': multi_change_trend_task,
    'tms_ift_task': ift_task,
}


import json
import argparse

model_name = "test"  
parser = argparse.ArgumentParser(description="Specify the evaluation dataset")

parser.add_argument('--evaluation_datasets', type=str, default='base', help='Evaluation dataset (default: base)')
parser.add_argument('--model_name', type=str, default='base', help='Evaluation dataset (default: base)')
parser.add_argument('--save_folder', type=str, default='base', help='Evaluation dataset (default: base)')
parser.add_argument('--model_qa_name', type=str, default='base', help='Evaluation dataset (default: base)')

args = parser.parse_args()

save_folder = args.save_folder

model_name = args.model_name
evaluation_datasets = [ args.evaluation_datasets ]

file_metric_pairs = [
    (f'{save_folder}/data/{model_name}_{dataset}.json', dataset, f'./datasets/{dataset}_chatts.json') 
    for dataset in evaluation_datasets
]

for filename, metric_key, targetname in file_metric_pairs:
    log_file = f'{save_folder}/result_{metric_key}.json'
    if os.path.exists(log_file):
        print(f'{log_file} 已存在')
        exit
        continue
    result_list = {}
    result_list['type'] = metric_key
    result_list['score'] = 0
    result_list['details'] = []
    answer = []
    target = []
    if not os.path.exists(targetname): 
        targetname = targetname.replace('datasets/result/', 'datasets/timeeval/')
    print(f'{filename} {targetname}')
    d2 = json.load(open(targetname, 'r', encoding='utf-8'))
    for item in d2:
        question = item['question']
        try :
            timeseries = item['timeseries']
        except :
            item['timeseries'] = []
            timeseries = []
        if timeseries != [] and type(timeseries[0]) != list: 
            timeseries = [timeseries]
        for ts in timeseries:
            ts_str = str(ts)
            item['question'] = item['question'].replace('<ts><ts/>', ts_str, 1)
    m_name = filename.split('/')[-1]
    if os.path.exists(filename) or 'chatts' in m_name.lower() or 'qwen' not in m_name.lower():
        
        llm = LLM(model=EVA_MODEL_PATH, trust_remote_code=True, max_model_len=ctx_length, tensor_parallel_size=gpu_per_model, gpu_memory_utilization=0.95)
        print(f'filename == {filename}')
        print(m_name)
        d1 = json.load(open(filename, 'r', encoding='utf-8'))
        if type(d1) != list:
            if 'outputs' in d1:
                d1 = d1['outputs']
            else :
                raise ValueError(f'cant resolve d1 type : {type(d1)}')
        answer = d1
    else :
        TRAIN_MODEL_PATH = args.model_qa_name
        llm = LLM(model=TRAIN_MODEL_PATH, trust_remote_code=True, max_model_len=ctx_length, tensor_parallel_size=gpu_per_model, gpu_memory_utilization=0.95)
        save_fder = f'{save_folder}/data'
        os.makedirs(save_fder, exist_ok=True)
        save_f = f'{save_folder}/data/{model_name}_{metric_key}.json'
        answer = []        
        for d in tqdm(d2, desc=f'generating'):
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
            answer.append(a)
        json.dump(answer, open(save_f, 'w'), ensure_ascii=False, indent=4)
        del llm
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        llm = LLM(model=EVA_MODEL_PATH, trust_remote_code=True, max_model_len=ctx_length, tensor_parallel_size=gpu_per_model, gpu_memory_utilization=0.95)
        model_max_length = llm.llm_engine.model_config.max_model_len
        model_max_length = min(ctx_length, model_max_length)
    target = [i['answer'] for i in d2]

    for index in range(len(d2)):
        i = d2[index]
        result_list['details'].append({
            'quesiton' : i['question'],
            'answer' : i['answer'],
            'output' : answer[index]
        })
    
    score = metrix_tools[metric_key](answer, target, result_list['details'])
    print(f"{metric_key}: {score}")
    result_list['score'] = score
    output_file = f'{save_folder}/result.txt'
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"{metric_key}: {score} \n")
    
    json.dump(result_list, open(log_file, 'w'), ensure_ascii=False, indent=4)


