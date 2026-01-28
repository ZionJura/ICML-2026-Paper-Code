
from volcenginesdkarkruntime import Ark
import os
api_key = os.getenv("VOLCENGINE_ARK_API_KEY")
client = Ark(
    api_key = api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

messages = [
    {"role": "user", "content": "北京今天的天气如何？"}
]
model_name = 'doubao-seed-1-6'
import matplotlib,random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json,os,random
def visual_func(length, func:list):
    x = np.arange(length)
    if len(func) > 10:
        print(f'func shape: {np.array(func).shape}')
    for i in func:
        y = []
        for t in range(length):
            try :
                y.append(i[t])
            except Exception as e:
                raise ValueError(f'{e}\nfunc element length error: {i=} {t=}')
        plt.plot(y, label=f'{func.index(i)+1}')
    plt.legend() 
    plt.show()

def get_model_response(prompt: str, model_name: str = 'doubao-seed-1-6') -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        thinking={
            "type": "disabled"
        },
        max_tokens=10240
    )
    
    return completion.choices[0].message.content


def gen_trend(length, trend_type='stable', change_type='random', start=None, end=None):
    if trend_type == 'random': 
        trend_type = random.choices(['stable', 'up', 'down'], [0.2, 0.4, 0.4])[0]
    if start is None:
        if trend_type == 'up':
            start = random.uniform(0.1, 0.4)
        elif trend_type == 'down':
            start = random.uniform(0.6, 0.9)
        else:
            start = random.uniform(0.3, 0.7)

    if end is None:
        if trend_type == 'up':
            end = random.uniform(0.6, 0.9)
        elif trend_type == 'down':
            end = random.uniform(0.1, 0.4)
        else:
            end = start
        
    if trend_type == 'stable':
        if start > end:
            start, end = end, start
    elif trend_type == 'up':
        if start >= end:
            end = start + random.uniform(0.1, 0.5)
            if end > 1:
                end = 1
    elif trend_type == 'down':
        if start <= end:
            end = start - random.uniform(0.1, 0.5)
            if end < 0:
                end = 0
    else:
        raise ValueError("trend_type must be in ['stable', 'up', 'down']")
    if change_type == 'random':
        change_type = random.choice(['linear', 'quadratic', 'exponential'])
    if trend_type == 'stable':
        change_type = 'none'
    x = np.arange(length)
    meta = {
        'trend_type': trend_type,
        'change_type': change_type,
        'change_type_zh': {'linear': '线性', 'quadratic': '二次', 'exponential': '指数', 'none': 'none'}[change_type],
        'text_zh' : "整体趋势呈" + {'linear': '线性', 'quadratic': '二次', 'exponential': '指数', 'none':''}[change_type] + {'stable': '平稳' , 'up': '上升', 'down': '下降'}[trend_type] + "趋势",
        'text': "The overall trend in this interval is " + {'linear': 'linear', 'quadratic': 'quadratic', 'exponential': 'exponential', 'none':''}[change_type] + {'stable': 'stable' , 'up': 'upward', 'down': 'downward'}[trend_type] + " trend",
        'start': start,
        'end': end
    }
    if change_type == 'linear':
        trend = np.linspace(start, end, length)
    elif change_type == 'quadratic':
        a = (end - start) / (length ** 2)
        meta['change_alpha'] = a
        trend = a * (x ** 2) + start
    elif change_type == 'exponential':
        if start <= 0 or end <= 0:
            start += 0.1
            end += 0.1
        a = (end / start) ** (1 / length)
        meta['change_alpha'] = a
        trend = start * (a ** x)
    else :
        trend = np.full(length, start)
    noise = np.random.normal(0, 0.005, length)  # 添加噪声
    if trend_type == 'stable':
        noise = np.zeros(length)
    trend = trend + noise
    return trend, meta



def gen_season(length, season_type='stable', period=None, amplitude=None):
    """
    - season_type in ['stable', 'square', 'sine']
    """
    if period is None:
        period_num = random.randint(3, 6)
        period = length // period_num
    if amplitude is None:
        amplitude = random.uniform(0.1, 0.5)
    x = np.arange(length)
    if season_type == 'random': 
        season_type = random.choice(['stable', 'square', 'sine'])
    if season_type == 'stable':
        season = np.zeros(length)
        text_zh = "无明显周期性变化"
        text = "No significant seasonal variation"
    elif season_type == 'square':
        season = amplitude * ((x % period) > (period / 2)).astype(float)
        text_zh = f"具有周期为{period}的方波周期特征"
        text = f"The seasonal variation is a square wave with a period of {period}"
    elif season_type == 'sine':
        season = amplitude * (np.sin(2 * np.pi * x / period)) / 2
        text_zh = f"具有周期为{period}的正弦波周期特征"
        text = f"The seasonal variation is a sine wave with a period of {period}"
    else:
        raise ValueError("season_type must be in ['stable', 'square', 'sine']")
    meta = {
        'season_type': season_type,
        'period': period,
        'amplitude': amplitude,
        'text_zh': text_zh,
        'text': text
    }
    return season, meta

def gen_segment(length, segment_type="none", start_point=0, ts_num=None):
    seg = np.zeros(length)
    seg_scale = np.random.uniform(0.2, 0.5)
    seg_length = int(np.ceil(length * seg_scale))
    start_idx = random.randint(0, length - seg_length - 1)
    end_idx = start_idx + seg_length
    if segment_type == 'random':
        segment_type = random.choice(['up', 'down', 'volatile', 'platform'])
    if segment_type == 'none':
        meta = {
            'segment_type': 'none',
            'text_zh': "无明显的分段变化",
            'text': "No significant segment variation"
        }
        return seg, meta
    if segment_type == 'up':
        seg[start_idx : end_idx] = np.linspace(0, random.uniform(0.7, 1), seg_length)
        seg[start_idx] = 0
        seg[end_idx:] = seg[end_idx-1]
        text_zh = f"在第 {start_idx+start_point} 个点到第 {end_idx+start_point} 个点内，数值保持上升"
        text = f"In the interval [{start_idx+start_point}, {end_idx+start_point}), the value shows an upward trend"
    elif segment_type == 'down':
        seg[start_idx : end_idx] = np.linspace(0, -random.uniform(0.7, 1), seg_length)
        seg[start_idx] = 0
        seg[end_idx :] = seg[end_idx-1]
        text_zh = f"在第 {start_idx+start_point} 个点到第 {end_idx+start_point}) 个点内，数值保持下降"
        text = f"In the interval [{start_idx+start_point}, {end_idx+start_point}), the value shows an downward trend"
    elif segment_type == 'volatile':
        seg[start_idx : end_idx] = np.random.uniform(-1, 1, seg_length)
        text_zh = f"在第 {start_idx+start_point} 个点到第 {end_idx+start_point} 个点内，数值呈剧烈波动"
        text = f"In the interval [{start_idx+start_point}, {end_idx+start_point}), the value shows fluctuations"
    elif segment_type == 'platform':
        level = random.uniform(-0.5, 0.5)
        seg[start_idx : end_idx] = level
        text_zh = f"在第 {start_idx+start_point} 个点到第 {end_idx+start_point} 个点内，数值呈平台状态，数值整体{'上升' if level>0 else '下降' if level<0 else ''}。"
        text = f"In the interval [{start_idx+start_point}, {end_idx+start_point}), the value shows a platform state, level {'rising' if level>0 else 'falling' if level<0 else ''}"
    else:
        raise ValueError("segment_type must be in ['none', 'up', 'down', 'volatile', 'platform']")
    meta = {
        'segment_type': segment_type,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'text_zh': text_zh,
        'text': text
    }
    return seg, meta

def gen_spike(length, spike_type="none", start_point=0, spike_num = 1):
    spike = np.zeros(length)
    if spike_type == 'none':
        meta = {
            'spike_type': 'none',
            'text_zh': "无明显的突发变化",
            'text': "No significant spike variation"
        }
        return spike, meta
    spike_indices = random.sample(range(length), spike_num)
    spike_indices = sorted(spike_indices)
    text_zh = ''
    text = ''
    if spike_type == 'random':
        spike_type = [random.choice(['up', 'down']) for _ in range(spike_num)]
    elif spike_type in ['up', 'down']:
        spike_type = [spike_type] * spike_num
    for idx in spike_indices:
        spike_type_i = spike_type[spike_indices.index(idx)]
        if spike_type_i == 'up':
            spike[idx] = random.uniform(0.5, 1)
            text_zh += f"在第 {idx+start_point} 个点出现突发上升。"
            text += f"At point {idx+start_point}, the value shows a sudden upward spike."
        elif spike_type_i == 'down':
            spike[idx] = -random.uniform(0.5, 1)
            text_zh += f"在第 {idx+start_point} 个点出现突发下降。"
            text += f"At point {idx+start_point}, the value shows a sudden downward spike."
        else:
            raise ValueError("spike_type must be in ['none', 'up', 'down']")    
    meta = {
        'spike_type': spike_type,
        'spike_indices': [idx + start_point for idx in spike_indices],
        'text_zh': "",
        'text': ""
    }
    meta['text_zh'] = text_zh if text_zh else "无明显的突发变化"
    meta['text'] = text if text else "No significant spike variation"
    return spike, meta

def visual_func(length, func:list):
    import matplotlib.pyplot as plt
    x = np.arange(length)
    if len(func) > 10:
        print(f'func shape: {np.array(func).shape}')
    for i in func:
        y = []
        for t in range(length):
            y.append(i[t])
        plt.plot(y, label=f'{func.index(i)+1}')
    plt.legend() 
    plt.show()




def gen_extreme_task(length=None, ts_num=1):
    output_text = ''
    meta = {}
    meta['text'] = ''
    if ts_num == 1:
        if length is None:
            length = random.choice([64, 128, 256, 336, 512])
        base, meta_base = gen_trend(length, 'random')
        meta['text'] += meta_base['text'] + '\n'
        season, meta_season = gen_season(length, 'random')
        meta['text'] += meta_base['text'] + '\n'
        segment, meta_segment = gen_segment(length, 'random')
        meta['text'] += meta_segment['text'] + '\n'
        spike, meta_spike = gen_spike(length, 'random', spike_num=random.randint(1, 6))
        meta['text'] += meta_spike['text'] + '\n'
        noise = np.random.normal(0, 0.05, length)
        ts_mean = np.random.uniform(-0.3, 0.3)
        ts_factor = np.random.uniform(1, 300)
        func = (ts_mean + base + season + segment + spike + noise) * ts_factor
        func = np.round(func, 3)
        input_text = f'Observe the following time series of length {length}: <ts><ts/>. Please identify the minimum and maximum values of this time series.'
        input_format = f'The answer format is: The minimum value of the time series is xxx, and the maximum value is xxx.'
        answer = {
            'min': float(np.min(func)),
            'max': float(np.max(func))
        }
        output_text = f'Based on the provided information, the minimum and maximum values are:\nThe minimum value of the time series is {answer["min"]:.2f}, and the maximum value is {answer["max"]:.2f}.'
        return {
            'question': input_text,
            'answer': answer,
            'input': input_text+input_format,
            'output': output_text,
            'timeseries': [func.tolist()],
        }
    else :
        if length is None:
            length = random.choice([64, 128, 256, 336, 512])
        if ts_num == -1:
            ts_num = random.randint(2, min(3000//length, 10))
        ts_list = []
        for i in range(ts_num):
            base, meta_base = gen_trend(length, 'random')
            meta['text'] += meta_base['text'] + '\n'
            season, meta_season = gen_season(length, 'random')
            meta['text'] += meta_season['text'] + '\n'
            segment, meta_segment = gen_segment(length, 'random')
            meta['text'] += meta_segment['text'] + '\n'
            spike, meta_spike = gen_spike(length, 'random', spike_num=random.randint(1, 6))
            meta['text'] += meta_spike['text'] + '\n'
            noise = np.random.normal(0, 0.05, length)
            ts_mean = np.random.uniform(-0.3, 0.3)
            ts_factor = np.random.uniform(1, 300)
            func = (ts_mean + base + season + segment + spike + noise) * ts_factor
            func = np.round(func, 3)
            ts_list.append(func)
        answer = {}
        input_text = f'Observe the following multiple time series, where '
        input_text += ', '.join([f'Time series {i+1} is a time series of length {length}: <ts><ts/>;' for i in range(ts_num)]) + ' Please identify the minimum and maximum values for each time series in order.'
        input_format = f'The answer format is: The minimum value of time series i is xxx, and the maximum value is xxx.'
        output_text = f'Based on the provided information, the minimum and maximum values are as follows:\n'
        for t in range(ts_num):
            answer[f'ts_{t+1}'] = {
                'min': float(np.min(ts_list[t])),
                'max': float(np.max(ts_list[t]))
            }
            output_text += f'时间序列{t+1}的最小值为 {answer[f"ts_{t+1}"]["min"]:.2f}，最大值为 {answer[f"ts_{t+1}"]["max"]:.2f}。\n'
        return {
            'question': input_text,
            'answer': answer,
            'timeseries': [ts.tolist() for ts in ts_list],
            'input': input_text+input_format,
            'output': output_text,
            'meta':meta['text']
        }

def gen_change_point(length=None, ts_num=1, ts_change_num=1):
    meta = {'text': ''}
    def gen_up_down_trend(length, trend, start_value=0, prev_alpha=None): # 生成线性上升或下降，返回斜率
        alpha = None
        cnt = 0.1 * length
        while alpha is None or (prev_alpha is not None and abs(alpha - prev_alpha) < 0.5):
            if trend == 'up':
                end_value = start_value + random.uniform(cnt, 0.5 + cnt)
            else:
                end_value = start_value - random.uniform(cnt, 0.5 + cnt)
            cnt += 0.5
            alpha = (end_value - start_value) / length
        trend = np.linspace(start_value, end_value, length)
        return trend, alpha

    def gen_oscillate_trend(length, start_value=0): # 生成周期振荡，返回周期和振幅
        period = random.randint(1, length // 2)
        amplitude = random.uniform(0.3, 0.7)
        x = np.arange(length)
        trend = start_value + amplitude * np.sin(2 * np.pi * x / period)
        return trend, period, amplitude

    def gen_platform_trend(length, trend, start_value=0): # 整个段是一个整体的平台，返回平台高度
        if trend == 'platform_up':
            level = start_value + random.uniform(0.5, 1)
        else:
            level = start_value - random.uniform(0.5, 1)
        trend = np.ones(length) * level
        trend[-1] = start_value
        return trend, level

    def norm_ts(ts):
        ts_min = ts.min()
        ts_max = ts.max()
        if ts_max - ts_min < 1e-6:
            return ts - ts_min
        return (ts - ts_min) / (ts_max - ts_min)

    def reconstruct_ts(ts, ts_factor=None):
        ts = norm_ts(ts)
        ts_mean = random.uniform(-0.3, 0.3)
        ts_noise = np.random.normal(0, 0.01, len(ts))
        if ts_factor is None:
            ts_factor = random.uniform(1, 300)
        ts = (ts_mean + ts + ts_noise) * ts_factor
        return ts
    
    if length is None:
        length = random.choice([64, 128, 256])
        if random.randint(1, 10) > 7:
            length = random.randint(64, 336)
    if ts_num == -1:
        ts_num = random.randint(2, min(3000//length, 5))
    if ts_change_num is None:
        ts_change_num = random.randint(0, 1)

    trend_types = ['up', 'down', 'stable', 'oscillate', 'platform_up', 'platform_down']
    weights = [0.3, 0.3, 0.3, 0.1, 0., 0.]
    trend_desc = {
        'up': '上升',
        'down': '下降',
        'stable': '平稳',
        'oscillate': '振荡',
        'platform_up': '平台上升',
        'platform_down': '平台下降'
    }

    if ts_num == 1:
        change_idx_list = []
        if ts_change_num > 0:
            segment_len = length // (ts_change_num + 1)
            for i in range(ts_change_num):
                change_idx = random.randint(i * segment_len + segment_len // 2, (i + 1) * segment_len + segment_len // 2 - 1)
                change_idx_list.append(change_idx)
            change_idx_list = sorted(change_idx_list)
        segment_trend_list = []
        start_idx = 0
        prev_trend = None
        prev_alpha = None
        ts = np.array([])
        for change_idx in change_idx_list + [length]:
            segment_len = change_idx - start_idx
            trend_type = random.choices(trend_types, weights)[0]
            if prev_trend is not None:
                while trend_type == prev_trend:
                    trend_type = random.choices(trend_types, weights)[0]
            prev_trend = trend_type
            segment_trend_list.append((change_idx, trend_type))
            if trend_type in ['up', 'down']:
                segment, alpha = gen_up_down_trend(segment_len, trend_type, start_value=ts[-1] if len(ts) > 0 else 0, prev_alpha=prev_alpha)
                prev_alpha = alpha
            elif trend_type == 'stable':
                segment = np.ones(segment_len) * (ts[-1] if len(ts) > 0 else 0)
                prev_alpha = None
            elif trend_type == 'oscillate':
                segment, period, amplitude = gen_oscillate_trend(segment_len, start_value=ts[-1] if len(ts) > 0 else 0)
                prev_alpha = None
            elif trend_type in ['platform_up', 'platform_down']:
                segment, level = gen_platform_trend(segment_len, trend_type, start_value=ts[-1] if len(ts) > 0 else 0)
                prev_alpha = None
            ts = np.concatenate([ts, segment])
            start_idx = change_idx
        ts = reconstruct_ts(ts)
        ts = np.round(ts, 3)
        input_text = f'Observe the following time series of length {length}: <ts><ts/>. Please identify whether there are any change points in this time series. If so, at which points do the change points occur?'
        input_format = f'The answer format is: The time series has k trend change points, which are at points i1, i2, ..., ik. / The time series has a consistent overall trend, with no trend change points. There is no need to consider specific trend types. Please only answer the format content.'
        output_text = f'The time series has {len(change_idx_list)} trend change point(s), which are at point(s) {", ".join(str(idx + 1) for idx in change_idx_list)}.'  if len(change_idx_list) > 0 else 'The time series has a consistent overall trend, with no trend change points.'
        answer = {
            'has_change_point': len(change_idx_list) > 0,
            "change_points": [idx + 1 for idx in change_idx_list]  # 从1开始计数
        }
        meta['text'] = f'Each segment of the time series has a trend of {", ".join(trend for _, trend in segment_trend_list)}.'
        return {
            'question': input_text,
            'answer': answer,
            'timeseries': [ts.tolist()],
            'input': input_text + input_format,
            'output': output_text,
            'meta': meta['text']
        }
    else:
        ts_list = []
        answer = {}
        output_text = ''
        input_text = f'Observe the following multiple time series, where '
        input_text += ', '.join([f'Time series {i+1} is a time series of length {length}: <ts><ts/>;' for i in range(ts_num)]) + f' Please identify for each time series whether there are any change points. If so, at which points do the change points occur?'
        input_format = f'The answer format is: For time series i, there are k trend change points, which are at points i1, i2, ..., ik. / The time series has a consistent overall trend, with no trend change points. There is no need to consider specific trend types. Please only answer the format content.'
        multi_trend_list = []
        for t in range(ts_num):
            change_idx_list = []
            now_change_num = random.choices([0, 1], [0.4, 0.6])[0]
            if now_change_num > 0:
                segment_len = length // (now_change_num + 1)
                change_idx = random.randint(segment_len // 2, segment_len + segment_len // 2 - 1)
                change_idx_list.append(change_idx)
            segment_trend_list = []
            start_idx = 0
            prev_trend = None
            prev_alpha = None
            ts = np.array([])
            for change_idx in change_idx_list + [length]:
                segment_len = change_idx - start_idx
                trend_type = random.choices(trend_types, weights)[0]
                if prev_trend is not None:
                    while trend_type == prev_trend:
                        trend_type = random.choices(trend_types, weights)[0]
                prev_trend = trend_type
                segment_trend_list.append((change_idx, trend_type))
                if trend_type in ['up', 'down']:
                    segment, alpha = gen_up_down_trend(segment_len, trend_type, start_value=ts[-1] if len(ts) > 0 else 0, prev_alpha=prev_alpha)
                    prev_alpha = alpha
                elif trend_type == 'stable':
                    segment = np.ones(segment_len) * (ts[-1] if len(ts) > 0 else 0)
                    prev_alpha = None
                elif trend_type == 'oscillate':
                    segment, period, amplitude = gen_oscillate_trend(segment_len, start_value=ts[-1] if len(ts) > 0 else 0)
                    prev_alpha = None
                elif trend_type in ['platform_up', 'platform_down']:
                    segment, level = gen_platform_trend(segment_len, trend_type, start_value=ts[-1] if len(ts) > 0 else 0)
                    prev_alpha = None
                ts = np.concatenate([ts, segment])
                start_idx = change_idx
            ts = reconstruct_ts(ts)
            ts = np.round(ts, 3)
            ts_list.append(ts.tolist())
            if change_idx_list:
                answer[f'ts_{t+1}'] = {
                    'has_change_point': True,
                    'change_points': [idx + 1 for idx in change_idx_list],
                    'trend_prev': trend_desc[segment_trend_list[0][1]],
                    'trend_next': trend_desc[segment_trend_list[1][1]]
                }
                meta['text'] += f'Time series {t+1} has a trend of {", ".join(trend for _, trend in segment_trend_list)}.\n'
                output_text += f'Time series {t+1} has {len(change_idx_list)} trend change point(s), which are at point(s) {", ".join(str(idx + 1) for idx in change_idx_list)}.\n'
            else:
                answer[f'ts_{t+1}'] = {
                    'has_change_point': False,
                    'change_points': []
                }
                output_text += f'Time series {t+1} has a consistent overall trend, with no trend change points.\n'
                meta['text'] += f'Time series {t+1} has a consistent overall trend {trend_desc[segment_trend_list[0][1]]}, with no trend change points.\n'



        return {
            'question': input_text,
            'answer': answer,
            'timeseries': ts_list,
            'input': input_text + input_format,
            'output': output_text,
            'meta': meta['text']
        }


def gen_trend_task(length=None, ts_num=1):
    zh_trend = {
        'up' : '上升',
        'down' : '下降',
        'stable' : '平稳'
    }
    if ts_num == 1:
        if length is None:
            length = random.choice([64, 128, 256, 336, 512])
        ts_factor = np.random.uniform(1, 300) 
        trend_type = random.choices(['up', 'down', 'stable'], [0.4, 0.4, 0.2])[0]
        base, meta_base = gen_trend(length, trend_type=trend_type)
        noise = np.random.normal(0, 0.005, length)
        
        ts_mean = np.random.uniform(-0.3, 0.3)
        ts = ts_mean + base + noise
        func = ts * ts_factor
        func = np.round(func, 3)
        input_text = f'Observe the following time series of length {length}: <ts><ts/>. Please identify the overall trend of this time series: is it upward, downward, or stable?'
        input_format = f'The answer format is: The overall trend of the time series is ___. (upward, downward, stable)'
        output_text = f'The overall trend of the time series is {trend_type}.'
        answer = {
            "trend": trend_type
        }
        return {
            'question': input_text,
            'answer': answer,
            'input': input_text+input_format,
            'output': output_text,
            'timeseries': [func.tolist()],
        }
    else :
        if length is None:
            length = random.choice([64, 128, 256, 336, 512])
        if ts_num == -1:
            ts_num = random.randint(2, min(3000//length, 10))
        ts_factor = np.random.uniform(1, 300)
        ts_list = []
        trend_list = []
        for i in range(ts_num):
            trend_type = random.choices(['up', 'down', 'stable'], [0.4, 0.4, 0.2])[0]
            base, meta_base = gen_trend(length, trend_type=trend_type)
            noise = np.random.normal(0, 0.05, length)
            ts_mean = np.random.uniform(-0.3, 0.3)
            func = (ts_mean + base + noise) * ts_factor
            func = np.round(func, 3)
            ts_list.append(func)
            trend_list.append(trend_type)
        input_text = f'Observe the following multiple time series, where '
        input_text += ', '.join([f'Time series {i+1} is a time series of length {length}: <ts><ts/>;' for i in range(ts_num)]) + f' Please identify for each time series the overall trend: is it upward, downward, or stable?'
        input_format = f'The answer format is: For time series i, the overall trend is ___. (upward, downward, stable)'
        output_text = ''
        answer = {}
        for t in range(ts_num):
            answer[f'ts_{t+1}'] = {
                "trend": trend_list[t]
            }
            output_text += f'Time series {t+1} overall trend is {trend_list[t]}.\n'
        return {
            'question': input_text,
            'answer': answer,
            'timeseries': [ts.tolist() for ts in ts_list],
            'input': input_text+input_format,
            'output': output_text
        }


def gen_period_task(length=None, ts_num=1):
    if ts_num == 1:
        if length is None:
            length = random.choice([64, 128, 256, 336, 512])
        has_season = random.choices([True, False], [0.6, 0.4])[0]
        if has_season:
            season_type = random.choice(['square', 'sine'])
            season, meta_season = gen_season(length, season_type=season_type)
            period = meta_season['period']
        else:
            season, meta_season = gen_season(length, season_type='stable')
            period = None
        base, meta_base = gen_trend(length, 'random')
        noise = np.random.normal(0, 0.05, length)
        ts_mean = np.random.uniform(-0.3, 0.3)
        ts_factor = np.random.uniform(1, 300)
        func = (ts_mean + base + season + noise) * ts_factor
        func = np.round(func, 3)
        input_text = f'Observe the following time series of length {length}: <ts><ts/>. Please identify whether there is any seasonal variation in this time series. If so, what is the period of the seasonal variation?'
        input_format = f'The answer format is: The time series has seasonal variation with a period of xxx. / The time series has no seasonal variation.'
        if has_season:
            output_text = f'The time series has seasonal variation with a period of {period}.'
        else:
            output_text = f'The time series has no seasonal variation.'
        answer = {
            'has_season': has_season,
            'period': period
        }
        return {
            'question': input_text,
            'answer': answer,
            'input': input_text+input_format,
            'output': output_text,
            'timeseries': [func.tolist()],
        }
    else :
        if length is None:
            length = random.choice([64, 128, 256, 336, 512])
        if ts_num == -1:
            ts_num = random.randint(2, min(3000//length, 10))
        ts_list = []
        answer = {}
        input_text = f'Observe the following multiple time series, where '
        input_text += ', '.join([f'Time series {i+1} is a time series of length {length}: <ts><ts/>;' for i in range(ts_num)]) + f' Please identify for each time series whether there is any seasonal variation. If so, what is the period of the seasonal variation?'
        input_format = f'The answer format is: For time series i, the time series has seasonal variation with a period of xxx. / The time series has no seasonal variation.'
        output_text = ''
        for t in range(ts_num):
            has_season = random.choices([True, False], [0.6, 0.4])[0]
            if has_season:
                season_type = random.choice(['square', 'sine'])
                season, meta_season = gen_season(length, season_type=season_type)
                period = meta_season['period']
            else:
                season, meta_season = gen_season(length, season_type='stable')
                period = None
            base, meta_base = gen_trend(length, 'random')
            noise = np.random.normal(0, 0.05, length)
            ts_mean = np.random.uniform(-0.3, 0.3)
            ts_factor = np.random.uniform(1, 300)
            func = (ts_mean + base + season  + noise) * ts_factor
            func = np.round(func, 3)
            ts_list.append(func)
            if has_season:
                answer[f'ts_{t+1}'] = {
                    'has_season': True,
                    'period': period
                }
                output_text += f'Time series {t+1} has seasonal variation with a period of {period}.\n'
            else:
                answer[f'ts_{t+1}'] = {
                    'has_season': False,
                    'period': None
                }
                output_text += f'Time series {t+1} has no seasonal variation.\n'
        return {
            'question': input_text,
            'answer': answer,
            'timeseries': [ts.tolist() for ts in ts_list],
            'input': input_text+input_format,   
            'output': output_text
        }

def gen_double_check(length=None):
    if length is None:
        length = random.choice([64, 128, 256, 336, 512])
    ts_factor = np.random.uniform(1, 300)
    base1, meta_base1 = gen_trend(length, 'random')
    base2 = base1.copy()
    season1, meta_season1 = gen_season(length, 'random')
    season2 = season1.copy()
    segment1, meta_segment1 = gen_segment(length, 'random')
    segment2 = segment1.copy()
    spike1, meta_spike1 = gen_spike(length, 'random', spike_num=random.randint(0, 3))
    spike2 = spike1.copy()
    noise1 = np.random.normal(0, 0.01, length)
    noise2 = np.random.normal(0, 0.01, length)
    ts_mean1 = np.random.uniform(-0.3, 0.3)
    ts_mean2 = ts_mean1
    func1 = (ts_mean1 + base1 + season1 + segment1 + spike1 + noise1) * ts_factor
    func2 = (ts_mean2 + base2 + season2 + segment2 + spike2 + noise2) * ts_factor
    func1 = np.round(func1, 3)
    func2 = np.round(func2, 3)
    has_difference = random.choices([True, False], [0.7, 0.3])[0]
    if has_difference:
        change_length = random.randint(max(3, length // 5), max(length // 4, 5))
        change_start = random.randint(0, length - change_length - 1)
        change_end = change_start + change_length - 1
        change_type = random.choice(['segment'])
        if change_type == 'trend':
            trend_type = random.choices(['up', 'down', 'stable'], [0.4, 0.4, 0.2])[0]
            new_trend, meta_new_trend = gen_trend(change_end - change_start + 1, trend_type=trend_type)
            base2[change_start:change_end + 1] = new_trend
        elif change_type == 'season':
            new_season, meta_new_season = gen_season(change_end - change_start + 1, 'random')
            season2[change_start:change_end + 1] = new_season
        elif change_type == 'segment':
            new_segment, meta_new_segment = gen_segment(change_end - change_start + 1, 'random')
            segment2[change_start:change_end + 1] = new_segment
        elif change_type == 'spike':
            new_spike, meta_new_spike = gen_spike(change_end - change_start + 1, spike_type='random', spike_num=1)
            spike2[change_start:change_end + 1] = new_spike
        func2 = (ts_mean2 + base2 + season2 + segment2 + spike2 + noise2) * ts_factor
        func2 = np.round(func2, 3)
        input_text = f'Observe the following two time series of length {length}: Time series A is <ts><ts/>, and time series B is <ts><ts/>. Please answer the following question: Does time series B have any significant differences compared to time series A? If so, please specify the interval where the differences occur (from which point to which point).'
        input_format = f'The answer format is: Time series B has significant differences compared to time series A, with the differences occurring from point k to point m. / Time series B does not have any significant differences compared to time series A.'
        output_text = f'Time series B has significant differences compared to time series A, with the differences occurring from point {change_start + 1} to point {change_end + 1}.'
        answer = {
            "has_difference": True,
            "differences_interval": [change_start + 1, change_end + 1]  # 从1开始计数
        }
    else:
        input_text = f'Observe the following two time series of length {length}: Time series A is <ts><ts/>, and time series B is <ts><ts/>. Please answer the following question: Does time series B have any significant differences compared to time series A? If so, please specify the interval where the differences occur (from which point to which point).'
        input_format = f'The answer format is: Time series B has significant differences compared to time series A, with the differences occurring from point k to point m. / Time series B does not have any significant differences compared to time series A.'
        output_text = f'Time series B does not have any significant differences compared to time series A.'
        answer = {
            "has_difference": False,
            "differences_interval": []
        }
    return {
        'question': input_text,
        'answer': answer,
        'input': input_text + input_format,
        'output': output_text,
        'timeseries': [func1.tolist(), func2.tolist()],
    }


def gen_segment_task(length=None, ts_num=None):
    if length is None or length // 24 < 1:
        length = random.choice([64, 128, 256])
    ts_change_num = random.randint(1, min(4, length // 24))
    if length // 24 >= 3 and random.randint(0, 100) > 50:
        ts_change_num = random.randint(3, min(5, length // 24))
    change_idx_list = []
    if ts_change_num > 0:
        segment_len = length // (ts_change_num + 1)
        for i in range(ts_change_num):
            change_idx = random.randint(i * segment_len + segment_len // 2, (i + 1) * segment_len + segment_len // 2 - 1)
            change_idx_list.append(change_idx)
        change_idx_list = sorted(change_idx_list)
        
    segment_trend_list = []
    trend_types = ['up', 'down', 'stable', 'oscillate', 'platform_up', 'platform_down']
    weights = [0.35, 0.35, 0.3, 0., 0., 0.]
    trend_desc = {
        'up': '上升',
        'down': '下降',
        'stable': '平稳',
        'oscillate': '振荡',
        'platform_up': '平台上升',
        'platform_down': '平台下降'
    }
    def gen_up_down_trend(length, trend, start_value=0, prev_alpha=None): 
        alpha = None
        cnt = 0.1 * length
        while alpha is None or (prev_alpha is not None and abs(alpha - prev_alpha) < 0.5):
            if trend == 'up':
                end_value = start_value + random.uniform(cnt, 0.5+cnt)
            else :
                end_value = start_value - random.uniform(cnt, 0.5+cnt)
            cnt += 0.5
            alpha = (end_value - start_value) / length
        trend = np.linspace(start_value, end_value, length)
        return trend, alpha
    def gen_oscillate_trend(length, start_value=0): 
        period = random.randint(1, length // 2)
        amplitude = random.uniform(0.3, 0.7)
        x = np.arange(length)
        trend = start_value + amplitude * np.sin(2 * np.pi * x / period)
        return trend, period, amplitude
    def gen_platform_trend(length, trend, start_value=0):
        if trend == 'platform_up':
            level = start_value + random.uniform(0.5, 1)
        else :
            level = start_value - random.uniform(0.5, 1)
        trend = np.ones(length) * level
        trend[-1] = start_value
        return trend, level
    def norm_ts(ts):
        ts_min = ts.min()
        ts_max = ts.max()
        if ts_max - ts_min < 1e-6:
            return ts - ts_min
        return (ts - ts_min) / (ts_max - ts_min)
    def reconstruct_ts(ts, ts_factor=None):
        ts = norm_ts(ts)
        ts_mean = random.uniform(-0.3, 0.3)
        ts_noise = np.random.normal(0, 0.01, len(ts))
        if ts_factor is None:
            ts_factor = random.uniform(1, 300)
        ts = (ts_mean + ts + ts_noise) * ts_factor
        return ts
    start_idx = 0
    prev_trend = None
    prev_alpha = None
    ts = np.array([])
    for change_idx in change_idx_list + [length]:
        segment_len = change_idx - start_idx
        trend_type = random.choices(trend_types, weights)[0]
        if prev_trend is not None:
            while trend_type == prev_trend:
                trend_type = random.choices(trend_types, weights)[0]
        prev_trend = trend_type
        segment_trend_list.append((change_idx, trend_type))
        if trend_type in ['up', 'down']:
            segment, alpha = gen_up_down_trend(segment_len, trend_type, start_value=ts[-1] if len(ts) > 0 else 0, prev_alpha=prev_alpha)
            prev_alpha = alpha
        elif trend_type == 'stable':
            segment = np.ones(segment_len) * (ts[-1] if len(ts) > 0 else 0)
            prev_alpha = None
        elif trend_type == 'oscillate':
            segment, period, amplitude = gen_oscillate_trend(segment_len, start_value=ts[-1] if len(ts) > 0 else 0)
            prev_alpha = None
        elif trend_type in ['platform_up', 'platform_down']:
            segment, level = gen_platform_trend(segment_len, trend_type, start_value=ts[-1] if len(ts) > 0 else 0)
            prev_alpha = None
        ts = np.concatenate([ts, segment])
        start_idx = change_idx
    ts = reconstruct_ts(ts)
    ts = np.round(ts, 3)
    input_text = f'Observe the following time series of length {length}: <ts><ts/>. Please identify the points at which trend change points occur in this time series. Please list all change points in order (if there are multiple change points, please list them in ascending order).'
    input_format = f'The answer format is: The time series has k trend change points, which are at points i1, i2, ..., ik. / The time series has a consistent overall trend, with no trend change points. There is no need to consider specific trend types. Please only answer the format content.'
    output_text = f'The time series has {len(change_idx_list)} trend change point(s), which are at point(s) {", ".join(str(idx + 1) for idx in change_idx_list)}.'  if len(change_idx_list) > 0 else 'The time series has a consistent overall trend, with no trend change points.'
    if len(change_idx_list) > 0:
        answer = {
            "change_points": [idx + 1 for idx in change_idx_list]  # 从1开始计数
        }
    else:
        answer = {
            "change_points": []
        }
    return {
        'question': input_text,
        'answer': answer,
        'timeseries': [ts.tolist()],
        'input': input_text + input_format,
        'output': output_text,
        'meta': {
            'segment_trend_list': [(idx + 1, trend_desc[trend]) for idx, trend in segment_trend_list]
        }
    }



def gen_relative_trend(length=None, ts_num=None):
    if length is None:
        length = random.choice([64, 128, 256, 336, 512])
    change_idx = random.randint(length // 4, 3 * length // 4)
    trend_type1 = random.choices(['up', 'down', 'stable'], [0.4, 0.4, 0.2])[0]
    if trend_type1 == 'up':
        trend_type2 = random.choices(['up', 'down', 'stable'], [0.6, 0.1, 0.3])[0]
    elif trend_type1 == 'down':
        trend_type2 = random.choices(['up', 'down', 'stable'], [0.1, 0.6, 0.3])[0]
    else:
        trend_type2 = random.choices(['up', 'down'], [0.5, 0.5])[0]
    def gen_up_down_trend(length, trend, start_value=0, prev_alpha=0): # 生成线性上升或下降，返回斜率
        alpha = None
        cnt = 0.5
        while alpha is None or abs(alpha - prev_alpha) < 0.3:
            if trend == 'up':
                end_value = start_value + random.uniform(cnt, 0.5+cnt)
            else :
                end_value = start_value - random.uniform(cnt, 0.5+cnt)
            cnt += 0.5
            alpha = (end_value - start_value) / length
        trend = np.linspace(start_value, end_value, length)
        return trend, alpha
    def gen_stable_trend(length, start_value=0): # 生成平稳
        trend = np.ones(length) * start_value
        return trend
    def norm_ts(ts):
        ts_min = ts.min()
        ts_max = ts.max()
        if ts_max - ts_min < 1e-6:
            return ts - ts_min
        return (ts - ts_min) / (ts_max - ts_min)
    def reconstruct_ts(ts, ts_factor=None):
        ts = norm_ts(ts)
        ts_mean = random.uniform(-0.3, 0.3)
        ts_noise = np.random.normal(0, 0.01, len(ts))
        if ts_factor is None:
            ts_factor = random.uniform(1, 100)
        ts = (ts_mean + ts + ts_noise) * ts_factor
        return ts
    if trend_type1 in ['up', 'down']:
        segment1, alpha1 = gen_up_down_trend(change_idx, trend_type1, start_value=0)
    else:
        segment1 = gen_stable_trend(change_idx, start_value=0)
        alpha1 = 0
    if trend_type2 in ['up', 'down']:
        segment2, alpha2 = gen_up_down_trend(length - change_idx, trend_type2, start_value=segment1[-1], prev_alpha=alpha1)
    else:
        segment2 = gen_stable_trend(length - change_idx, start_value=segment1[-1])
        alpha2 = 0
    ts = np.concatenate([segment1, segment2])
    ts = reconstruct_ts(ts)
    ts = np.round(ts, 3)
    input_text = f'Observe the following time series of length {length}: <ts><ts/>. Please answer the following questions: The trend change point in this time series occurs at point {change_idx + 1}. What is the trend of the first segment: upward, downward, or stable? Compared to the first segment, is the trend of the second segment accelerating, decelerating, undergoing a sudden change, or entering a new stable state? Please answer these two questions in order.'
    input_format = f'The answer format is: The trend of the first segment is xxx, and compared to the first segment, the trend of the second segment is xxx.'
    relative_trend = {
        'up': 'upward trend',
        'down': 'downward trend',
        'stable': 'stable state'
    }[trend_type2]
    if trend_type1 != 'stable':
        if trend_type2 == trend_type1:
            if abs(alpha2) > abs(alpha1) + 0.3:
                relative_trend = 'an accelerating ' + relative_trend
            elif abs(alpha2) < abs(alpha1) - 0.3:
                relative_trend = 'a decelerating ' + relative_trend
        else:
            relative_trend = 'an abrupt ' + relative_trend
            if trend_type2 == 'stable':
                relative_trend += 'a new stable state'
    
    answer = {
        "first_trend": {
            'up': 'upward trend',
            'down': 'downward trend',
            'stable': 'stable state'
        }[trend_type1],
        "relative_trend": relative_trend
    }
    output_text = f'The trend of the first segment is a {answer["first_trend"]}, and compared to the first segment, the trend of the second segment is {answer["relative_trend"]}.'
    return {
        'question': input_text,
        'answer': answer,
        'timeseries': [ts.tolist()],
        'input': input_text + input_format,
        'output': output_text,
        'meta': {
            'change_point': change_idx + 1,
            'trend_types': (trend_type1, trend_type2),
            'alphas': (alpha1, alpha2)
        }
    }


qa_example = gen_relative_trend(length=128)
print(qa_example['timeseries'])
visual_func(len(qa_example['timeseries'][0]), qa_example['timeseries'])

def multi_anomaly_detection(length=None):
    if length == None:
        length = random.randint(128, 512)
        if random.uniform(0, 1) > 0.9:
            length = 305
    ts_num = random.randint(5, min(3000//length, 8))
    ts_list = []
    ts_text_list = []
    segment_info = []
    anomaly_length = random.randint(length // 10, length // 5)
    start_idx = random.randint(0, length - anomaly_length - 1)
    end_idx = start_idx + anomaly_length - 1
    input_text = f'Observe the following multiple time series, where '
    rule_text = f'Under normal circumstances, the time series remain stable with minor spikes. In anomalous states, there will be consistent anomalies throughout the entire interval. Different time series have different types of anomalies (upward and downward; upward indicates that the values in this interval are higher than normal, while downward indicates that the values in this interval are lower than normal. Note that even if there is a platform change, it is still considered normal if it does not meet the anomaly criteria). Please determine which series exhibit anomalies in the interval {start_idx+1}-{end_idx+1}, and specify whether they are upward or downward anomalies. The anomaly determination rules for each series are as follows:\n'
    answer = []
    for i in range(ts_num):
        if random.uniform(0, 1) > 0.7:
            anomaly_type = 'normal'
            base, base_meta = gen_trend(length, 'stable')
            spike, spike_meta = gen_spike(length, 'random', spike_num=random.randint(1, 6))
            noise = np.random.normal(0, 0.05, length)
            ts = base + spike + noise
            ts_mean = np.random.uniform(-0.3, 0.3)
            ts_factor = np.random.uniform(1, 300)
            ts = (ts_mean + ts) * ts_factor
            ts_base_mean = (ts_mean + base.mean()) * ts_factor
            ts_list.append(ts)
            ts_text_list.append(f'Time series {i+1} does not show significant anomalies in this interval compared to other positions, remaining around the value of {ts_base_mean:.2f}.')
            input_text += f'Time series {i+1} is a time series of length {length}:<ts><ts/>\n'
            any_anomal = 'deviations above normal are considered anomalies' if random.uniform(0, 1) > 0.5 else 'deviations below normal are considered anomalies'
            rule_text += f'- Series {i+1}: {any_anomal}\n'
        elif random.uniform(0, 1) < 0.7:
            anomaly_type = 'up' if random.uniform(0, 1) > 0.5 else 'down'
            base, base_meta = gen_trend(length, 'stable')
            
            segment = np.zeros(length)
            segment[start_idx:end_idx+1] = random.uniform(0.7, 1) if anomaly_type == 'up' else -random.uniform(0.5, 1)
            spike, spike_meta = gen_spike(length, 'random', spike_num=random.randint(1, 6))
            noise = np.random.normal(0, 0.05, length)
            ts = base + segment + spike + noise
            ts_mean = np.random.uniform(-0.3, 0.3)
            ts_factor = np.random.uniform(1, 300)
            ts = (ts_mean + ts) * ts_factor
            ts_base_mean = (ts_mean + base.mean()) * ts_factor
            ts_list.append(ts)
            anomal = '上升' if anomaly_type == 'up' else '下降'
            anomal_info = '高于' if anomaly_type == 'up' else '低于'
            input_text += f'Time series {i+1} is a time series of length {length}:<ts><ts/>\n'
            ts_text_list.append(f'Time series {i+1} shows a significant platform {anomal} in this interval, with values changing from {ts_base_mean:.2f} to {ts[end_idx]:.2f}. Based on the relevant information "Time series {i+1}: {anomal_info} normal values are considered anomalies", it is determined that time series {i+1} is in an anomalous state, classified as a {anomal} anomaly.')
            rule_text += f'- Series {i+1}: {anomal_info} normal values are considered anomalies\n'
            answer.append({
                "index": i,
                "direction": anomal,
            })
        else :
            anomaly_type = 'up' if random.uniform(0, 1) > 0.5 else 'down'
            base, base_meta = gen_trend(length, 'stable')
            segment = np.zeros(length)
            segment[start_idx:end_idx+1] = random.uniform(0.7, 1) if anomaly_type == 'down' else -random.uniform(0.5, 1)
            spike, spike_meta = gen_spike(length, 'random', spike_num=random.randint(1, 6))
            noise = np.random.normal(0, 0.05, length)
            ts = base + segment + spike + noise
            ts_mean = np.random.uniform(-0.3, 0.3)
            ts_factor = np.random.uniform(1, 300)
            ts = (ts_mean + ts) * ts_factor
            ts_base_mean = (ts_mean + base.mean()) * ts_factor
            ts_list.append(ts)
            anomal = 'downward' if anomaly_type == 'up' else 'upward'
            anomal_info = 'deviations below' if anomaly_type == 'down' else 'deviations above'
            input_text += f'Time series {i+1} is a time series of length {length}:<ts><ts/>\n'
            ts_text_list.append(f'Time series {i+1} shows a significant platform {anomal} in this interval, with values changing from {ts_base_mean:.2f} to {ts[end_idx]:.2f}. Based on the relevant information "Time series {i+1}: {anomal_info} normal values are considered anomalies", which does not align with the platform {anomal} of this time series, it is determined that time series {i+1} remains in a normal state.')
            rule_text += f'- Series {i+1}: {anomal_info} normal values are considered anomalies\n'
    input_text += '。\n' + rule_text
    output_text = f'Based on the relevant information, the anomaly determinations for the time series are as follows:\n'
    output_text += '\n'.join(ts_text_list)
    return {
        'question': input_text,
        'answer': answer,
        'input': input_text,
        'output': output_text,
        'timeseries': [ts.tolist() for ts in ts_list]
    }
    return {
        'input': input_text,
        'output': output_text,
        'timeseries': [ts.tolist() for ts in ts_list]
    }


def multi_anomaly_rca(length=None, anomal_type='random'):
    if length == None:
        length = random.randint(128, 512)
        if random.uniform(0, 1) > 0.9:
            length = 305
    ts_num = random.randint(5, min(3000//length, 8))
    ts_list = []
    ts_text_list = []
    segment_info = []
    anomaly_length = random.randint(length // 10, length // 5)
    
    input_text = f'Observe the following multiple time series, where '
    rule_text = f'Under normal circumstances, the time series remain stable with minor spikes. In anomalous states, there will be consistent anomalies throughout the entire interval. Different time series have different types of anomalies (upward and downward; upward indicates that the values in this interval are higher than normal, while downward indicates that the values in this interval are lower than normal. Note that even if there is a platform change, it is still considered normal if it does not meet the anomaly criteria). Please identify the anomalies present in the above time series and attempt to locate the root causes of the anomalies.\nThe causes of anomalies are divided into two categories: overall environmental changes causing all time series to exhibit anomalies in the same interval, without any temporal order of anomaly intervals; propagation of internal anomalies within the time series, where the earliest occurring anomaly is the root cause, and the timing of anomaly occurrence reflects the propagation path. Please use the relevant information to determine whether each time series has anomalies, locate the root cause of the anomalies, and trace the propagation path (the path is represented as Time Series i -> Time Series j -> ...). The anomaly determination rules for each series are as follows:\n'
    input_format = f'The answer format is: The time series with anomalies are: i1, i2, i3 (sorted by index). \nBased on the temporal order of anomaly occurrences, the propagation path is: ix -> iy -> .... \nThe root cause of the anomalies is Time Series k.'
    answer = []
    if anomal_type == 'random':
        anomal_type = random.choices(['same_interval', 'different_interval'], [0.5, 0.5])[0]
        anomal_type = 'different_interval'
    rca_summary = ''
    rca_link = 'test'
    if anomal_type == 'same_interval':
        start_idx = random.randint(0, length - anomaly_length - 1)
        end_idx = start_idx + anomaly_length - 1
        rca_text = f'Based on the relevant information, multiple time series simultaneously exhibit significant interval anomalies from point {start_idx+1} to point {end_idx+1}. Therefore, the anomalies in this interval may be related to overall environmental changes.'
        for i in range(ts_num):
            if random.uniform(0, 1) > 0.7:
                anomaly_type = 'normal'
                base, base_meta = gen_trend(length, 'stable')
                spike, spike_meta = gen_spike(length, 'random', spike_num=random.randint(1, 6))
                noise = np.random.normal(0, 0.05, length)
                ts = base + spike + noise
                ts_mean = np.random.uniform(-0.3, 0.3)
                ts_factor = np.random.uniform(1, 300)
                ts = (ts_mean + ts) * ts_factor
                ts_base_mean = (ts_mean + base.mean()) * ts_factor
                ts_list.append(ts)
                ts_text_list.append(f'Time series {i+1} does not show significant anomalies in this interval compared to other positions, remaining around the value of {ts_base_mean:.2f}.')
                input_text += f'Time series {i+1} is a time series of length {length}:<ts><ts/>\n'
                any_anomal = 'higher than normal values are considered anomalies' if random.uniform(0, 1) > 0.5 else 'lower than normal values are considered anomalies'
                rule_text += f'- Series {i+1}: {any_anomal}\n'
            elif random.uniform(0, 1) < 0.7:
                anomaly_type = 'up' if random.uniform(0, 1) > 0.5 else 'down'
                base, base_meta = gen_trend(length, 'stable')
                
                segment = np.zeros(length)
                segment[start_idx:end_idx+1] = random.uniform(0.7, 1) if anomaly_type == 'up' else -random.uniform(0.5, 1)
                spike, spike_meta = gen_spike(length, 'random', spike_num=random.randint(1, 6))
                noise = np.random.normal(0, 0.05, length)
                ts = base + segment + spike + noise
                ts_mean = np.random.uniform(-0.3, 0.3)
                ts_factor = np.random.uniform(1, 300)
                ts = (ts_mean + ts) * ts_factor
                ts_base_mean = (ts_mean + base.mean()) * ts_factor
                ts_list.append(ts)
                anomal = 'upward' if anomaly_type == 'up' else 'downward'
                anomal_info = 'higher than' if anomaly_type == 'up' else 'lower than'
                input_text += f'Time series {i+1} is a time series of length {length}:<ts><ts/>\n'
                ts_text_list.append(f'Time series {i+1} shows a significant platform {anomal} in this interval, with values changing from {ts_base_mean:.2f} to {ts[end_idx]:.2f}. Based on the relevant information "Time series {i+1}: {anomal_info} normal values are considered anomalies", it is determined that time series {i+1} is in an anomalous state, classified as a {anomal} anomaly.')
                rule_text += f'- Series {i+1}: {anomal_info} normal values are considered anomalies\n'
                answer.append({
                    "index": i,
                    "direction": anomal,
                })
            else :
                anomaly_type = 'up' if random.uniform(0, 1) > 0.5 else 'down'
                base, base_meta = gen_trend(length, 'stable')
                segment = np.zeros(length)
                segment[start_idx:end_idx+1] = random.uniform(0.7, 1) if anomaly_type == 'down' else -random.uniform(0.5, 1)
                spike, spike_meta = gen_spike(length, 'random', spike_num=random.randint(1, 6))
                noise = np.random.normal(0, 0.05, length)
                ts = base + segment + spike + noise
                ts_mean = np.random.uniform(-0.3, 0.3)
                ts_factor = np.random.uniform(1, 300)
                ts = (ts_mean + ts) * ts_factor
                ts_base_mean = (ts_mean + base.mean()) * ts_factor
                ts_list.append(ts)
                anomal = 'downward' if anomaly_type == 'up' else 'upward'
                anomal_info = 'lower than' if anomaly_type == 'down' else 'higher than'
                input_text += f'Time series {i+1} is a time series of length {length}:<ts><ts/>\n'
                ts_text_list.append(f'Time series {i+1} shows a significant platform {anomal} in this interval, with values changing from {ts_base_mean:.2f} to {ts[end_idx]:.2f}. Based on the relevant information "Time series {i+1}: {anomal_info} normal values are considered anomalies", which does not align with the platform {anomal} of this time series, it is determined that time series {i+1} remains in a normal state.')
                rule_text += f'- Series {i+1}: {anomal_info} normal values are considered anomalies\n'
    else :
        anomal_num = random.randint(2, ts_num)
        anomal_index = random.sample(range(ts_num), anomal_num)
        rank_list = [i for i in anomal_index]
        random.shuffle(rank_list)
        rca_link = 'The order of anomalies in the time series is: ' + ' -> '.join([f'Time series {rank_list[i]+1}' for i in range(anomal_num)]) + '.'
        answer = {}
        answer['root_cause'] = rank_list[0]+1
        answer['rank'] = [rank_list[i]+1 for i in range(anomal_num)]
        anomal_interval_list = []
        rca_text = f'Based on the relevant information, multiple time series exhibit significant interval anomalies in different intervals, and these anomalies have a sequential order.'
        start_idx_list = random.sample(range(0, length - anomaly_length - 1), anomal_num)
        start_idx_list.sort()
        output_text = f'The time series exhibiting anomalies are: '
        output_text += ', '.join([f'Time series {i+1}' for i in sorted(anomal_index)]) + '.\n'
        output_text += f'The order of anomalies based on their occurrence time is: ' + ' -> '.join([f'Time series {rank_list[i]+1}' for i in range(anomal_num)]) + '.\n'
        output_text += f'The root cause of the anomaly is Time series {answer["root_cause"]}.'
        for i in range(anomal_num):
            start_idx = start_idx_list[i]
            end_idx = start_idx + anomaly_length - 1
            anomal_interval_list.append((start_idx, end_idx))
        ts_factor = np.random.uniform(1, 300)
        for i in range(ts_num):
            if i not in anomal_index:
                anomaly_type = 'normal'
                base, base_meta = gen_trend(length, 'stable')
                spike, spike_meta = gen_spike(length, 'random', spike_num=random.randint(1, 6))
                noise = np.random.normal(0, 0.05, length)
                ts = base + spike + noise
                ts_mean = np.random.uniform(-0.3, 0.3)
                
                ts = (ts_mean + ts) * ts_factor
                ts_base_mean = (ts_mean + base.mean()) * ts_factor
                ts_list.append(ts)
                ts_text_list.append(f'Time series {i+1} does not show significant anomalies, remaining around the value of {ts_base_mean:.2f}.')
                input_text += f'Time series {i+1} is a time series of length {length}:<ts><ts/>\n'
                any_anomal = 'Values higher than normal are considered anomalies' if random.uniform(0, 1) > 0.5 else 'Values lower than normal are considered anomalies'
                rule_text += f'- Series {i+1}: {any_anomal}\n'
            else:
                anomaly_type = 'up' if random.uniform(0, 1) > 0.5 else 'down'
                base, base_meta = gen_trend(length, 'stable')
                segment = np.zeros(length)
                start_idx, end_idx = anomal_interval_list[rank_list.index(i)]
                segment[start_idx:end_idx+1] = random.uniform(0.7, 1) if anomaly_type == 'up' else -random.uniform(0.5, 1)
                spike, spike_meta = gen_spike(length, 'random', spike_num=random.randint(1, 6))
                noise = np.random.normal(0, 0.05, length)
                ts = base + segment + spike + noise
                ts_mean = np.random.uniform(-0.3, 0.3)
                ts = (ts_mean + ts) * ts_factor
                ts_base_mean = (ts_mean + base.mean()) * ts_factor
                ts_list.append(ts)  
                anomal = 'upward' if anomaly_type == 'up' else 'downward'
                anomal_info = 'higher than' if anomaly_type == 'up' else 'lower than'
                input_text += f'Time series {i+1} is a time series of length {length}:<ts><ts/>\n'
                ts_text_list.append(f'Time series {i+1} exhibits a significant {anomal} plateau from point {start_idx+1} to point {end_idx+1}, with values changing from {ts_base_mean:.2f} to {ts[end_idx]:.2f}. Based on the information "Time series {i+1}: {anomal_info} normal is considered anomalous", it is determined that Time series {i+1} is in an anomalous state, exhibiting a {anomal} anomaly.')
                rule_text += f'- Series {i+1}: {anomal_info} normal is considered anomalous\n'
                if rank_list.index(i) == 0:
                    rca_text += f'Time series {i+1} exhibits an anomaly from point {start_idx+1} to point {end_idx+1}, thus the anomaly in Time series {i+1} may be the root cause.'
    input_text += '。\n' + rule_text

    return {
        'question': input_text,
        'answer': answer,
        'input': input_text + input_format,
        'output': output_text,
        'timeseries': [ts.tolist() for ts in ts_list]
    }
    return {
        'input': input_text,
        'output': output_text,
        'timeseries': [ts.tolist() for ts in ts_list]
    }



def gen_multi_change_trend_task(length=None, question_type='random', ts_num=None):
    if length == None:
        length = random.randint(28, 512)
    if question_type == 'random':
        question_type = random.choices(['unify', 'differ'], [1, 0])[0]
    if ts_num is None:
        ts_num = random.randint(2, min(3000//length, 5))
    else :
        ts_num = 1
    ts_list = []
    if question_type == 'unify':
        ts_change_num = random.randint(1, min(3, length // 5))
        change_idx_list = []
        if ts_change_num > 0:
            segment_len = length // (ts_change_num + 1)
            for i in range(ts_change_num):
                change_idx = random.randint(i * segment_len + segment_len // 2, (i + 1) * segment_len + segment_len // 2 - 1)
                change_idx_list.append(change_idx)
            change_idx_list = sorted(change_idx_list)
            
        segment_trend_list = []
        trend_types = ['up', 'down', 'stable', 'oscillate', 'platform_up', 'platform_down']
        def gen_up_down_trend(length, trend, start_value=0, prev_alpha=None): # 生成线性上升或下降，返回斜率
            alpha = None
            cnt = 0.1 * length
            while alpha is None or (prev_alpha is not None and abs(alpha - prev_alpha) < 0.5):
                if trend == 'up':
                    end_value = start_value + random.uniform(cnt, 0.5+cnt)
                else :
                    end_value = start_value - random.uniform(cnt, 0.5+cnt)
                cnt += 0.5
                alpha = (end_value - start_value) / length
            trend = np.linspace(start_value, end_value, length)
            return trend, alpha
        def gen_oscillate_trend(length, start_value=0): # 生成周期振荡，返回周期和振幅
            period = random.randint(1, length // 2)
            amplitude = random.uniform(0.3, 0.7)
            x = np.arange(length)
            trend = start_value + amplitude * np.sin(2 * np.pi * x / period)
            return trend, period, amplitude
        def gen_platform_trend(length, trend, start_value=0): # 整个段是一个整体的平台，返回平台高度
            if trend == 'platform_up':
                level = start_value + random.uniform(0.5, 1)
            else :
                level = start_value - random.uniform(0.5, 1)
            trend = np.ones(length) * level
            trend[-1] = start_value
            return trend, level
        def norm_ts(ts):
            ts_min = ts.min()
            ts_max = ts.max()
            if ts_max - ts_min < 1e-6:
                return ts - ts_min
            return (ts - ts_min) / (ts_max - ts_min)
        def reconstruct_ts(ts, ts_factor=None):
            ts_mean = random.uniform(-0.3, 0.3)
            ts_noise = np.random.normal(0, 0.01, len(ts))
            if ts_factor is None:
                ts_factor = random.uniform(1, 300)
            ts = (ts_mean + ts + ts_noise) * ts_factor
            return ts
        trend_weights = [0.3, 0.3, 0.4, 0., 0., 0.]
        trend_desc = {
            'up': 'upward',
            'down': 'downward',
            'stable': 'stable',
            'oscillate': 'oscillate',
            'platform_up': 'platform upward',
            'platform_down': 'platform downward'
        }
        input_text = f'Observe the following multiple time series, where '
        input_text += ', '.join([f'Time series {i+1} is a time series of length {length}:<ts><ts/>' for i in range(ts_num)]) + f'. Perform trend identification for each time series. If there are significant trend change points in the time series, please indicate the positions of these change points and explain the trend changes before and after them. (Since the anomalies have the same source, the change point coordinates of the anomalous time series are the same.) You need to identify the basic trends: upward, downward, stable. The types of trend changes include: significant acceleration/deceleration upward or downward, interval oscillation, platform upward or downward, and significant periodic changes.\nIn your specific response, please follow the format:\nFirst determine whether there are trend change points. If not, describe the overall trend. If yes, first indicate the positions of the change points, then analyze the specific trends of each segment formed by the change points in each time series.\nPlease note that the trend description should include numerical information, such as the rate of increase or decrease, to indicate whether there is acceleration, deceleration, or trend reversal.\nWhen describing each time series, first indicate the segment from which point to which point, then describe the trend type of that segment (stable, upward, downward, and determine whether the upward or downward trend is accelerating, decelerating, or abrupt (changing from deceleration to acceleration or vice versa)), and finally provide numerical information.\nPlease ensure the description is professional and systematic.'
        input_format = f'First determine whether each time series has trend change points. The format is: According to the relevant information, there are k change points common to all time series, located at points i1, i2, i3, ... / None of the above time series have significant trend changes.'
        input_format += f'If there are change points, analyze the trend of each segment in each time series in order. The format is: The trends of each segment of time series i are: xxx, yyy, ... (using the specified trend terms) / Otherwise, answer the overall trend of each time series in order. The format is: The overall trend of time series i is: xxx.'
        ts_desc_list = []
        ts_list = []
        output_text = ''
        if ts_change_num > 0:
            output_text += f'According to the relevant information, there are {len(change_idx_list)} change points common to all time series, located at points ' + ', '.join([str(idx + 1) for idx in change_idx_list]) + '.\n'
        else :
            output_text += f'None of the above time series have significant trend changes.\n'
        answer = {}
        for ts_i in range(ts_num):
            segment_trends = []
            prev_trend = None
            prev_alpha = None
            prev_beta = None
            ts = np.zeros(length)
            ts_desc = f'' 
            answer[ts_i] = []
            if ts_change_num == 0:
                trend = random.choices(trend_types, [0.3, 0.3, 0.4, 0, 0, 0])[0]
                segment_trends.append(trend)
                ts_desc += f'Time series {ts_i+1} has an overall trend of: {trend_desc[trend]}'
                if trend != 'stable':
                    trend_segment, alpha = gen_up_down_trend(length, trend, ts[0])
                    ts[:] = trend_segment
                    ts = reconstruct_ts(ts)
                else :
                    ts[:] = 0
                    ts = reconstruct_ts(ts)
                answer[ts_i].append(trend)
                ts_desc_list.append(ts_desc)
                ts_list.append(ts)
                output_text += ts_desc + '\n'
                continue
            else :
                output_text += f'Time series {ts_i+1} has segment trends as follows: ' 
                ts_desc = ''
            for seg in range(ts_change_num + 1):
                trend = random.choices(trend_types, trend_weights)[0]
                if seg == 0:
                    trend = random.choices(['up', 'down', 'stable'], [0.4, 0.4, 0.2])[0]
                while trend == prev_trend and trend in ['stable']: # 避免连续两个段为平稳
                    trend = random.choices(trend_types, trend_weights)[0]
                segment_trends.append(trend)
                prev_trend = trend
                start_idx = change_idx_list[seg - 1] if seg > 0 else 0
                end_idx = change_idx_list[seg] - 1 if seg < ts_change_num else length - 1
                seg_length = end_idx - start_idx + 1
                ts[start_idx] = 0 if start_idx == 0 else ts[start_idx - 1]
                if trend == 'stable': # 平稳
                    ts[start_idx:end_idx + 1] = ts[start_idx]
                elif trend in ['up', 'down']:
                    if prev_trend == trend:
                        trend_segment, alpha = gen_up_down_trend(seg_length, trend, ts[start_idx], prev_alpha)
                    else :
                        trend_segment, alpha = gen_up_down_trend(seg_length, trend, ts[start_idx])
                    ts[start_idx:end_idx + 1] = trend_segment
                elif trend == 'oscillate':
                    trend_segment, alpha, beta = gen_oscillate_trend(seg_length, ts[start_idx])
                    ts[start_idx:end_idx + 1] = trend_segment
                elif trend in ['platform_up', 'platform_down']:
                    trend_segment, alpha = gen_platform_trend(seg_length, trend, ts[start_idx])
                    ts[start_idx:end_idx + 1] = trend_segment
                if trend in ['up', 'down']:
                    segment_trends[-1] = (trend, start_idx, end_idx, alpha)
                    prev_alpha = alpha
                elif trend == 'oscillate':
                    segment_trends[-1] = (trend, start_idx, end_idx, alpha, beta)
                    prev_alpha = alpha
                    prev_beta = beta
                elif trend in ['platform_up', 'platform_down']:
                    segment_trends[-1] = (trend, start_idx, end_idx, alpha)
                    prev_alpha = None
                else :
                    segment_trends[-1] = (trend, start_idx, end_idx)
                    prev_alpha = None
                ts_desc_list.append(ts_desc)
                if segment_trends[-1] == 0:
                    print(f'debug {trend}')
            ts = norm_ts(ts)
            ts = reconstruct_ts(ts)
            ts_list.append(ts)
            answer['change_points'] = [i for i in change_idx_list]
            ts_descs = []
            for seg in range(ts_change_num + 1):   
                segment_trend = segment_trends[seg]
                prev_trend = segment_trends[seg - 1][0] if seg > 0 else None
                prev_alpha = segment_trends[seg - 1][3] if seg > 0 and type(segment_trends[seg - 1]) is not str and segment_trends[seg - 1][0] in ['up', 'down'] else None
                prev_beta = segment_trends[seg - 1][4] if seg > 0 and type(segment_trends[seg - 1]) is not str and segment_trends[seg - 1][0] == 'oscillate' else None
                

                if type(segment_trend) is str:
                    trend = segment_trend
                    start_idx = change_idx_list[seg - 1] if seg > 0 else 0
                    end_idx = change_idx_list[seg] - 1 if seg < ts_change_num else length - 1
                    alpha = None
                elif segment_trend[0] in ['up', 'down']:
                    trend, start_idx, end_idx, alpha = segment_trend
                elif segment_trend[0] == 'oscillate':
                    trend, start_idx, end_idx, alpha, beta = segment_trend
                elif segment_trend[0] in ['platform_up', 'platform_down']:
                    trend, start_idx, end_idx, alpha = segment_trend
                else :
                    trend, start_idx, end_idx = segment_trend
                    alpha = None
                if trend in ['up', 'down']:
                    alpha = (ts[end_idx] - ts[start_idx]) / (end_idx - start_idx + 1)
                    
                    if prev_alpha is not None:
                        if (trend == 'up' and prev_trend == 'up') or (trend == 'down' and prev_trend == 'down'):
                            if abs(alpha) > abs(prev_alpha) + 0.1:
                                ts_desc += f'This segment shows an accelerating {trend_desc[trend]} trend, with values changing linearly from {ts[start_idx]:.2f} to {ts[end_idx]:.2f}, at a rate of approximately {alpha:.4f}. Compared to the previous {trend_desc[prev_trend]} trend, the rate of change has significantly increased.\n'
                                ts_desc = 'Accelerating ' + trend_desc[trend]
                                trend = 'accelerate_' + trend
                                
                            elif abs(alpha) < abs(prev_alpha) - 0.1:
                                ts_desc += f'This segment shows a decelerating {trend_desc[trend]} trend, with values changing linearly from {ts[start_idx]:.2f} to {ts[end_idx]:.2f}, at a rate of approximately {alpha:.4f}. Compared to the previous {trend_desc[prev_trend]} trend, the rate of change has significantly decreased.\n'
                                ts_desc = 'Decelerating ' + trend_desc[trend]
                                trend = 'decelerate_' + trend
                                
                        elif (trend == 'up' and prev_trend == 'down') or (trend == 'down' and prev_trend == 'up'):
                            ts_desc += f'This segment shows a reversing {trend_desc[trend]} trend, with values changing linearly from {ts[start_idx]:.2f} to {ts[end_idx]:.2f}, at a rate of approximately {alpha:.4f}. Compared to the previous {trend_desc[prev_trend]} trend, the direction of the trend has reversed.\n'
                            ts_desc = 'Reversing ' + str(trend_desc[trend])
                            trend = 'reverse_' + trend
                            
                        else :
                            ts_desc += f'This segment shows a {trend_desc[trend]} trend, with values changing linearly from {ts[start_idx]:.2f} to {ts[end_idx]:.2f}, at a rate of approximately {alpha:.4f}.\n'
                            ts_desc = trend_desc[trend]
                    else :
                        ts_desc += f'This segment shows a {trend_desc[trend]} trend, with values changing linearly from {ts[start_idx]:.2f} to {ts[end_idx]:.2f}, at a rate of approximately {alpha:.4f}.\n'
                        ts_desc = trend_desc[trend]
                elif trend == 'stable':
                    ts_desc += f'This segment shows a {trend_desc[trend]} trend, with values remaining around {ts[start_idx]:.2f}.'

                    if seg > 0 : 
                        ts_desc += f'At this time, the time series has reached a new stable state.\n'
                        ts_desc = 'New stable state'
                    else :
                        ts_desc += '\n'
                        ts_desc = 'Stable state'
                elif trend == 'oscillate':
                    alpha = (ts.max() - ts.min()) / 2
                    ts_desc += f'This segment shows a {trend_desc[trend]} trend, with values oscillating around {ts[start_idx-1]:.2f} with a period of approximately {beta} and an amplitude of approximately {alpha:.2f}.'
                    if prev_trend == 'oscillate':
                        if abs(alpha) > abs(prev_beta) * 1.5:
                            ts_desc += f'Compared to the previous {trend_desc[prev_trend]} trend, the amplitude has significantly increased.'
                            trend = 'amplitude_increase_oscillate'
                        elif abs(alpha) < abs(prev_beta) * 0.67:
                            ts_desc += f'Compared to the previous {trend_desc[prev_trend]} trend, the amplitude has significantly decreased.'
                            trend = 'amplitude_decrease_oscillate'
                        if abs(beta) > abs(prev_beta) * 1.5:
                            ts_desc += f', the period has significantly increased.'
                            trend = 'period_increase_oscillate'
                        elif abs(beta) < abs(prev_beta) * 0.67:
                            ts_desc += f', the period has significantly decreased.'
                            trend = 'period_decrease_oscillate'
                        if ts_desc.endswith('，'):
                            ts_desc = ts_desc[:-1] + '。\n'
                        else :
                            ts_desc += '\n'
                elif trend in ['platform_up', 'platform_down']:
                    ts_desc += f'This segment shows a {trend_desc[trend]} trend, with values suddenly changing from {ts[start_idx-1]:.2f} to {ts[start_idx]:.2f}, and then returning to {ts[end_idx]:.2f}, forming a distinct {trend_desc[trend]}.'
                    
                    if prev_trend in ['platform_up', 'platform_down']:
                        if prev_trend != trend:
                            ts_desc += f'Compared to the previous {trend_desc[prev_trend]} trend, the trend direction has reversed.\n'
                            trend = 'reverse_' + trend
                        else :
                            ts_desc += f'Compared to the previous {trend_desc[prev_trend]} trend, '
                            alpha = ts[start_idx] - ts[start_idx-1]
                            prev_alpha = ts[change_idx_list[seg - 1] - 1] - ts[change_idx_list[seg - 1] - 2] if seg > 1 else ts[change_idx_list[seg - 1] - 1] - ts[change_idx_list[seg - 1] - 1]
                            if abs(alpha) > abs(prev_alpha) * 1.5:
                                ts_desc += f'The amplitude of the platform change has significantly increased.\n'
                                trend = 'amplitude_increase_' + trend
                            elif abs(alpha) < abs(prev_alpha) * 0.67:
                                ts_desc += f'The amplitude of the platform change has significantly decreased.\n'
                                trend = 'amplitude_decrease_' + trend
                            else :
                                ts_desc += '\n'
                answer[ts_i].append(trend)
                prev_trend = trend
                ts_descs.append(ts_desc)
            ts_desc_list.append(ts_desc)
            if ts_change_num > 0 :
                output_text += ', '.join(ts_descs)
                output_text = output_text.strip()
                output_text += '。\n'
        timeseries = [ts.tolist() for ts in ts_list]
        return {
            "input": input_text+input_format,
            "output": output_text,
            "timeseries": timeseries
        }
def gen_spike_task(length=None, ts_num=1):

    if ts_num == 1:
        if length is None:
            length = random.choice([64, 128, 256, 336, 512])
        base, meta_base = gen_trend(length, 'random')
        noise = np.random.normal(0, 0.005, length)
        ts_mean = np.random.uniform(-0.3, 0.3)
        ts_factor = np.random.uniform(1, 100)
        func = (ts_mean + base + noise) * ts_factor 
        func = np.round(func, 3)
        has_spike = random.choices([True, False], [0.7, 0.3])[0]
        input_format = f'The answer format is: If there is a spike, the time series has a spike at point i, with an upward/downward direction. Otherwise, answer: The time series does not have a spike. There is no need to consider the specific amplitude of the spike, just answer the spike information directly.'
        if has_spike:
            spike_point = random.randint(1, length - 2)
            spike_type = random.choice(['up', 'down'])
            spike = np.zeros(length)
            spike[spike_point] = np.random.uniform(0.5, 1.) if spike_type == 'up' else np.random.uniform(-1., -0.5)
            func += spike * ts_factor
            func = np.round(func, 3)
            input_text = f'Given a time series of length {length}:<ts><ts/>, please answer the following question. Does this time series contain a spike? If so, at which point does the spike occur? Also, please indicate the direction of the spike.'
            spike_type_en = 'upward' if spike_type == 'up' else 'downward'
            output_text = f'The time series has a spike at point {spike_point + 1}, with a {spike_type_en} direction.'
            answer = {
                "has_spike": True,
                "spike_points": [spike_point + 1]  # 从1开始计数
            }
        else:
            input_text = f'Given a time series of length {length}:<ts><ts/>, please answer the following question. Does this time series contain a spike? If so, at which point does the spike occur?'
            output_text = f'The time series does not have a spike.'
            answer = {
                "has_spike": False,
                "spike_points": []
            }
        return {
            'question': input_text,
            'answer': answer,
            'input': input_text + input_format,
            'output': output_text,
            'timeseries': [func.tolist()],
        }
    else :
        if length is None:
            length = random.choice([64, 128, 256, 336, 512])
        if ts_num == -1:
            ts_num = random.randint(2, min(3000//length, 10))
        ts_list = []
        spike_points_list = []
        ts_factor = np.random.uniform(1, 100)
        spike_en = {}
        for i in range(ts_num):
            base, meta_base = gen_trend(length, 'random')
            noise = np.random.normal(0, 0.005, length)
            ts_mean = np.random.uniform(-0.3, 0.3)
            
            func = (ts_mean + base + noise) * ts_factor
            func = np.round(func, 3)
            has_spike = random.choices([True, False], [0.7, 0.3])[0]
            
            if has_spike:
                spike_point = random.randint(1, length - 2)
                spike_type = random.choice(['up', 'down'])
                spike = np.zeros(length)
                spike[spike_point] = np.random.uniform(0.5, 1.) if spike_type == 'up' else np.random.uniform(-1., -0.5)

                func += spike * ts_factor
                func = np.round(func, 3)
                spike_points_list.append(spike_point + 1)
                spike_en[f'ts_{i+1}'] = 'upward' if spike_type == 'up' else 'downward'
            else:
                spike_points_list.append(None)
            ts_list.append(func)
        input_text = f'Given multiple time series, where'
        input_text += ', '.join([f'Time series {i+1} is a time series of length {length}:<ts><ts/>;' for i in range(ts_num)]) + f'. Please answer in order whether each time series contains a spike. If so, at which point does the spike occur? Also, please indicate the direction of the spike.'
        input_format = f'The answer format is: Time series i has a spike at point k, which is an upward/downward spike. / Time series i does not have a spike.\nThere is no need to consider the specific amplitude of the spike, just answer the spike information directly.'
        answer = {}
        output_text = f'The analysis of each time series is as follows:\n'
        for t in range(ts_num):
            if spike_points_list[t] is not None:
                answer[f'ts_{t+1}'] = {
                    "has_spike": True,
                    "spike_points": [spike_points_list[t]]
                }
                spike_type_en = spike_en[f'ts_{t+1}']
                output_text += f'Time series {t+1} has a spike at point {spike_points_list[t]}, with a {spike_type_en} direction.\n'
            else:
                answer[f'ts_{t+1}'] = {
                    "has_spike": False,
                    "spike_points": []
                }
                output_text += f'Time series {t+1} does not have a spike.\n'
        return {
            'question': input_text,
            'answer': answer,
            'input': input_text + input_format,
            'output': output_text,
            'timeseries': [ts.tolist() for ts in ts_list],
        }





def gen_ift_task(length=None, question_type='random', ts_num=None):
    task_mapper = {
        "extreme_task": {
            "func": gen_extreme_task,
            "type": "atomic",
            "weight": 1        
        },
        "change_point_task": {
            "func": gen_change_point,
            "type": "atomic",
            "weight": 1.5        
        },
        "trend_task":{
            "func": gen_trend_task,
            "type": "atomic",
            "weight": 1        
        },
        "period_task":{
            "func": gen_period_task,
            "type": "atomic",
            "weight": 1.2        
        },
        "segment_task":{
            "func": gen_segment_task,
            "type": "atomic",
            "weight": 1.5        
        },
        "relative_task":{
            "func": gen_relative_trend,
            "type": "atomic",
            "weight": 1.2        
        },
        "spike_task":{
            "func": gen_spike_task,
            "type": "atomic",
            "weight": 1        
        }
    }
    if length is None:
        length = random.randint(36, 512)
    if ts_num is None:
        ts_num = random.randint(2, 4)
    time_series_list = []
    input_text = 'There are multiple time series questions. Each time series is a sequence of numbers. And each question is about the time series with the same index. Please answer each question with index.\n'
    output_text = ''
    answer = []
    for ts in range(ts_num):
        question = random.choice(list(task_mapper.keys()))
        question_info = task_mapper[question]
        question_func = question_info['func']
        qa_example = question_func(length=length, ts_num=1)

        question_text = qa_example['input']
        answer_text = qa_example['output']
        answer_meta = qa_example['answer']
        answer.append({
            'question_type': question,
            'answer_meta': answer_meta
        })
        input_text += f'Q{ts+1}: Based on time seires:<ts><ts/>. Answer the question: {question_text}\n'
        output_text += f'A{ts+1}: {answer_text}\n'
        if type(qa_example["timeseries"][0]) != list: 
          print(f'Error at {question}')
          qa_example["timeseries"] = [qa_example["timeseries"]]
        if len(qa_example["timeseries"][0]) != length:
          raise ValueError(f'{len(qa_example["timeseries"][0])} != {length} at {question}')
        time_series_list.append(qa_example["timeseries"][0])
    return {
        'input': input_text,
        'output': output_text,
        'question': input_text,
        'answer': answer,
        'timeseries': time_series_list
    }
task_mapper = {
    "extreme_task": {
        "func": gen_extreme_task,
        "type": "atomic",
        "weight": 1        
    },
    "change_point_task": {
        "func": gen_change_point,
        "type": "atomic",
        "weight": 1        
    },
    "trend_task":{
        "func": gen_trend_task,
        "type": "atomic",
        "weight": 1        
    },
    "period_task":{
        "func": gen_period_task,
        "type": "atomic",
        "weight": 1        
    },
    "spike_task":{
        "func": gen_spike_task,
        "type": "atomic",
        "weight": 1        
    },
    "comparison_task":{
        "func": gen_double_check,
        "type": "molecular",
        "weight": 1        
    },
    "segment_task":{
        "func": gen_segment_task,
        "type": "molecular",
        "weight": 1.5        
    },
    "relative_task":{
        "func": gen_relative_trend,
        "type": "molecular",
        "weight": 1.2        
    },
    "anomaly_detection_task":{
        "func": multi_anomaly_detection,
        "type": "composite",
        "weight": 1         
    },
    "root_cause_analysis_task":{
        "func": multi_anomaly_rca,
        "type": "composite",
        "weight": 1         
    },
    "describe_task":{
        "func": gen_multi_change_trend_task,    
        "type": "composite",
        "weight": 1         
    },
    "ift_task":{
        "func": gen_ift_task,
        "type": "composite",
        "weight": 2
    }
}

          
    


def prompt_polish(question, answer) -> str:
    prompt_template = '''Rewrite the answer to sound like a natural response from a time-series expert.

Constraints:
- Keep the original conclusion unchanged.
- Do not add new reasoning steps or explanations.
- Use smooth, non-template language.
- The answer should feel like it comes from visually or analytically inspecting the time series.
Original Question:
{QUESTION}
Original Answer:
{ANSWER}
Rewritten Answer:'''
    return prompt_template.format(QUESTION=question, ANSWER=answer)
q = 'Observe the following time series of length 336: <ts><ts/>. Please identify whether there is any seasonal variation in this time series. If so, what is the period of the seasonal variation?The answer format is: The time series has seasonal variation with a period of xxx. / The time series has no seasonal variation.'
a = 'The time series has seasonal variation with a period of 84.'
print(prompt_polish(q, a))

print(get_model_response(prompt_polish(q, a)))
import re

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
def polish_q_and_a(q, a, t):
    prompt_template = f'''
You are an expert assistant refining synthetic question–answer pairs for time-series analysis tasks.

Your task is to rewrite BOTH the question and the answer to improve naturalness and fluency, while keeping them strictly aligned in meaning.

Constraints:
- Preserve the original intent and information requirement of the question.
- Preserve the factual conclusion of the answer.
- The rewritten answer MAY include a brief, high-level description of the reasoning or observation that supports the conclusion.
- Do NOT introduce new conclusions, assumptions, or numerical results.
- Do NOT provide step-by-step derivations or detailed calculations.
- Avoid rigid or template-style phrasing.
- Keep the rewritten question and answer concise and well-matched.
- Must keep the special <ts><ts/> to mark the time series in the question.

Output format (STRICT):
- Return ONLY a valid JSON object.
- The JSON object MUST contain exactly two keys:
  - "polished_question": a string
  - "polished_answer": a string
- Do NOT include any extra text, comments, or formatting outside the JSON.
- Do NOT wrap the JSON in Markdown code blocks.

Original Question:
{q}

Original Answer:
{a}
'''
    prompt = prompt_template
    def checker(q_, a_):
        if q_.count('<ts><ts/>') != len(t):
            return False
        return True
    RETRY_LIMIT = 10
    while RETRY_LIMIT > 0:
        RETRY_LIMIT -= 1
        ans = parse_llm_json(get_model_response(prompt))
        if checker(ans['polished_question'], ans['polished_answer']):
            return ans
    return None


from tqdm import tqdm
import json
import random

qa_count = 50000
output_file = 'polished_qa_1222.json'
output_data = []
count = 0
qa_idx = 0

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

def generate_one_qa(task_mapper):
    type_list = list(task_mapper.keys())
    weights = [task_mapper[t]['weight'] for t in type_list]
    task_name = random.choices(type_list, weights=weights)[0]

    task_func = task_mapper[task_name]['func']
    qa_example = task_func()

    ans = polish_q_and_a(
        qa_example['input'],
        qa_example['output'],
        qa_example['timeseries']
    )
    ans['timeseries'] = qa_example['timeseries']
    return ans  # 可能是 None



output_data = []
count = 0

max_workers = 64   # ✅ 根据机器 / API 限速调
lock = threading.Lock()
import json

import os
count = len(output_data)
file_name = 'generated_timeseries_bench_plus_polished'
description = '在模型output中添加了多个变点'
ori_file_path = f'/zzr/data/timeeval_train/generate_data/{file_name}'
os.makedirs(ori_file_path, exist_ok=True)
file_path = f'{ori_file_path}/{count//1000}K.jsonl'
if os.path.exists(file_path):
    os.remove(file_path)
from tqdm import tqdm
test_data = []
with open(file_path, 'a') as f:   
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pbar = tqdm(total=qa_count)

        futures = {
            executor.submit(generate_one_qa, task_mapper)
            for _ in range(max_workers)
        }

        while count < qa_count:
            for future in as_completed(futures):
                futures.remove(future)

                try:
                    ans = future.result()
                except Exception as e:
                    ans = None
                    print(f"[WARN] QA generation failed: {e}")

                if ans is not None:
                    with lock:
                        if count < qa_count:
                            if type(ans['timeseries'][0]) != list:
                                ans['timeseries'] = [ans['timeseries']]
                            json.dump(ans, f, ensure_ascii=False)
                            f.write('\n')
                            if len(test_data) < 10:
                                test_data.append(ans)
                            count += 1
                            pbar.update(1)

                if count < qa_count:
                    futures.add(
                        executor.submit(generate_one_qa, task_mapper)
                    )

                if count >= qa_count:
                    break

        pbar.close()

test_data_file = file_path.replace('.jsonl', '_test.jsonl')
json.dump(test_data, open(test_data_file, 'w'), ensure_ascii=False, indent=4)
info_data = json.load(open('/zzr/scripts/TimeSense/ChatTS-Training/data/dataset_info.json', 'r'))
info_data[file_name] = {
    "file_name": file_path,
    "columns": {
        "prompt": "input",
        "response": "output",
        "timeseries": "timeseries"
    },
    "description": description
}
json.dump(info_data, open('/zzr/scripts/TimeSense/ChatTS-Training/data/dataset_info.json', 'w'), ensure_ascii=False, indent=4)