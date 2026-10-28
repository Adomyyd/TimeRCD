import sys
import numpy as np
import random
from tqdm import tqdm
import re
import json
from typing import *
from trend_utils import generate_random_points, generate_trend_prompt, generate_trend_curve, generate_trend_list, generate_arima_series
import copy
import pywt
from scipy.interpolate import interp1d
from config import ALL_ATTRIBUTE_SET

# Config
ENABLE_MULTIPLE_TREND = True
ENABLE_DROP_PROMPT = True
ENABLE_MULTIPLE_SEASONAL = False
ENABLE_MULTIPLE_NOISE = False

BACKGROUND_PERIODIC_SPIKE_PROB = 0.02
BACKGROUND_SPIKE_MIN_PERIOD = 16
BACKGROUND_SPIKE_WIDTH_RATIO_MIN = 0.10
BACKGROUND_SPIKE_WIDTH_RATIO_MAX = 0.20
BACKGROUND_SPIKE_WIDTH_MIN = 2
BACKGROUND_SPIKE_WIDTH_MAX = 20
BACKGROUND_SPIKE_WIDTH_PERIOD_CAP = 0.35
BACKGROUND_SPIKE_BASE_AMP_MIN = 5.0
BACKGROUND_SPIKE_BASE_AMP_MAX = 10.0
BACKGROUND_SPIKE_JITTER_RATIO = 0.5
BACKGROUND_PERIODIC_NOISE_MODULATION_PROB = 0.10
BACKGROUND_NOISE_MULTIPLIER_MEAN = 5.0
BACKGROUND_NOISE_MULTIPLIER_STD = 2.0
BACKGROUND_NOISE_MULTIPLIER_MIN = 0.0
BACKGROUND_NOISE_MULTIPLIER_MAX = 10.0
BACKGROUND_NOISE_SEGMENT_RATIO_MIN = 0.20
BACKGROUND_NOISE_SEGMENT_RATIO_MAX = 0.80

all_wavelet_families = ['db2', 'db4', 'sym2', 'sym5', 'coif1', 'coif3', 'haar', 'bior1.3', 'dmey']
all_attribute_set = ALL_ATTRIBUTE_SET


def generate_random_attributes(overall_attribute: Dict[str, Dict[str, float]], change_attribute: Dict[str, float], change_positions: Optional[List[Tuple[Optional[int], Optional[float]]]] = None, seq_len: int = 512):
    if change_positions is None:
        change_positions = [(None, None) for _ in range(random.randint(0, 5))]
    description = {}

    description["seasonal"] = {
        "type": np.random.choice(list(overall_attribute['seasonal']), p=np.array(list(overall_attribute['seasonal'].values()))/sum(list(overall_attribute['seasonal'].values())))
    }

    trend_candidates = overall_attribute['trend'].copy()
    if not ENABLE_MULTIPLE_TREND and "multiple" in trend_candidates:
        trend_candidates.pop("multiple")
    trend_char = np.random.choice(list(trend_candidates), p=np.array(list(trend_candidates.values()))/sum(list(trend_candidates.values())))
    
    description["trend"] = {
        "type": trend_char
    }

    num_local_chars = len(change_positions)
    local_chars = list(np.random.choice(list(change_attribute), size=num_local_chars, p=np.array(list(change_attribute.values()))/sum(list(change_attribute.values()))))
    
    description["local"] = []
    for char in local_chars:
        local_position, local_amplitude = change_positions.pop()
        description["local"].append({
            "type": char,
            "position_start": local_position,
            "amplitude": local_amplitude
        })

    if 'no periodic fluctuation' not in description["seasonal"]['type']:
        description["frequency"] = {'type': np.random.choice(list(overall_attribute['frequency']), p=np.array(list(overall_attribute['frequency'].values()))/sum(list(overall_attribute['frequency'].values())))}
    else:
        description["frequency"] = {'type': 'no periodicity'}
        
    description["noise"] = {'type': np.random.choice(list(overall_attribute['noise']), p=np.array(list(overall_attribute['noise'].values()))/sum(list(overall_attribute['noise'].values())))}

    return description

def generate_controlled_attributes(attribute_set, change_positions: Optional[List[Tuple[Optional[int], Optional[float]]]] = None, num_seasonal_anomalies: Optional[int] = None):
    """
    从 attribute_set 生成受控属性。
    
    Parameters:
    ----------
    attribute_set : dict
        属性集合，通常来自 synthetic.json 中的 metric 配置
    change_positions : list of tuple, optional
        局部变化的位置和振幅列表
    num_seasonal_anomalies : int, optional
        季节性异常的数量。如果为 None，会根据情况自动确定
    
    Returns:
    -------
    dict
        生成的属性字典，包含 num_seasonal_anomalies 字段
    """
    if change_positions is None:
        change_positions = [(None, None) for _ in range(random.randint(0, 5))]
    description = {}

    seasonal_p = [all_attribute_set['overall_attribute']['seasonal'][i] for i in attribute_set['seasonal']['attributes']]
    description["seasonal"] = {
        "type": np.random.choice(list(attribute_set['seasonal']['attributes']), p=np.array(seasonal_p)/sum(seasonal_p)),
        "amplitude": random.uniform(attribute_set['seasonal']['amplitude']['min'], attribute_set['seasonal']['amplitude']['max'])
    }
    
    if not ENABLE_MULTIPLE_TREND:
        if "multiple" in attribute_set['trend']['attributes']:
            attribute_set['trend']['attributes'].remove("multiple")
            if len(attribute_set['trend']['attributes']) == 0:
                attribute_set['trend']['attributes'] = ['increase', 'decrease', 'keep steady']
    trend_p = [all_attribute_set['overall_attribute']['trend'][i] for i in attribute_set['trend']['attributes']]
    description["trend"] = {
        "type": np.random.choice(list(attribute_set['trend']['attributes']), p=np.array(trend_p)/sum(trend_p)),
        "start": random.uniform(attribute_set['trend']['start']['min'], attribute_set['trend']['start']['max']),
        "amplitude": random.uniform(attribute_set['trend']['amplitude']['min'], attribute_set['trend']['amplitude']['max'])
    }
    if description["trend"]["type"] == "arima":
        description["seasonal"]['type'] = 'no periodic fluctuation'
        description["seasonal"]['amplitude'] = 0.0

    num_local_chars = len(change_positions)
    change_p = [all_attribute_set['change'][i] for i in attribute_set['change']['attributes']]
    local_chars = list(np.random.choice(list(attribute_set['change']['attributes']), size=num_local_chars, p=np.array(change_p)/sum(change_p)))
    
    description["local"] = []
    for char in local_chars:
        description["local"].append({
            "type": char,
            "position_start": None,
            "amplitude": random.uniform(attribute_set['change']['amplitude']['min'], attribute_set['change']['amplitude']['max'])
        })

    if 'no periodic fluctuation' not in description["seasonal"]['type']:
        # Generate period and then determine type
        period = random.uniform(attribute_set['seasonal']['period']['min'], attribute_set['seasonal']['period']['max'])
        if period < 30.0:
            description["frequency"] = {'type': 'high frequency', 'period': round(period, 1)}
        else:
            description["frequency"] = {'type': 'low frequency', 'period': round(period, 1)}
    else:
        description["frequency"] = {'type': 'no periodicity'}
        
    noise_p = [all_attribute_set['overall_attribute']['noise'][i] for i in attribute_set['noise']['attributes']]
    description["noise"] = {'type': np.random.choice(list(attribute_set['noise']['attributes']), p=np.array(noise_p)/sum(noise_p))}
    while True:
        if description["seasonal"]['type'] != "no periodic fluctuation" and description["noise"]['type'] not in ['almost no noise', 'low noise']:
            description["noise"] = {'type': np.random.choice(list(attribute_set['noise']['attributes']), p=np.array(noise_p)/sum(noise_p))}
        else:
            break
    if description["trend"]['type'] == "arima":
        description["noise"]["type"] = "almost no noise"
    
    # 确定季节性异常数量（与 generate_attributes_from_set 保持一致）
    if num_seasonal_anomalies is None:
        if description["seasonal"]['type'] != "no periodic fluctuation":
            # 有周期性时，可能有季节性异常
            num_seasonal_anomalies = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            # 无周期性时，不会有季节性异常
            num_seasonal_anomalies = 0
    
    description['num_seasonal_anomalies'] = num_seasonal_anomalies
    
    return description

def generate_seasonal_wave(period, amplitude_list, split_points, seq_len, wave_type=None, params_dict=None):
    # Time array
    t = np.linspace(1, seq_len, seq_len)
    data = np.zeros(seq_len)
    base_frequency = 1 / period

    # Amplitude series  
    amplitude_series = np.zeros(seq_len)
    for i in range(len(amplitude_list)):
        amplitude_series[split_points[i]:split_points[i + 1]] = amplitude_list[i]
    sliding_window = 5
    for i in range(seq_len - sliding_window):
        amplitude_series[i + sliding_window // 2] = np.mean(amplitude_series[i:i + sliding_window])

    if params_dict is None:  # when generating normal series is None, when generating abnormal series is not None
        # Generate random parameters
        if wave_type is None:
            wave_type = str(np.random.choice(['sin', 'square', 'triangle', 'wavelet'], p=[0.4, 0.1, 0.1, 0.4]))
        params_dict = {'wave_type': wave_type}

        if wave_type == 'sin':
            # Ensure upper bound > 2 even when period is small
            upper = max(3, min(max(int(period) // 6, 2), 10) + 1)
            num_harmonics = np.random.randint(2, upper)
            params_dict['num_harmonics'] = num_harmonics
            harmonics = []
            for n in range(1, num_harmonics + 1):
                phase = np.random.uniform(0, 2 * np.pi)
                amp_mod_depth = np.random.uniform(0, 0.05)
                mod_freq_factor = np.random.uniform(1, 3)
                mod_phase = np.random.uniform(0, 2 * np.pi)
                harmonics.append({
                    'phase': phase,
                    'amp_mod_depth': amp_mod_depth,
                    'mod_freq_factor': mod_freq_factor,
                    'mod_phase': mod_phase
                })
            params_dict['harmonics'] = harmonics
        elif wave_type == 'square':
            start = np.random.uniform(0, 0.3)
            duration = np.random.uniform(0.1, 0.3)
            params_dict['start'] = start
            params_dict['duration'] = duration
        elif wave_type == 'triangle':
            start = np.random.uniform(0, 0.3)
            duration = np.random.uniform(0.1, 0.6)
            params_dict['start'] = start
            params_dict['duration'] = duration
        elif wave_type == 'wavelet':
            params_dict['period'] = period
            num_families = np.random.randint(2, 4)
            wavelet_components = []
            # Expanded available_families to include more wavelet types
            available_families = all_wavelet_families
            for _ in range(num_families):
                family = np.random.choice(available_families)
                num_wavelets = np.random.randint(2, 4)
                wavelets = []
                for _ in range(num_wavelets):
                    s = np.random.uniform(period / 8, period / 2)
                    tau = np.random.uniform(0, period)
                    A = np.random.uniform(0.5, 1.0)
                    wavelets.append({'s': s, 'tau': tau, 'A': A})
                wavelet_components.append({'family': family, 'wavelets': wavelets})
            params_dict['wavelet_components'] = wavelet_components
    else:
        # Validate provided params_dict
        if 'wave_type' not in params_dict:
            raise ValueError("params_dict must contain 'wave_type'")
        wave_type = params_dict['wave_type']
        if wave_type == 'sin':
            if 'num_harmonics' not in params_dict or 'harmonics' not in params_dict:
                raise ValueError("For 'sin' wave, params_dict must contain 'num_harmonics' and 'harmonics'")
        elif wave_type in ['square', 'triangle']:
            if 'start' not in params_dict or 'duration' not in params_dict:
                raise ValueError(f"For '{wave_type}' wave, params_dict must contain 'start' and 'duration'")
        elif wave_type == 'wavelet':
            if 'period' not in params_dict or 'wavelet_components' not in params_dict:
                raise ValueError("For 'wavelet' wave_type, params_dict must contain 'period' and 'wavelet_components'")
        else:
            raise ValueError(f"Unsupported wave_type: {wave_type}")

    # Generate wave using params_dict
    if wave_type == 'sin':
        num_harmonics = params_dict['num_harmonics']
        harmonics = params_dict['harmonics']
        for n in range(1, num_harmonics + 1):
            harmonic_params = harmonics[n - 1]
            phase = harmonic_params['phase']
            amp_mod_depth = harmonic_params['amp_mod_depth']
            mod_freq_factor = harmonic_params['mod_freq_factor']
            mod_phase = harmonic_params['mod_phase']
            harmonic_amplitude = amplitude_series / n * (
                1 + amp_mod_depth * np.sin(mod_freq_factor * np.pi * t / seq_len + mod_phase))
            data += harmonic_amplitude * np.sin(2 * np.pi * base_frequency * n * t + phase)
    elif wave_type == 'square':
        start = params_dict['start']
        duration = params_dict['duration']
        for i in range(seq_len):
            cycle_pos = (t[i] % period) / period
            if start <= cycle_pos < start + duration:
                data[i] = amplitude_series[i]
            else:
                data[i] = 0.0
    elif wave_type == 'triangle':
        start = params_dict['start']
        duration = params_dict['duration']
        end = start + duration
        for i in range(seq_len):
            cycle_pos = (t[i] % period) / period
            if start <= cycle_pos < end:
                if cycle_pos < (start + end) / 2:
                    data[i] = amplitude_series[i] * 2 * (cycle_pos - start) / duration
                else:
                    data[i] = amplitude_series[i] * 2 * (end - cycle_pos) / duration
            else:
                data[i] = 0.0
    elif wave_type == 'wavelet':
        period = params_dict['period']
        wavelet_components = params_dict['wavelet_components']
        for component in wavelet_components:
            family = component['family']
            wavelets = component['wavelets']
            wavelet = pywt.Wavelet(family)
            wavefun_result = wavelet.wavefun(level=10)
            if len(wavefun_result) == 3:
                phi, psi, x = wavefun_result
                wavelet_func = psi
            elif len(wavefun_result) == 5:
                phi_d, psi_d, phi_r, psi_r, x = wavefun_result
                wavelet_func = psi_r
            else:
                raise ValueError("Unexpected wavefun output")
            psi_interp = interp1d(x, wavelet_func, kind='linear', bounds_error=False, fill_value=0)
            for wp in wavelets:
                s = wp['s']
                tau = wp['tau'] % int(period)
                A = wp['A']
                u = (t  % int(period) - tau) / s
                if random.random() < 0.5:
                    u = (t % int(period) - tau) / period * (x.max() - x.min()) + x.min()
                else:
                    scale_factor = random.uniform(2, 6)
                    u = (t % int(period) - tau) / period * (x.max() - x.min()) * scale_factor - scale_factor / 2 * (x.max() - x.min())
                # scale_factor = 4
                # u = (t % int(period) - tau) / period * (x.max() - x.min()) * scale_factor - scale_factor / 2 * (x.max() - x.min())
                psi_values = psi_interp(u)
                data += A / np.sqrt(s) * psi_values

    # Normalize data
    if data.max() > data.min():
        data = data / (data.max() - data.min() + 1e-7) * max(amplitude_list)
    data -= np.mean(data)

    return data, params_dict


def generate_sin_noise(amplitude, seq_len):
    # Time array
    t = np.linspace(0, seq_len, seq_len)
    data = np.zeros(seq_len)

    num_harmonics = 200
    for n in range(1, num_harmonics + 1):
        phase = np.random.uniform(0, 2 * np.pi)
        cur_freq = np.random.uniform(50 / seq_len, 200 / seq_len)
        data += np.sin(cur_freq * t + phase) * np.random.uniform(0.3, 1.0)

    # normalize to amplitude
    data = data / (data.max() - data.min() + 1e-7) * amplitude
    data -= np.mean(data)

    return data

def generate_ts_change(length: int, amplitude: float, add_random_noise: bool=True):
    x = np.arange(length) / length
    func = random.choice([
        lambda x: x ** 2,
        lambda x: np.sin(x * np.pi / 2),
        lambda x: x,
        lambda x: 1.0 - (1.0 - x) ** 2
    ])
    cur_value = func(x)

    if add_random_noise:
        # Randomly add noise
        if random.random() > 0.8 and length > 3:
            cur_value += np.random.uniform(-1.0, 1.0, length) * np.random.uniform(0.1, 0.3)

    cur_value = cur_value / (cur_value.max() - cur_value.min() + 1e-7) * amplitude

    return cur_value

def generate_spike(amplitude: float):
    rise_length = np.random.choice([1, 2, 3], p = [0.8, 0.15, 0.05])
    fall_length = np.random.choice([1, 2, 3], p = [0.8, 0.15, 0.05])
    peak_length = np.random.choice([0, 1, 2], p = [0.96, 0.03, 0.01])

    result = np.zeros(rise_length + fall_length + peak_length, dtype=np.float32)
    result[:rise_length] += generate_ts_change(rise_length, amplitude)
    result[rise_length:] += amplitude
    result[rise_length + peak_length:] += generate_ts_change(fall_length, -amplitude)
    
    return result


def generate_background_periodic_spikes(seq_len: int, overall_amplitude: float):
    background_signal = np.zeros(seq_len, dtype=np.float32)
    metadata = {
        "enabled": False,
        "count": 0
    }

    if seq_len < BACKGROUND_SPIKE_MIN_PERIOD or overall_amplitude <= 0:
        return background_signal, metadata
    if random.random() >= BACKGROUND_PERIODIC_SPIKE_PROB:
        return background_signal, metadata

    period_min = max(BACKGROUND_SPIKE_MIN_PERIOD, int(round(seq_len / 500.0)))
    period_max = max(period_min + 1, int(round(seq_len / 50.0)))
    period = random.randint(period_min, period_max)

    raw_width = int(round(period * random.uniform(BACKGROUND_SPIKE_WIDTH_RATIO_MIN, BACKGROUND_SPIKE_WIDTH_RATIO_MAX)))
    raw_width = max(BACKGROUND_SPIKE_WIDTH_MIN, min(raw_width, BACKGROUND_SPIKE_WIDTH_MAX))
    safe_width_cap = max(BACKGROUND_SPIKE_WIDTH_MIN, int(np.floor(period * BACKGROUND_SPIKE_WIDTH_PERIOD_CAP)))
    total_width = min(raw_width, safe_width_cap)
    total_width = max(BACKGROUND_SPIKE_WIDTH_MIN, total_width)

    direction = 1 if random.random() >= 0.5 else -1
    base_amplitude = random.uniform(BACKGROUND_SPIKE_BASE_AMP_MIN, BACKGROUND_SPIKE_BASE_AMP_MAX) * overall_amplitude
    jitter_bound = BACKGROUND_SPIKE_JITTER_RATIO * overall_amplitude
    offset = random.randint(0, max(0, period - 1))

    positions = []
    amplitudes = []

    for anchor in range(offset, seq_len - total_width + 1, period):
        jitter = random.randint(-1, 1)
        spike_start = min(max(0, anchor + jitter), seq_len - total_width)
        if positions and spike_start < positions[-1] + total_width:
            continue

        spike_amplitude = base_amplitude + random.uniform(-jitter_bound, jitter_bound)
        spike_amplitude = max(0.1 * overall_amplitude, spike_amplitude)
        signed_amplitude = direction * spike_amplitude

        peak_length = 1 if total_width >= 4 and random.random() < 0.2 else 0
        max_rise_length = max(1, total_width - peak_length - 1)
        rise_length = int(round(total_width * random.uniform(0.25, 0.40)))
        rise_length = max(1, min(rise_length, max_rise_length))
        fall_length = total_width - rise_length - peak_length
        if fall_length < 1:
            fall_length = 1
            rise_length = max(1, total_width - peak_length - fall_length)

        spike = np.zeros(total_width, dtype=np.float32)
        spike[:rise_length] += generate_ts_change(rise_length, signed_amplitude, add_random_noise=False)
        spike[rise_length:] += signed_amplitude
        spike[rise_length + peak_length:] += generate_ts_change(fall_length, -signed_amplitude, add_random_noise=False)

        background_signal[spike_start:spike_start + total_width] += spike
        positions.append(int(spike_start))
        amplitudes.append(float(spike_amplitude))

    if not positions:
        return background_signal, metadata

    metadata = {
        "enabled": True,
        "direction": "positive" if direction > 0 else "negative",
        "period": int(period),
        "period_range": [int(period_min), int(period_max)],
        "width": int(total_width),
        "base_amplitude": round(float(base_amplitude), 3),
        "jitter_bound": round(float(jitter_bound), 3),
        "count": len(positions),
        "positions": positions,
        "amplitude_range": [
            round(float(min(amplitudes)), 3),
            round(float(max(amplitudes)), 3)
        ]
    }
    return background_signal, metadata


def apply_background_periodic_noise_modulation(noise: np.ndarray, attribute_pool: dict):
    seq_len = len(noise)
    metadata = {
        "enabled": False,
        "count": 0
    }

    frequency = attribute_pool.get("frequency", {})
    period = frequency.get("period", 0)
    if frequency.get("type") == "no periodicity" or period is None or period <= 0:
        return noise, metadata
    if random.random() >= BACKGROUND_PERIODIC_NOISE_MODULATION_PROB:
        return noise, metadata

    period_int = max(1, int(round(period)))
    num_cycles = seq_len // period_int
    if num_cycles < 2 or period_int < 2:
        return noise, metadata

    segment_length = int(round(
        period_int * random.uniform(BACKGROUND_NOISE_SEGMENT_RATIO_MIN, BACKGROUND_NOISE_SEGMENT_RATIO_MAX)
    ))
    segment_length = max(1, min(segment_length, period_int - 1))
    segment_offset = random.randint(0, period_int - segment_length)
    multiplier = float(np.clip(
        random.normalvariate(BACKGROUND_NOISE_MULTIPLIER_MEAN, BACKGROUND_NOISE_MULTIPLIER_STD),
        BACKGROUND_NOISE_MULTIPLIER_MIN,
        BACKGROUND_NOISE_MULTIPLIER_MAX,
    ))

    modulated_noise = noise.copy()
    affected_segments = []
    for cycle_idx in range(num_cycles):
        start = cycle_idx * period_int + segment_offset
        end = start + segment_length
        modulated_noise[start:end] *= multiplier
        affected_segments.append((int(start), int(end)))

    metadata = {
        "enabled": True,
        "period": int(period_int),
        "num_cycles": int(num_cycles),
        "segment_offset": int(segment_offset),
        "segment_length": int(segment_length),
        "multiplier": round(multiplier, 3),
        "count": len(affected_segments),
        "segments": affected_segments,
        "mode": "amplify" if multiplier > 1.0 else ("shrink" if multiplier < 1.0 else "unchanged"),
    }
    return modulated_noise, metadata


def generate_noise(attribute_pool, y, overall_amplitude_, seq_len):
    max_change = np.abs(np.max(y) - np.min(y))
    noise_level = attribute_pool["noise"]['type']
    if 'amplitude' in attribute_pool['seasonal'] and attribute_pool['seasonal']['amplitude'] != 0.0:
        overall_amplitude = 5 * attribute_pool['seasonal']['amplitude']
    elif 'amplitude' in attribute_pool['trend'] and attribute_pool['trend']['amplitude'] != 0.0:
        overall_amplitude = 5 * attribute_pool['trend']['amplitude']
    else:
        overall_amplitude = overall_amplitude_
    overall_amplitude = np.abs(random.normalvariate(overall_amplitude, overall_amplitude / 5))

    if noise_level == "almost no noise":
        std = 0.003 * overall_amplitude
        noise = np.random.normal(0, std, seq_len)
        attribute_pool["noise"]["std"] = round(std, 3)
        attribute_pool["noise"]["detail"] = (
            f"The overall noise standard deviation is around {std:.2f}, "
            "very small compared to the overall change of the curve. "
            "The curve is overall smooth with almost no noise."
        )
    elif noise_level == "low noise":
        std = 0.01 * overall_amplitude
        noise = np.random.normal(0, std, seq_len)
        attribute_pool["noise"]["std"] = round(std, 3)
        attribute_pool["noise"]["detail"] = (
            f"There is low random noise with standard deviation {std:.2f}."
        )
    elif noise_level in ["moderate noise", "high noise"]:
        if noise_level == "moderate noise":
            base_std = 0.02 * overall_amplitude
            num_segments_max = 2
            amp_range = (1.0, 3.0)
        elif noise_level == "high noise":
            base_std = 0.03 * overall_amplitude
            num_segments_max = 3
            amp_range = (1.0, 5.0)

        # 对于 "high noise"，在特定条件下生成正弦噪音
        if (noise_level == "high noise" and random.random() > 0.5 and
                max_change > overall_amplitude / 2 and
                attribute_pool["frequency"]['type'] == "no periodicity"):
            noise = generate_sin_noise(0.2 * overall_amplitude, seq_len)
            noise += np.random.normal(0, 0.01 * overall_amplitude, seq_len)
            std = round(float(np.std(noise)), 3)
            attribute_pool["noise"]["detail"] = (
                "There is an irregular fluctuating noise, indicating a noisy curve: "
            )
        else:
            std = base_std
            noise = np.random.normal(0, std, seq_len)
            attribute_pool["noise"]["detail"] = (
                f"There is random noise with base standard deviation {std:.2f}, "
                f"indicating {noise_level}: "
            )

        # 为 "moderate noise" 和 "high noise" 应用噪音片段
        num_noise_segments = random.randint(1, num_segments_max)
        attribute_pool["noise"]["segments"] = []
        noise_segments = generate_split_points(seq_len, num_noise_segments)
        for i in range(num_noise_segments):
            noise_start = noise_segments[i]
            noise_end = noise_segments[i + 1]
            noise_std_amp = np.random.uniform(*amp_range)
            noise[noise_start:noise_end] *= noise_std_amp
            segment_std = noise_std_amp * std
            attribute_pool["noise"]["segments"].append({
                "position_start": noise_start,
                "position_end": noise_end,
                "amplitude": round(segment_std, 2),
                "description": (
                    f"the noise std is {segment_std:.2f} "
                    f"between point {noise_start} and point {noise_end}"
                )
            })
            attribute_pool["noise"]["detail"] += (
                f"the noise std is {segment_std:.2f} "
                f"between point {noise_start} and point {noise_end}, "
            )
        attribute_pool["noise"]["detail"] = attribute_pool["noise"]["detail"][:-2] + ". "
    else:
        raise ValueError("Unknown noise type")

    noise, modulation_metadata = apply_background_periodic_noise_modulation(noise, attribute_pool)
    attribute_pool["background_periodic_noise_modulation"] = modulation_metadata
    if modulation_metadata["enabled"]:
        attribute_pool["noise"]["detail"] += (
            f" A periodic noise modulation repeats every {modulation_metadata['period']} points, "
            f"affecting {modulation_metadata['segment_length']} points per cycle "
            f"with multiplier {modulation_metadata['multiplier']:.2f}."
        )
    else:
        attribute_pool["background_periodic_noise_modulation"] = {
            "enabled": False,
            "count": 0
        }

    return noise


# def find_high_variance_subinterval(data, start, end, sub_len, window_size=3, variance_threshold=0.01):
#     candidate_starts = []
#     for i in range(start, end - sub_len + 1):
#         sub_interval = data[i:i + sub_len]
#         is_valid = True
#         for j in range(0, sub_len - window_size + 1):
#             window = sub_interval[j:j + window_size]
#             var = np.mean(window ** 2)
#             if var <= variance_threshold:
#                 is_valid = False
#                 break
#         if is_valid:
#             candidate_starts.append(i)

#     if candidate_starts:
#         return random.choice(candidate_starts), sub_len
#     else:
#         return find_high_variance_subinterval(data, start, end, int(sub_len * 0.9))


def find_high_variance_subinterval(data, start, end, sub_len, window_size=3, variance_threshold=0.01):
    if sub_len <= 0:
        return start, 1
    
    candidate_starts = []
    
    # Vectorized computation for all possible subintervals
    for i in range(start, end - sub_len + 1):
        sub_interval = data[i:i + sub_len]
        
        # Vectorized sliding window variance computation
        if len(sub_interval) < window_size:
            continue
            
        # Create sliding windows using array slicing
        windows = np.array([sub_interval[j:j + window_size] for j in range(len(sub_interval) - window_size + 1)])
        
        # Compute variance for all windows at once
        window_vars = np.mean(windows ** 2, axis=1)
        
        # Check if all windows meet the variance threshold
        if np.all(window_vars > variance_threshold):
            candidate_starts.append(i)
    
    if candidate_starts:
        return random.choice(candidate_starts), sub_len
    else:
        if sub_len > 1:
            return find_high_variance_subinterval(data, start, end, int(sub_len * 0.9), window_size, variance_threshold)
        else:
            return start, 1

def rand_union(a: float, b: float, c: float, d: float) -> float:
    len1 = b - a
    len2 = d - c
    total = len1 + len2
    r = random.random() * total
    if r < len1:
        return a + r
    else:
        return c + (r - len1)

def generate_random_intervals(n: int, num_generated: Optional[int]=None, min_len: int=1) -> list[tuple[int, int]]:
    """ Generate random intervals within a range of n.
    Args:
        n (int): The upper limit of the range.
        num_generated (int, optional): The number of intervals to generate. Defaults to None, which generates a random number of intervals.
        min_len (int, optional): The minimum length of each interval. Defaults to 1.
    Returns:
        list[tuple[int, int]]: A list of tuples representing the start index and length of each interval.
    """

    assert n >= 0, "n must be non-negative"

    if n < 1:
        return []
    
    # Ensure min_len is valid
    min_len = max(1, min(min_len, n))

    if num_generated is None:
        num_tuples = random.randint(1, 2)
    else:
        num_tuples = num_generated

    # If min_len covers the full range, return a single interval.
    if min_len >= n:
        return [(0, n)] if num_tuples >= 1 else []
    
    max_attempts = 20 
    for _ in range(max_attempts):
        # Allow max_len to be at least min_len, but cap at n
        max_len = max(n // 4 + 1, min_len)
        if max_len > n:
            max_len = n
            
        intervals = []
        occupied = [False] * n
        total_len = 0
        valid_generation = True

        for _ in range(num_tuples):
            scale_factor = 2
            # Generate length in range [min_len, max_len]
            # Using the same distribution shape but shifted
            range_size = max_len - min_len + 1
            rand_offset = int(np.floor((random.random()**scale_factor) * range_size))
            len_sample = min_len + rand_offset
            
            length = min(len_sample, max_len)

            if total_len + length >= n:
                valid_generation = False
                break
            
            possible_k = [
                i for i in range(n - length + 1) 
                if not any(occupied[i : i + length])
            ]

            if not possible_k:
                valid_generation = False
                break

            k = random.choice(possible_k)

            if k + length > n or any(occupied[k : k + length]):
                valid_generation = False
                break
            
            for i in range(k, k + length):
                occupied[i] = True
            
            intervals.append((k, length))
            total_len += length

        if valid_generation and total_len < n:
            return intervals

    return []


def generate_seasonal(attribute_pool, overall_amplitude, seq_len, num_seasonal_anomalies=None):
    y_normal = np.zeros(seq_len)
    y_abnormal = np.zeros(seq_len)
    if "no period" not in attribute_pool["seasonal"]['type']:
        # Determine wave type
        if attribute_pool["seasonal"]['type'] == "periodic fluctuation":
            wave_type = None
        else:
            wave_type = attribute_pool["seasonal"]["type"].split(" ")[0]

        # Set amplitude and split points for the entire sequence
        if 'amplitude' not in attribute_pool['seasonal']:
            num_seasonal = random.randint(1, 3) if 'ENABLE_MULTIPLE_SEASONAL' in globals() and ENABLE_MULTIPLE_SEASONAL else 1
            amp = [random.uniform(1.0, 2.0) * overall_amplitude for _ in range(num_seasonal)]
            split_points = sorted([0] + [random.randint(0, seq_len) for _ in range(num_seasonal - 1)] + [seq_len])
        else:
            amp = [attribute_pool['seasonal']['amplitude']]
            split_points = [0, seq_len]

        # Generate normal seasonal wave for the entire sequence
        period = attribute_pool['frequency']['period']
        normal_wave, params_dict = generate_seasonal_wave(
            period, amp, split_points, seq_len, wave_type
        )

        y_normal = copy.deepcopy(normal_wave)
        y_abnormal = copy.deepcopy(normal_wave)

        # Determine anomaly periods
        num_cycles = int(np.floor(seq_len / period))
        min_points = max(1, seq_len // 30)

        # Guard against invalid period
        if period <= 0:
            return y_normal, y_abnormal

        # Calculate minimum cycles needed to cover at least min_points
        min_cycles_needed = max(1, int(np.ceil(min_points / period)))

        if num_cycles < min_cycles_needed:
            anomaly_cycle_indices = []
        else:
            anomaly_cycle_indices = generate_random_intervals(
                num_cycles, num_seasonal_anomalies, min_len=min_cycles_needed
            )

        # Prepare anomaly type selection based on wave_type
        select_anomaly_type = None
        if "seasonal_anomalies" in attribute_pool and attribute_pool["seasonal_anomalies"]:
            print("Here are the available seasonal anomaly types:")
            print(next(iter(attribute_pool["seasonal_anomalies"][0].keys())))
            # sys.exit()
            select_anomaly_type = next(iter(attribute_pool["seasonal_anomalies"][0].keys()))
            select_anomaly_type = next(iter(attribute_pool["seasonal_anomalies"][0][select_anomaly_type].keys()))
        attribute_pool["seasonal_anomalies"] = []

        valid_anomalies = all_attribute_set["seasonal_anomalies"]["common"].copy()
        if wave_type in all_attribute_set["seasonal_anomalies"]:
            valid_anomalies.update(all_attribute_set["seasonal_anomalies"][wave_type])
        anomaly_types = list(valid_anomalies.keys())
        anomaly_weights = list(valid_anomalies.values())
        total_weight = sum(anomaly_weights)
        anomaly_probs = [w / total_weight for w in anomaly_weights]

        for k, len_ in anomaly_cycle_indices:
            start = int(k * period)
            end = min(int((k + len_) * period), seq_len)
            segment_length = end - start
            segment_idx = next((i for i, sp in enumerate(split_points[1:]) if sp > start), len(amp) - 1)
            cur_amp = amp[segment_idx]

            # Select anomaly subinterval length
            upper_bound = max(1, int(period * len_))
            if upper_bound < min_points or segment_length < min_points:
                continue

            lower_bound = max(1, int(0.2 * period * len_), min_points)
            if lower_bound > upper_bound:
                lower_bound = upper_bound
            
            anomaly_len = random.randint(lower_bound, upper_bound)

            if anomaly_len > segment_length:
                anomaly_len = segment_length

            anomaly_start_idx, anomaly_len = find_high_variance_subinterval(normal_wave, start, end, anomaly_len)
            if anomaly_len < min_points:
                anomaly_len = min_points
                anomaly_start_idx = random.randint(start, end - min_points)
            anomaly_end_idx = min(anomaly_start_idx + anomaly_len, seq_len)

            # Select anomaly type
            anomaly_type = select_anomaly_type if select_anomaly_type else str(
                np.random.choice(anomaly_types, p=anomaly_probs)
            )
            if anomaly_type == "none":
                continue

            # Copy params_dict for modification
            anomaly_params_dict = copy.deepcopy(params_dict)

            # Generate anomaly waveform for the entire cycle, apply to subinterval
            if anomaly_type == "waveform_change" and wave_type == 'sin':
                anomaly_wave_type = random.choice(["square", "triangle"])
                anomaly_wave, _ = generate_seasonal_wave(
                    period, [cur_amp], [0, segment_length], segment_length, anomaly_wave_type
                )
                anomaly_details = f"A seasonal waveform change anomaly, with the sine wave transformed to a {anomaly_wave_type} wave in the subinterval"

            elif anomaly_type == "frequency_change":
                anomaly_period = rand_union(0.2, 0.6, 1.5, 3.0) * period
                anomaly_wave, _ = generate_seasonal_wave(
                    anomaly_period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                )
                anomaly_details = f"A seasonal period change anomaly, with the period adjusted to {anomaly_period:.1f} in the subinterval"

            elif anomaly_type == "phase_shift" and wave_type == 'sin':
                phase_shift = random.uniform(0, 2 * np.pi)
                while abs(phase_shift) < 0.5 or abs(phase_shift - 2 * np.pi) < 0.5:  # Ensure at least 0.5 radian change
                    phase_shift = random.uniform(0, 2 * np.pi)
                for harmonic in anomaly_params_dict['harmonics']:
                    harmonic['phase'] += phase_shift
                anomaly_wave, _ = generate_seasonal_wave(
                    period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                )
                anomaly_details = f"A seasonal phase shift anomaly, with a shift of {phase_shift:.2f} radians applied in the subinterval"

            elif anomaly_type == "add_harmonic" and wave_type == 'sin':
                if len(anomaly_params_dict['harmonics']) < max(1, min(int(period) // 6, 10)):
                    new_harmonic = {
                        'phase': random.uniform(0, 2 * np.pi),
                        'amp_mod_depth': random.uniform(0, 0.05),
                        'mod_freq_factor': random.uniform(1, 3),
                        'mod_phase': random.uniform(0, 2 * np.pi)
                    }
                    anomaly_params_dict['harmonics'].append(new_harmonic)
                    anomaly_params_dict['num_harmonics'] += 1
                    anomaly_wave, _ = generate_seasonal_wave(
                        period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                    )
                    anomaly_details = "A seasonal harmonic addition anomaly, with an additional harmonic introduced in the subinterval"
                else:
                    anomaly_wave = normal_wave[start:end]
                    anomaly_details = "An attempted seasonal harmonic addition anomaly, with no change due to the maximum harmonic count being reached"

            elif anomaly_type == "remove_harmonic" and wave_type == 'sin':
                if len(anomaly_params_dict['harmonics']) > 1:
                    # Prioritize removing lower-order harmonics (which have higher amplitude 1/n)
                    # Pick from the first min(3, len) harmonics
                    limit = min(3, len(anomaly_params_dict['harmonics']))
                    remove_idx = random.randint(0, limit - 1)
                    
                    del anomaly_params_dict['harmonics'][remove_idx]
                    anomaly_params_dict['num_harmonics'] -= 1
                    anomaly_wave, _ = generate_seasonal_wave(
                        period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                    )
                    anomaly_details = "A seasonal harmonic removal anomaly, with a significant harmonic omitted in the subinterval"
                else:
                    anomaly_wave = normal_wave[start:end]
                    anomaly_details = "An attempted seasonal harmonic removal anomaly, with no change as only one harmonic remains"

            elif anomaly_type == "modify_harmonic_phase" and wave_type == 'sin':
                if anomaly_params_dict['harmonics']:
                    harmonic = random.choice(anomaly_params_dict['harmonics'])
                    original_phase = harmonic['phase']
                    harmonic['phase'] = random.uniform(0, 2 * np.pi)
                    while abs(harmonic['phase'] - original_phase) < 1.5:
                        harmonic['phase'] = random.uniform(0, 2 * np.pi)
                    anomaly_wave, _ = generate_seasonal_wave(
                        period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                    )
                    anomaly_details = "A seasonal harmonic phase anomaly, with the phase of a harmonic altered in the subinterval"

            elif anomaly_type == "modify_harmonic_amp_mod" and wave_type == 'sin':
                if anomaly_params_dict['harmonics']:
                    harmonic = random.choice(anomaly_params_dict['harmonics'])
                    original_amp_mod_depth = harmonic['amp_mod_depth']
                    # Significantly increase modulation depth to make it visible (0.2 to 0.5)
                    harmonic['amp_mod_depth'] = random.uniform(0.2, 0.5)
                    anomaly_wave, _ = generate_seasonal_wave(
                        period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                    )
                    anomaly_details = "A seasonal harmonic amplitude modulation anomaly, with the modulation depth of a harmonic adjusted in the subinterval"

            elif anomaly_type == "modify_harmonic_mod_freq" and wave_type == 'sin':
                if anomaly_params_dict['harmonics']:
                    harmonic = random.choice(anomaly_params_dict['harmonics'])
                    
                    # Ensure modulation is visible
                    harmonic['amp_mod_depth'] = random.uniform(0.3, 0.6)
                    
                    original_mod_freq_factor = harmonic['mod_freq_factor']
                    harmonic['mod_freq_factor'] = random.uniform(1, 3)
                    while abs(harmonic['mod_freq_factor'] - original_mod_freq_factor) < 0.5:
                        harmonic['mod_freq_factor'] = random.uniform(1, 3)
                    anomaly_wave, _ = generate_seasonal_wave(
                        period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                    )
                    anomaly_details = "A seasonal harmonic modulation frequency anomaly, with the modulation frequency factor of a harmonic changed in the subinterval"

            elif anomaly_type == "modify_harmonic_mod_phase" and wave_type == 'sin':
                if anomaly_params_dict['harmonics']:
                    harmonic = random.choice(anomaly_params_dict['harmonics'])
                    
                    # Ensure modulation is visible
                    harmonic['amp_mod_depth'] = random.uniform(0.3, 0.6)
                    
                    original_mod_phase = harmonic['mod_phase']
                    harmonic['mod_phase'] = random.uniform(0, 2 * np.pi)
                    while abs(harmonic['mod_phase'] - original_mod_phase) < 0.1:
                        harmonic['mod_phase'] = random.uniform(0, 2 * np.pi)
                    anomaly_wave, _ = generate_seasonal_wave(
                        period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                    )
                    anomaly_details = "A seasonal harmonic modulation phase anomaly, with the modulation phase of a harmonic modified in the subinterval"

            elif anomaly_type == "pulse_shift" and wave_type in ['square', 'triangle']:
                shift = rand_union(-0.3, -0.1, 0.1, 0.3)
                anomaly_params_dict['start'] = (anomaly_params_dict['start'] + shift) % 1
                anomaly_wave, _ = generate_seasonal_wave(
                    period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                )
                anomaly_details = f"A seasonal pulse shift anomaly, with the pulse starting point shifted by {shift:.2f} in the subinterval"

            elif anomaly_type == "pulse_width_modulation" and wave_type in ['square', 'triangle']:
                scale = rand_union(0.5, 0.8, 1.4, 2.0)
                new_duration = anomaly_params_dict['duration'] * scale
                max_duration = 1 - anomaly_params_dict['start']
                anomaly_params_dict['duration'] = min(new_duration, max_duration)
                anomaly_wave, _ = generate_seasonal_wave(
                    period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                )
                anomaly_details = f"A seasonal pulse width anomaly, with the pulse width scaled by a factor of {scale:.2f} in the subinterval"

            elif anomaly_type == "family_change" and wave_type == 'wavelet':
                all_wavelet_families = ['db2', 'db4', 'sym2', 'sym5', 'coif1', 'coif3', 'haar', 'bior1.3', 'dmey']
                current_families = [comp['family'] for comp in anomaly_params_dict['wavelet_components']]
                new_family = random.choice([f for f in all_wavelet_families if f not in current_families] or all_wavelet_families)
                for comp in anomaly_params_dict['wavelet_components']:
                    comp['family'] = new_family
                anomaly_wave, _ = generate_seasonal_wave(
                    period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                )
                anomaly_details = f"A seasonal wavelet family anomaly, with the wavelet family changed to {new_family} in the subinterval"

            elif anomaly_type == "scale_change" and wave_type == 'wavelet':
                for comp in anomaly_params_dict['wavelet_components']:
                    for wp in comp['wavelets']:
                        original_s = wp['s']
                        wp['s'] *= rand_union(0.5, 0.8, 1.4, 2.0)
                        while abs(wp['s'] - original_s) < 0.2 * original_s:
                            wp['s'] = original_s * rand_union(0.5, 0.8, 1.4, 2.0)
                anomaly_wave, _ = generate_seasonal_wave(
                    period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                )
                anomaly_details = "A seasonal wavelet scale anomaly, with the scales of wavelets modified in the subinterval"

            elif anomaly_type == "shift_change" and wave_type == 'wavelet':
                for comp in anomaly_params_dict['wavelet_components']:
                    for wp in comp['wavelets']:
                        original_tau = wp['tau']
                        wp['tau'] = (wp['tau'] + random.uniform(-period/4, period/4)) % period
                        while abs(wp['tau'] - original_tau) < 0.2 * period:
                            wp['tau'] = (original_tau + random.uniform(-period/4, period/4)) % period
                anomaly_wave, _ = generate_seasonal_wave(
                    period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                )
                anomaly_details = "A seasonal wavelet shift anomaly, with the shifts of wavelets altered in the subinterval"

            elif anomaly_type == "amplitude_change" and wave_type == 'wavelet':
                for comp in anomaly_params_dict['wavelet_components']:
                    for wp in comp['wavelets']:
                        original_A = wp['A']
                        wp['A'] *= rand_union(0.2, 0.6, 1.5, 3.0)
                        while abs(wp['A'] - original_A) < 0.2 * original_A:
                            wp['A'] = original_A * rand_union(0.2, 0.6, 1.5, 3.0)
                anomaly_wave, _ = generate_seasonal_wave(
                    period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                )
                anomaly_details = "A seasonal wavelet amplitude anomaly, with the amplitudes of wavelets adjusted in the subinterval"

            elif anomaly_type == "add_wavelet" and wave_type == 'wavelet':
                all_wavelet_families = ['db2', 'db4', 'sym2', 'sym5', 'coif1', 'coif3', 'haar', 'bior1.3', 'dmey']
                new_family = random.choice(all_wavelet_families)
                new_wavelets = [{
                    's': random.uniform(period / 4, period / 2),
                    'tau': random.uniform(0, period),
                    'A': random.uniform(0.8, 1.5)
                }]
                anomaly_params_dict['wavelet_components'].append({'family': new_family, 'wavelets': new_wavelets})
                anomaly_wave, _ = generate_seasonal_wave(
                    period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                )
                anomaly_details = "A seasonal wavelet addition anomaly, with a new wavelet component added in the subinterval"

            elif anomaly_type == "remove_wavelet" and wave_type == 'wavelet':
                if len(anomaly_params_dict['wavelet_components']) > 1:
                    # Find the component with the highest energy contribution to remove
                    max_energy = -1
                    remove_idx = 0
                    for idx, comp in enumerate(anomaly_params_dict['wavelet_components']):
                        # Estimate energy as sum of A/sqrt(s)
                        energy = sum(w['A'] / np.sqrt(w['s']) for w in comp['wavelets'])
                        if energy > max_energy:
                            max_energy = energy
                            remove_idx = idx
                            
                    del anomaly_params_dict['wavelet_components'][remove_idx]
                    anomaly_wave, _ = generate_seasonal_wave(
                        period, [cur_amp], [0, segment_length], segment_length, wave_type, anomaly_params_dict
                    )
                    anomaly_details = "A seasonal wavelet removal anomaly, with the most significant wavelet component removed in the subinterval"
                else:
                    anomaly_wave = normal_wave[start:end]
                    anomaly_details = "An attempted seasonal wavelet removal anomaly, with no change as only one wavelet component remains"

            elif anomaly_type == "waveform_inversion":
                anomaly_wave = -normal_wave[start:end]
                anomaly_details = "A seasonal waveform inversion anomaly, with the waveform inverted in the subinterval"

            elif anomaly_type == "amplitude_scaling":
                scale_factor = rand_union(0.5, 0.8, 1.4, 2.0)
                anomaly_wave = normal_wave[start:end] * scale_factor
                anomaly_details = f"A seasonal amplitude scaling anomaly, with the amplitude scaled by {scale_factor:.2f} in the subinterval"

            elif anomaly_type == "noise_injection":
                noise_level = random.uniform(0.1, 0.2) * cur_amp
                noise = np.random.normal(0, noise_level, segment_length)
                anomaly_wave = normal_wave[start:end] + noise
                anomaly_details = f"A seasonal noise injection anomaly, with noise of level {noise_level:.2f} added in the subinterval"

            # Apply anomaly to subinterval
            y_abnormal[anomaly_start_idx:anomaly_end_idx] = anomaly_wave[anomaly_start_idx - start:anomaly_end_idx - start]
            # Record anomaly details
            attribute_pool["seasonal_anomalies"].append({
                "type": anomaly_type,
                "start": anomaly_start_idx,
                "end": anomaly_end_idx,
                "indice": (k, len_),
                "details": anomaly_details
            })

        # Update seasonal description
        attribute_pool["seasonal"]['detail'] = f"Time series shows {attribute_pool['seasonal']['type']}: "
        attribute_pool["seasonal"]["segments"] = []
        for i in range(len(amp)):
            segment_info = {
                "amplitude": round(amp[i], 2),
                "position_start": split_points[i],
                "position_end": split_points[i + 1],
                "description": f"Periodic fluctuation with amplitude {amp[i]:.1f}, from point {split_points[i]} to point {split_points[i + 1]}"
            }
            attribute_pool["seasonal"]["segments"].append(segment_info)
            attribute_pool["seasonal"]['detail'] += f"Periodic fluctuation with amplitude {amp[i]:.1f}, from point {split_points[i]} to point {split_points[i + 1]}, "
        attribute_pool["seasonal"]['detail'] = attribute_pool["seasonal"]['detail'][:-2] + "."
    else:
        attribute_pool["seasonal"]["segments"] = []
        attribute_pool["seasonal"]['detail'] = f"No periodic fluctuations observed, showing {attribute_pool['seasonal']['type']}."
    return y_normal, y_abnormal

def get_max_anomaly_length(anomaly_type: str, seq_len: int) -> int:
    """
    根据异常类型返回该类型的最大可能长度。
    
    Parameters:
    -----------
    anomaly_type : str
        异常类型名称
    seq_len : int
        序列总长度
    
    Returns:
    --------
    int
        该异常类型的最大可能长度
    """
    # 基于代码中各异常类型的实际长度计算
    length_config = {
        # 单点/短尖峰类 (长度固定或很短)
        "outlier": 1,
        "upward spike": 8,  # rise(3) + fall(3) + peak(2) 最大
        "downward spike": 8,
        
        # 连续尖峰类 (多个尖峰叠加)
        "continuous upward spike": 5 * 8 + 5 * 5,  # 5个尖峰 + 间隔
        "continuous downward spike": 5 * 8 + 5 * 5,
        
        # 宽尖峰类 (基于 seq_len 的比例)
        "wide upward spike": int(seq_len * 0.08) * 2 + 4,  # rise + fall + peak
        "wide downward spike": int(seq_len * 0.08) * 2 + 4,
        
        # 凸起类 (最长)
        "upward convex": int(seq_len * 0.2) + 10,  # convex_length + start/end
        "downward convex": int(seq_len * 0.2) + 10,
        
        # 震荡类
        "shake": int(seq_len * 0.15),
        
        # 持续性变化类 (影响到序列末尾，但异常区间本身较短)
        "sudden increase": 20,  # drop_length + recover_length
        "sudden decrease": 20,
        
        # 组合变化类
        "rapid rise followed by slow decline": int(seq_len * 0.15) + 5,
        "slow rise followed by rapid decline": int(seq_len * 0.15) + 5,
        "rapid decline followed by slow rise": int(seq_len * 0.15) + 5,
        "slow decline followed by rapid rise": int(seq_len * 0.15) + 5,
        
        # 尖峰后变化类
        "decrease after upward spike": 8 + int(seq_len * 0.05),
        "increase after downward spike": 8 + int(seq_len * 0.05),
        "increase after upward spike": 8 + int(seq_len * 0.05),
        "decrease after downward spike": 8 + int(seq_len * 0.05),
    }
    
    # 默认保守估计
    return length_config.get(anomaly_type, int(seq_len * 0.2))


def generate_local_chars(attribute_pool, overall_amplitude, seq_len, contrast=True, is_multi=False):
    """
    Generate a time series with local characteristics based on the given attribute pool.
    Parameters:
    attribute_pool (dict): A dictionary containing local characteristics to be applied to the time series.
                           Each local characteristic should have a "type", "position_start", and "amplitude".
    overall_amplitude (float): The overall amplitude to scale the local characteristics.
    seq_len (int): The length of the time series to be generated.
    Returns:
    tuple: (y_normal, y_abnormal) where y_normal is the normal time series and y_abnormal is the time series with anomalies.
    """
    x = np.arange(seq_len)
    y_normal = np.zeros(seq_len)
    y_abnormal = np.zeros(seq_len)
    level_shifts = []  # 用于记录持续性变化的水平移位事件

    # 应用局部特征
    for local_char in attribute_pool["local"]:
        local_position = local_char["position_start"]
        
        # 1. Determine Base Scale (Priority: Seasonal > Trend > Overall)
        if 'amplitude' in attribute_pool['seasonal'] and attribute_pool['seasonal']['amplitude'] != 0.0:
            base_scale = attribute_pool['seasonal']['amplitude']
        elif 'amplitude' in attribute_pool['trend'] and attribute_pool['trend']['amplitude'] != 0.0:
            base_scale = attribute_pool['trend']['amplitude']
        else:
            base_scale = overall_amplitude

        # 2. Determine Local Amplitude
        local_amplitude = local_char.get("amplitude")

        if local_amplitude is None:
            # Calculate based on anomaly type groups
            anomaly_type = local_char["type"]
            
            if anomaly_type == "outlier":
                if random.random() < 0.05:
                    multiplier = random.uniform(10.0, 20.0)
                else:
                    multiplier = 1.0 + np.random.exponential(scale=2.0)
            elif anomaly_type in ["upward spike", "downward spike", "continuous upward spike", "continuous downward spike", "wide upward spike", "wide downward spike"]:
                multiplier = 0.5 + np.random.exponential(scale=1.0)
            else:
                multiplier = 0.3 + np.random.exponential(scale=0.3)
            
            local_amplitude = multiplier * base_scale
            local_amplitude = np.abs(random.normalvariate(local_amplitude, local_amplitude / 5))

        # 3. SNR Guarantee: Ensure anomaly is visible above noise floor
        noise_std_est = 0.0
        if "noise" in attribute_pool:
            if "std" in attribute_pool["noise"]:
                noise_std_est = attribute_pool["noise"]["std"]
            else:
                # Estimate noise std based on noise type (reference: 5 * base_scale)
                noise_scale_ref = 5.0 * base_scale
                
                noise_type = attribute_pool["noise"].get("type", "low noise")
                if noise_type == "almost no noise":
                    noise_std_est =  0.003 * noise_scale_ref
                elif noise_type == "low noise":
                    noise_std_est = 0.01 * noise_scale_ref
                elif noise_type == "moderate noise":
                    noise_std_est = 0.02 * noise_scale_ref * 3.0
                elif noise_type == "high noise":
                    noise_std_est = 0.03 * noise_scale_ref * 5.0
        
        # Apply minimum amplitude threshold (K * noise_std)
        high_visibility_types = ["outlier", "upward spike", "downward spike", "continuous upward spike", "continuous downward spike", "wide upward spike", "wide downward spike"]
        sigma_multiplier = 6.0 if local_char["type"] in high_visibility_types else 3.0
        min_amplitude = sigma_multiplier * noise_std_est

        if local_amplitude < min_amplitude:
            local_amplitude = min_amplitude

        # 4. Multivariate Scaling
        if is_multi:
            local_amplitude *= 2

        if local_position is None:
            # 动态计算位置上限，根据异常类型的最大长度确定
            max_anomaly_len = get_max_anomaly_length(local_char["type"], seq_len)
            position_upper_bound = max(1, seq_len - max_anomaly_len)
            
            # 动态调整最小间距：如果异常数量多，则允许间距更小
            num_anomalies = len(attribute_pool['local'])
            if num_anomalies <= 5:
                min_dist = seq_len / 10.0
            else:
                min_dist = max(max_anomaly_len * 1.2, seq_len // 100)
            
            while True:
                local_position = random.randint(1, position_upper_bound)
                if all([abs(local_position - i['position_start']) > min_dist for i in attribute_pool['local'] if i['position_start'] is not None]):
                    local_char['position_start'] = local_position
                    break

        # 处理 "shake" 类型（非持续性）
        if local_char["type"] == "shake":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            local_amplitude *= 0.2
            peak_start = local_position
            peak_length = random.randint(6, int(seq_len * 0.15))
            func = random.choice([
                lambda x: np.random.uniform(-1, 1, peak_length) * local_amplitude / 2,
                lambda x: np.sin(x[peak_start:peak_start + peak_length] * 5.0) * local_amplitude / 2
            ])
            y_abnormal[peak_start:peak_start + peak_length] += func(x)
            local_char["detail"] = f"A local shake anomaly, with rapid fluctuations of amplitude around {local_amplitude:.2f}"
            local_char['position_end'] = peak_start + peak_length

        # 处理 "upward spike" 类型（非持续性）
        elif local_char["type"] == "upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            local_amplitude *= 1.0
            peak_start = local_position
            spike = generate_spike(local_amplitude)
            peak_length = len(spike)
            y_abnormal[peak_start:peak_start + peak_length] += spike
            local_char["position_end"] = peak_start + peak_length
            local_char["detail"] = f"A local upward spike anomaly, with a sharp rise to amplitude {local_amplitude:.2f}"

        # 处理 "downward spike" 类型（非持续性）
        elif local_char["type"] == "downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            local_amplitude *= 1.0
            peak_start = local_position
            spike = generate_spike(-local_amplitude)
            peak_length = len(spike)
            y_abnormal[peak_start:peak_start + peak_length] += spike
            local_char["position_end"] = peak_start + peak_length
            local_char["detail"] = f"A local downward spike anomaly, with a sharp drop to amplitude {local_amplitude:.2f}"

        # 处理 "continuous upward spike" 类型（非持续性）
        elif local_char["type"] == "continuous upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            local_amplitude *= 0.8
            peak_region_start = local_position
            num_peaks = random.randint(2, 5)
            peaks = []
            all_amplitudes = []
            for _ in range(num_peaks):
                peak_start = random.randint(peak_region_start + 0, peak_region_start + 5)
                cur_amplitude = random.uniform(local_amplitude * 0.6, local_amplitude * 1.5)
                all_amplitudes.append(cur_amplitude)
                peaks.append(f"point {peak_start}")
                spike = generate_spike(cur_amplitude)
                peak_length = len(spike)
                y_abnormal[peak_start:peak_start + peak_length] += spike
                peak_region_start = peak_start + peak_length
            local_char["position_end"] = peak_start + peak_length
            local_amplitude = float(np.mean(all_amplitudes))
            local_char["detail"] = f"A local continuous upward spike anomaly, featuring {num_peaks} consecutive spikes with amplitudes from {min(all_amplitudes):.2f} to {max(all_amplitudes):.2f}"

        # 处理 "continuous downward spike" 类型（非持续性）
        elif local_char["type"] == "continuous downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            local_amplitude *= 0.8
            peak_region_start = local_position
            num_peaks = random.randint(2, 5)
            peaks = []
            all_amplitudes = []
            for _ in range(num_peaks):
                peak_start = random.randint(peak_region_start + 0, peak_region_start + 5)
                cur_amplitude = random.uniform(local_amplitude * 0.6, local_amplitude * 1.5)
                all_amplitudes.append(cur_amplitude)
                peaks.append(f"point {peak_start}")
                spike = generate_spike(-cur_amplitude)
                peak_length = len(spike)
                y_abnormal[peak_start:peak_start + peak_length] += spike
                peak_region_start = peak_start + peak_length
            local_char["position_end"] = peak_start + peak_length
            local_amplitude = float(np.mean(all_amplitudes))
            local_char["detail"] = f"A local continuous downward spike anomaly, featuring {num_peaks} consecutive spikes with amplitudes from {min(all_amplitudes):.2f} to {max(all_amplitudes):.2f}"

        # 处理 "upward convex" 类型（非持续性）
        elif local_char["type"] == "upward convex":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            convex_start = local_position
            start_length, end_length = random.randint(1, 5), random.randint(1, 5)
            # may exceed the sequence length
            convex_length = random.randint(int(seq_len * 0.03), int(seq_len * 0.2))
            y_abnormal[convex_start:convex_start + start_length] += generate_ts_change(start_length, local_amplitude)
            y_abnormal[convex_start + start_length:convex_start + start_length + convex_length] += local_amplitude
            # may discrete
            y_abnormal[convex_start + start_length + convex_length:convex_start + start_length + convex_length + end_length] += generate_ts_change(end_length, -local_amplitude) + local_amplitude
            local_char["position_end"] = convex_start + start_length + convex_length + end_length
            if random.random() > 0.7:
                y_abnormal[convex_start + start_length:convex_start + start_length + convex_length] += np.sin((0.8 + np.abs(random.normalvariate(0.0, 2.0))) * x)[convex_start + start_length:convex_start + start_length + convex_length]
            if random.random() > 0.7:
                y_abnormal[convex_start + start_length:convex_start + start_length + convex_length] += np.random.uniform(-1.0, 1.0, convex_length) * np.random.uniform(0.1, 0.5) * local_amplitude
            local_char["detail"] = f"A local upward convex anomaly, with a convex shape reaching height {local_amplitude:.2f}"

        # 处理 "downward convex" 类型（非持续性）
        elif local_char["type"] == "downward convex":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            convex_start = local_position
            start_length, end_length = random.randint(1, 5), random.randint(1, 5)
            convex_length = random.randint(int(seq_len * 0.03), int(seq_len * 0.2))
            y_abnormal[convex_start:convex_start + start_length] += generate_ts_change(start_length, -local_amplitude)
            y_abnormal[convex_start + start_length:convex_start + start_length + convex_length] -= local_amplitude
            # may discrete
            y_abnormal[convex_start + start_length + convex_length:convex_start + start_length + convex_length + end_length] += generate_ts_change(end_length, local_amplitude) - local_amplitude
            local_char["position_end"] = convex_start + start_length + convex_length + end_length
            if random.random() > 0.7:
                y_abnormal[convex_start + start_length:convex_start + start_length + convex_length] += np.sin((0.8 + np.abs(random.normalvariate(0.0, 2.0))) * x)[convex_start + start_length:convex_start + start_length + convex_length]
            if random.random() > 0.7:
                y_abnormal[convex_start + start_length:convex_start + start_length + convex_length] += np.random.uniform(-1.0, 1.0, convex_length) * np.random.uniform(0.1, 0.5) * local_amplitude
            local_char["detail"] = f"A local downward convex anomaly, with a convex shape reaching depth {local_amplitude:.2f}"

        # 处理 "sudden increase" 类型（持续性）
        elif local_char["type"] == "sudden increase":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            drop_length = random.randint(1, 10)
            y_abnormal[local_position:local_position + drop_length] += generate_ts_change(drop_length, local_amplitude)
            y_abnormal[local_position + drop_length:] += local_amplitude
            position_end = local_position + drop_length
            shift = local_amplitude
            if random.random() < 0.5:
                recover_length = random.randint(1, 10)
                recover_amplitude = random.uniform(0, local_amplitude / 3)
                y_abnormal[local_position + drop_length:local_position + drop_length + recover_length] += generate_ts_change(recover_length, -recover_amplitude)
                y_abnormal[local_position + drop_length + recover_length:] -= recover_amplitude
                position_end = local_position + drop_length + recover_length
                shift = local_amplitude - recover_amplitude
            local_char["position_end"] = position_end
            local_char["detail"] = f"A local sudden increase anomaly, with a sharp rise of {local_amplitude:.2f}"
            if 'recover_amplitude' in locals():
                local_char["detail"] += f", followed by a partial decrease of {recover_amplitude:.2f}"
            level_shifts.append((position_end, shift))

        # 处理 "sudden decrease" 类型（持续性）
        elif local_char["type"] == "sudden decrease":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            drop_length = random.randint(1, 10)
            y_abnormal[local_position:local_position + drop_length] += generate_ts_change(drop_length, -local_amplitude)
            y_abnormal[local_position + drop_length:] += -local_amplitude
            position_end = local_position + drop_length
            shift = -local_amplitude
            if random.random() < 0.5:
                recover_length = random.randint(1, 10)
                recover_amplitude = random.uniform(0, local_amplitude / 3)
                y_abnormal[local_position + drop_length:local_position + drop_length + recover_length] += generate_ts_change(recover_length, recover_amplitude)
                y_abnormal[local_position + drop_length + recover_length:] += recover_amplitude
                position_end = local_position + drop_length + recover_length
                shift = -local_amplitude + recover_amplitude
            local_char["position_end"] = position_end
            local_char["detail"] = f"A local sudden decrease anomaly, with a sharp drop of {local_amplitude:.2f}"
            if 'recover_amplitude' in locals():
                local_char["detail"] += f", followed by a partial increase of {recover_amplitude:.2f}"
            level_shifts.append((position_end, shift))

        # 处理 "rapid rise followed by slow decline" 类型（非持续性）
        elif local_char["type"] == "rapid rise followed by slow decline":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            rise_length = random.randint(1, 5)
            fall_length = random.randint(int(seq_len * 0.05), int(seq_len * 0.15))
            y_abnormal[local_position:local_position + rise_length] += generate_ts_change(rise_length, local_amplitude)
            y_abnormal[local_position + rise_length:local_position + rise_length + fall_length] += generate_ts_change(fall_length, -local_amplitude) + local_amplitude
            local_char["position_end"] = local_position + rise_length + fall_length
            local_char["detail"] = f"A local rapid rise followed by slow decline anomaly, with a rise amplitude of {local_amplitude:.2f}"

        # 处理 "slow rise followed by rapid decline" 类型（非持续性）
        elif local_char["type"] == "slow rise followed by rapid decline":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            rise_length = random.randint(int(seq_len * 0.05), int(seq_len * 0.15))
            fall_length = random.randint(1, 5)
            y_abnormal[local_position:local_position + rise_length] += generate_ts_change(rise_length, local_amplitude)
            y_abnormal[local_position + rise_length:local_position + rise_length + fall_length] += generate_ts_change(fall_length, -local_amplitude) + local_amplitude
            local_char["position_end"] = local_position + rise_length + fall_length
            local_char["detail"] = f"A local slow rise followed by rapid decline anomaly, with a rise amplitude of {local_amplitude:.2f}"

        # 处理 "rapid decline followed by slow rise" 类型（非持续性）
        elif local_char["type"] == "rapid decline followed by slow rise":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            drop_length = random.randint(1, 5)
            rise_length = random.randint(int(seq_len * 0.05), int(seq_len * 0.15))
            y_abnormal[local_position:local_position + drop_length] += generate_ts_change(drop_length, -local_amplitude)
            y_abnormal[local_position + drop_length:local_position + drop_length + rise_length] += generate_ts_change(rise_length, local_amplitude) - local_amplitude
            local_char["position_end"] = local_position + drop_length + rise_length
            local_char["detail"] = f"A local rapid decline followed by slow rise anomaly, with a decline amplitude of {local_amplitude:.2f}"

        # 处理 "slow decline followed by rapid rise" 类型（非持续性）
        elif local_char["type"] == "slow decline followed by rapid rise":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            drop_length = random.randint(int(seq_len * 0.05), int(seq_len * 0.15))
            rise_length = random.randint(1, 5)
            y_abnormal[local_position:local_position + drop_length] += generate_ts_change(drop_length, -local_amplitude)
            y_abnormal[local_position + drop_length:local_position + drop_length + rise_length] += generate_ts_change(rise_length, local_amplitude) - local_amplitude
            local_char["position_end"] = local_position + drop_length + rise_length
            local_char["detail"] = f"A local slow decline followed by rapid rise anomaly, with a decline amplitude of {local_amplitude:.2f}"

        # 处理 "decrease after upward spike" 类型（持续性）
        elif local_char["type"] == "decrease after upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            fall_amplitude = random.uniform(0.1, 0.7) * local_amplitude
            peak_start = local_position
            spike = generate_spike(local_amplitude)
            peak_length = len(spike)
            fall_length = random.randint(2, int(seq_len * 0.05))
            y_abnormal[peak_start:peak_start + peak_length] += spike
            y_abnormal[peak_start + peak_length:peak_start + peak_length + fall_length] += generate_ts_change(fall_length, -fall_amplitude)
            y_abnormal[peak_start + peak_length + fall_length:] -= fall_amplitude
            local_char["position_end"] = peak_start + peak_length + fall_length
            local_char["detail"] = f"A local upward spike followed by decrease anomaly, with spike amplitude {local_amplitude:.2f} and subsequent decrease {fall_amplitude:.2f}"
            level_shifts.append((peak_start + peak_length + fall_length, -fall_amplitude))

        # 处理 "increase after downward spike" 类型（持续性）
        elif local_char["type"] == "increase after downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            fall_amplitude = random.uniform(0.1, 0.7) * local_amplitude
            peak_start = local_position
            spike = generate_spike(-local_amplitude)
            peak_length = len(spike)
            rise_length = random.randint(2, int(seq_len * 0.05))
            y_abnormal[peak_start:peak_start + peak_length] += spike
            y_abnormal[peak_start + peak_length:peak_start + peak_length + rise_length] += generate_ts_change(rise_length, fall_amplitude)
            y_abnormal[peak_start + peak_length + rise_length:] += fall_amplitude
            local_char["position_end"] = peak_start + peak_length + rise_length
            local_char["detail"] = f"A local downward spike followed by increase anomaly, with spike amplitude {local_amplitude:.2f} and subsequent increase {fall_amplitude:.2f}"
            level_shifts.append((peak_start + peak_length + rise_length, fall_amplitude))

        # 处理 "increase after upward spike" 类型（持续性）
        elif local_char["type"] == "increase after upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            fall_amplitude = random.uniform(0.1, 0.7) * local_amplitude
            peak_start = local_position
            spike = generate_spike(local_amplitude)
            peak_length = len(spike)
            fall_length = random.randint(2, int(seq_len * 0.05))
            y_abnormal[peak_start:peak_start + peak_length] += spike
            y_abnormal[peak_start + peak_length:peak_start + peak_length + fall_length] += generate_ts_change(fall_length, fall_amplitude)
            y_abnormal[peak_start + peak_length + fall_length:] += fall_amplitude
            local_char["position_end"] = peak_start + peak_length + fall_length
            local_char["detail"] = f"A local upward spike followed by increase anomaly, with spike amplitude {local_amplitude:.2f} and subsequent increase {fall_amplitude:.2f}"
            level_shifts.append((peak_start + peak_length + fall_length, fall_amplitude))

        # 处理 "decrease after downward spike" 类型（持续性）
        elif local_char["type"] == "decrease after downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            fall_amplitude = random.uniform(0.1, 0.7) * local_amplitude
            peak_start = local_position
            spike = generate_spike(-local_amplitude)
            peak_length = len(spike)
            rise_length = random.randint(2, int(seq_len * 0.05))
            y_abnormal[peak_start:peak_start + peak_length] += spike
            y_abnormal[peak_start + peak_length:peak_start + peak_length + rise_length] += generate_ts_change(rise_length, -fall_amplitude)
            y_abnormal[peak_start + peak_length + rise_length:] -= fall_amplitude
            local_char["position_end"] = peak_start + peak_length + rise_length
            local_char["detail"] = f"A local downward spike followed by decrease anomaly, with spike amplitude {local_amplitude:.2f} and subsequent decrease {fall_amplitude:.2f}"
            level_shifts.append((peak_start + peak_length + rise_length, -fall_amplitude))

        # 处理 "wide upward spike" 类型（非持续性）
        elif local_char["type"] == "wide upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            local_amplitude *= 0.8
            rise_length = random.randint(int(seq_len * 0.02), min(int(seq_len * 0.08), 500))
            peak_length = random.randint(1, 4)
            fall_length = random.randint(int(seq_len * 0.02), min(int(seq_len * 0.08), 500))
            y_abnormal[local_position:local_position + rise_length] += generate_ts_change(rise_length, local_amplitude)
            y_abnormal[local_position + rise_length:local_position + rise_length + peak_length] += local_amplitude
            y_abnormal[local_position + rise_length + peak_length:local_position + rise_length + peak_length + fall_length] += generate_ts_change(fall_length, -local_amplitude) + local_amplitude
            local_char["position_end"] = local_position + rise_length + peak_length + fall_length
            local_char["detail"] = f"A local wide upward spike anomaly, with amplitude {local_amplitude:.2f}"

        # 处理 "outlier" 类型（单点异常）
        elif local_char["type"] == "outlier":
            if local_amplitude is None:
                local_amplitude = (4.0 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            local_amplitude *= 0.5
            direction = 1 if random.random() > 0.5 else -1
            y_abnormal[local_position] += direction * local_amplitude
            local_char["position_end"] = local_position + 1
            local_char["detail"] = f"A single point outlier with {'positive' if direction > 0 else 'negative'} amplitude {local_amplitude:.2f}"

        # 处理 "wide downward spike" 类型（非持续性）
        elif local_char["type"] == "wide downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            local_amplitude *= 0.8
            drop_length = random.randint(int(seq_len * 0.02), min(int(seq_len * 0.08), 500))
            peak_length = random.randint(1, 4)
            rise_length = random.randint(int(seq_len * 0.02), min(int(seq_len * 0.08), 500))
            y_abnormal[local_position:local_position + drop_length] += generate_ts_change(drop_length, -local_amplitude)
            y_abnormal[local_position + drop_length:local_position + drop_length + peak_length] -= local_amplitude
            y_abnormal[local_position + drop_length + peak_length:local_position + drop_length + peak_length + rise_length] += generate_ts_change(rise_length, local_amplitude) - local_amplitude
            local_char["position_end"] = local_position + drop_length + peak_length + rise_length
            local_char["detail"] = f"A local wide downward spike anomaly, with amplitude {local_amplitude:.2f}"

        local_char['amplitude'] = local_amplitude

    if contrast == True:
        # 应用持续性水平移位到 y_normal
        for start_position, shift in sorted(level_shifts, key=lambda x: x[0]):
            if start_position < seq_len:  # 确保不超出序列长度
                y_normal[start_position:] += shift

        # 按位置排序局部特征
    attribute_pool["local"] = sorted(attribute_pool["local"], key=lambda x: x['position_start'])

    return y_normal, y_abnormal

def generate_trend(attribute_pool, y_normal, y_abnormal, overall_amplitude, overall_bias, seq_len):
    # Apply trend attribute
    trend = attribute_pool["trend"]["type"]

    if 'amplitude' in attribute_pool['trend']:
        amplitude = attribute_pool['trend']['amplitude']
    else:
        amplitude = random.uniform(0.8, 3.0) * overall_amplitude
    if 'start' in attribute_pool['trend']:
        bias = attribute_pool['trend']['start']
    else:
        bias = overall_bias

    if trend == "decrease":
        cur_value = generate_ts_change(seq_len, -amplitude, add_random_noise=False) + bias
        y_normal += cur_value
        y_abnormal += cur_value
        attribute_pool["trend"]["detail"] = f"From the perspective of the slope, the overall trend is decreasing. "
        attribute_pool["trend"]["trend_list"] = [("decrease", 0, seq_len - 1)]
    elif trend == "increase":
        cur_value = generate_ts_change(seq_len, amplitude, add_random_noise=False) + bias
        y_normal += cur_value
        y_abnormal += cur_value
        attribute_pool["trend"]["detail"] = f"From the perspective of the slope, the overall trend is increasing. "
        attribute_pool["trend"]["trend_list"] = [("increase", 0, seq_len - 1)]
    elif trend == "multiple":
        # Ensure the generated trend has more than one type
        while True:
            points = generate_random_points(seq_len=seq_len)[0]
            if len(generate_trend_list(points, seq_len)) > 1:
                break
        trend_ts = generate_trend_curve(seq_len=seq_len, points=points)[1]
        y_normal += trend_ts * amplitude
        y_abnormal += trend_ts * amplitude
        attribute_pool["trend"]["detail"] = "From the perspective of the slope, the overall trend contains multiple different segments: " + generate_trend_prompt(points)
        attribute_pool["trend"]["trend_list"] = generate_trend_list(points, seq_len)
    elif trend == "arima":
        arima_ts, arima_para = generate_arima_series(length=seq_len)
        if arima_para['d'] == 0:
            amplitude /= 2
        elif arima_para['d'] == 1:
            amplitude /= 1.5
        y_normal += arima_ts / (arima_ts.max() - arima_ts.min()) * amplitude 
        y_abnormal += arima_ts / (arima_ts.max() - arima_ts.min()) * amplitude 
        attribute_pool["trend"]["detail"] = "From the perspective of the slope, the overall trend is generated by ARIMA model with parameters: " + str(arima_para)
        attribute_pool["trend"]["trend_list"] = [("arima", 0, seq_len - 1)]
    elif trend == "keep steady":
        y_normal += bias
        y_abnormal += bias
        attribute_pool["trend"]["detail"] = f"From the perspective of the slope, the overall trend is steady. "
        attribute_pool["trend"]["trend_list"] = [("keep steady", 0, seq_len - 1)]

    # Find increase or decrease in local char
    local_phase_change = [i['type'] for i in attribute_pool["local"] if 'increase' in i['type'] or 'decrease' in i['type']]
    if len(local_phase_change):
        attribute_pool["trend"]["detail"] += f"However, local phase changes were observed, including: {', '.join(local_phase_change)}. "
    attribute_pool["trend"]["start"] = round(float(y_normal[0]), 2)
    # attribute_pool["trend"]["amplitude"] = round(float(y_normal[-1] - y_normal[0]), 2)
    attribute_pool["trend"]["detail"] += (f"The value of time series starts from around {float(y_normal[0]):.2f} and ends at around {float(y_normal[-1]):.2f}, "
                                          f"with an overall amplitude of {float(y_normal[-1] - y_normal[0]):.2f}. ")
    return y_normal, y_abnormal

def generate_split_points(seq_len: int, num_segments: int) -> list:
    if num_segments < 1:
        raise ValueError("Number of segments must be at least 1.")
    if seq_len < num_segments:
        raise ValueError("Sequence length must be at least equal to the number of segments.")

    min_segment_len = seq_len / num_segments / 2  # Minimum segment length
    split_points = [0]  # Start with the first point
    
    for _ in range(num_segments - 1):
        # Determine the valid range for the next split point
        min_point = split_points[-1] + min_segment_len
        max_point = seq_len - (num_segments - len(split_points)) * min_segment_len
        if min_point >= max_point:
            raise ValueError("Cannot generate split points satisfying the constraints.")
        
        # Randomly select a split point within the valid range
        split_points.append(int(random.uniform(min_point, max_point)))
    split_points.append(seq_len)
    
    return split_points

def generate_time_series(attribute_pool, seq_len=512, num_seasonal_anomalies=None, contrast=True, is_multi=False):
    """
    Generate a time series based on the given attribute pool and sequence length.
    Parameters:
    attribute_pool (dict): A dictionary containing attributes that define the characteristics of the time series.
    seq_len (int, optional): The length of the generated time series. Default is 512.
    Returns:
    tuple: A tuple containing the generated time series (numpy array) and the updated attribute pool (dict).
    The attribute_pool dictionary can contain the following keys:
    - "seasonal": A dictionary with a "type" key that defines the type of seasonal pattern.
    - "trend": A dictionary with a "type" key that defines the type of trend.
    - "frequency": A dictionary with "type" and "period" keys that define the frequency characteristics.
    - "overall_amplitude": A float that defines the overall amplitude of the time series.
    - "overall_bias": A float that defines the overall bias of the time series.
    - "local": A list of dictionaries that define local characteristics of the time series.
    - "statistics": A dictionary that will be populated with statistical information about the generated time series.
    The function performs the following steps:
    1. Adapts to legacy behavior by modifying the attribute pool based on certain conditions.
    2. Generates the base time series using a linear space.
    3. Adjusts the period based on the frequency attribute.
    4. Sets an overall amplitude and bias for the time series.
    5. Applies seasonal attributes to the time series.
    6. Applies local changes to the time series.
    7. Applies trend attributes to the time series.
    8. Replaces details in local characteristics with actual values from the time series.
    9. Adds noise to the time series.
    10. Adds statistical information to the attribute pool.
    Returns the generated time series and the updated attribute pool.
    """
    # Adapt to legacy behavior
    if not ENABLE_MULTIPLE_TREND:
        # (Step 1) Remove seasonal type
        if "no period" not in attribute_pool["seasonal"]['type']:
            attribute_pool["seasonal"]["type"] = "periodic fluctuation"

        # (Step 2) Remove multiple trend
        if attribute_pool["trend"]["type"] == "multiple":
            attribute_pool["trend"]["type"] = random.choice(["increase", "decrease", "keep steady"])

    # Generate timeseries
    x = np.linspace(0, 10 * np.pi, seq_len)
    y_normal = np.zeros_like(x)
    y_abnormal = np.zeros_like(x)

    period = seq_len
    if "frequency" in attribute_pool:
        # Respect explicitly provided period if present
        if "period" in attribute_pool["frequency"]:
            period = attribute_pool["frequency"]["period"]
        else:
            freq_type = attribute_pool["frequency"]['type']
            if freq_type == "high frequency":
                period = random.uniform(10.0, 30.0)
            elif freq_type == "moderate frequency":
                period = random.uniform(30.0, 100.0)
            elif freq_type == "low frequency":
                # Uniformly pick number of cycles in [4,20], derive period, clamp to [100, 1500]
                num_cycles = random.uniform(4.0, 20.0)
                period = seq_len / num_cycles
                if period < 100.0:
                    period = 100.0
                elif period > 1500.0:
                    period = 1500.0
            elif freq_type == "no periodicity":
                pass
            else:
                period = random.uniform(30.0, 100.0)

        if attribute_pool["frequency"]['type'] == "no periodicity":
            attribute_pool["frequency"]['period'] = 0.0
            attribute_pool["frequency"]['detail'] = "No significant periodic fluctuations observed, overall almost no periodicity. "
        else:
            attribute_pool["frequency"]['period'] = round(period, 1)
            attribute_pool["frequency"]['detail'] = f"Each fluctuation period is approximately {period:.1f} points, thus the overall fluctuation is {attribute_pool['frequency']['type']}. "

    # Set an overall amplitude for all attributes (to ensure the time series is not too flat)
    if 'overall_amplitude' in attribute_pool and 'overall_bias' in attribute_pool:
        overall_amplitude = attribute_pool['overall_amplitude']
        overall_bias = attribute_pool['overall_bias']
    else:
        # Modified to match the scale of generate_attributes_from_set (0.1-10.0 for amp, -5.0-5.0 for bias)
        # Previous logic used powers of 10 which caused huge inconsistencies.
        overall_amplitude = random.uniform(0.1, 10.0)
        overall_bias = random.uniform(-5.0, 5.0)
        
        attribute_pool['overall_amplitude'] = overall_amplitude
        attribute_pool['overall_bias'] = overall_bias

    # Apply Seasonal Feature
    y_season_normal, y_season_anomaly = generate_seasonal(attribute_pool, overall_amplitude, seq_len, num_seasonal_anomalies)
    y_normal += y_season_normal
    y_abnormal += y_season_anomaly

    # Apply local changes
    y_local_normal, y_local_anomaly = generate_local_chars(attribute_pool, overall_amplitude, seq_len, contrast=contrast, is_multi=is_multi)
    y_normal += y_local_normal
    y_abnormal += y_local_anomaly

    # Apply trend attribute
    y_normal, y_abnormal = generate_trend(attribute_pool, y_normal, y_abnormal, overall_amplitude, overall_bias, seq_len)

    # Add optional periodic background spikes without affecting anomaly labels.
    y_for_noise = y_abnormal.copy()
    background_signal, background_metadata = generate_background_periodic_spikes(seq_len, overall_amplitude)
    attribute_pool["background_periodic_spike"] = background_metadata
    if background_metadata["enabled"]:
        y_normal += background_signal
        y_abnormal += background_signal

    # Add noise
    y_noise = generate_noise(attribute_pool, y_for_noise, overall_amplitude, seq_len)
    y_normal += y_noise
    y_abnormal += y_noise

    # Add statistic information to description
    attribute_pool["statistics"] = {
        "mean": round(float(np.mean(y_abnormal)), 2),
        "std": round(float(np.std(y_abnormal)), 2),
        "max": round(float(np.max(y_abnormal)), 2),
        "min": round(float(np.min(y_abnormal)), 2),
        "max_pos": int(np.argmax(y_abnormal)),
        "min_pos": int(np.argmin(y_abnormal))
    }

    return y_normal, y_abnormal, attribute_pool

def attribute_to_text(time_series: np.ndarray, attribute_pool: dict, generate_values: bool=True, include_attributes: List[str] = ['length', 'trend', 'periodicity', 'frequency', 'noise', 'local', 'statistic']) -> str:
    """
    Generates a textual description of a time series based on various attributes and attributes.
    Args:
        time_series (np.ndarray): The time series data as a numpy array.
        attribute_pool (dict): A dictionary containing attribute details for the time series.
        generate_values (bool, optional): Deprecated. Use 'statistic' in include_attributes instead. Defaults to True.
        include_attributes (List[str], optional): A list of attributes to include in the description. Defaults to ['length', 'trend', 'periodicity', 'frequency', 'noise', 'local', 'statistic'].
    Returns:
        str: A detailed textual description of the time series.
    """
    # Adapt to legacy parameters
    if not generate_values and 'statistic' in include_attributes:
        include_attributes.remove('statistic')
    elif generate_values and 'statistic' not in include_attributes:
        include_attributes.append('statistic')

    seq_len = len(time_series)
    segment_mean = [str(round(np.mean(time_series[i:i+seq_len//32]), 2)) for i in range(0, seq_len, seq_len // 32)]
    max_value = round(np.max(time_series), 2)
    min_value = round(np.min(time_series), 2)

    detailed_description = ''
    if 'length' in include_attributes:
        detailed_description += f"The length of the time series is {seq_len}. "
    if 'trend' in include_attributes:
        detailed_description += f"{attribute_pool['trend']['detail']}"
    if 'periodicity' in include_attributes:
        detailed_description += attribute_pool['seasonal']['detail']
    if "no" not in attribute_pool['seasonal']['type'] and 'frequency' in include_attributes:
        detailed_description += attribute_pool['frequency']['detail']
    if 'noise' in include_attributes:
        detailed_description += attribute_pool['noise']['detail']
    if 'local' in include_attributes:
        if len(attribute_pool["local"]):
            detailed_description += 'In terms of local characteristics, ' + ";".join([f"{i['detail']}, forming a {i['type']}" for i in attribute_pool['local']]) + '. '
        else:
            detailed_description += 'No local characteristics are found. '
    if 'statistic' in include_attributes:
        detailed_description += f"Specific data details: The time series is divided into 32 segments, with the approximate mean values for each {seq_len // 32}-point interval being: {segment_mean}. The maximum value of the entire series is {max_value}, and the minimum value is {min_value}."
    if 'periodicity' in include_attributes:
        detailed_description += attribute_pool['seasonal']['detail']
        if "seasonal_anomalies" in attribute_pool and attribute_pool["seasonal_anomalies"]:
            for anomaly in attribute_pool["seasonal_anomalies"]:
                detailed_description += f" Anomalous {anomaly['type']} observed between point {anomaly['start']} and {anomaly['end']}: {anomaly['details']}. "

    return detailed_description


def attribute_to_caption(time_series: np.ndarray, attribute_pool: dict, generate_values: bool=True) -> str:
    """
        Compared with text, caption is in a more natural and fluent way that combines the trend with the local flucations.
    """
    seq_len = len(time_series)
    segment_mean = [round(np.mean(time_series[i:i+seq_len//32]), 2) for i in range(0, seq_len, seq_len // 32)]
    max_value = round(np.max(time_series), 2)
    min_value = round(np.min(time_series), 2)

    # Some basic attribute_pool
    detailed_description = ''
    detailed_description += f"The length of the time series is {seq_len}. "
    detailed_description += attribute_pool['seasonal']['detail']
    if "no" not in attribute_pool['seasonal']['type']:
        detailed_description += attribute_pool['frequency']['detail']
    detailed_description += attribute_pool['noise']['detail']

    # Combine the multiple attribute_pool
    detailed_description += "In terms of the trend and changes of this time series: At the beginning, "
    all_local_changes = dict((int(v['position_start']), v) for v in attribute_pool['local'])
    cur_pos = 0
    while True:
        if cur_pos >= seq_len - 1:
            break

        # Find the next local change
        later_changes = sorted(k for k in all_local_changes if k >= cur_pos)
        later_trend = sorted(k[1] for k in attribute_pool["trend"]["trend_list"] if k[1] > cur_pos)
        cur_trend = [k for k in attribute_pool["trend"]["trend_list"] if (k[1] <= cur_pos < k[2])][0]

        if (len(later_changes) > 0 and len(later_trend) > 0 and later_changes[0] < later_trend[0]) or (len(later_changes) > 0 and len(later_trend) == 0):
            # Later is a change
            nxt_pos = later_changes[0]
            cur_change = [k for k in attribute_pool["local"] if k['position_start'] == nxt_pos][0]
            if nxt_pos > cur_pos:
                detailed_description += f"from point {cur_pos} to {nxt_pos}, the time series {cur_trend[0]} with values from {float(time_series[cur_pos]):.2f} to {float(time_series[nxt_pos]):.2f}; "
            detailed_description += f"from point {cur_change['position_start']} to point {cur_change['position_end']}, {cur_change['detail']}, forming a {cur_change['type']}; "
            cur_pos = cur_change['position_end']
        elif (len(later_changes) > 0 and len(later_trend) > 0 and later_changes[0] >= later_trend[0]) or (len(later_trend) > 0 and len(later_changes) == 0):
            # Later is a trend
            nxt_pos = later_trend[0]
            nxt_trend = [k for k in attribute_pool["trend"]["trend_list"] if k[1] == nxt_pos]
            if nxt_pos > cur_pos:
                detailed_description += f"from point {cur_pos} to {nxt_pos}, the time series {cur_trend[0]} with values from {float(time_series[cur_pos]):.2f} to {float(time_series[nxt_pos]):.2f}, and then the trend of the time series changes to {nxt_trend[0][0]}; "
            cur_pos = nxt_pos
        else:
            # Later is the end
            nxt_pos = seq_len - 1
            if nxt_pos > cur_pos:
                detailed_description += f"finally, from point {cur_pos} to {nxt_pos}, the time series {cur_trend[0]} with values from {float(time_series[cur_pos]):.2f} to {float(time_series[nxt_pos]):.2f}. "
            break
    
    if generate_values:
        detailed_description += f"Specific data details: The time series is divided into 32 segments, with the approximate mean values for each {seq_len // 32}-point interval being: {segment_mean}. The maximum value of the entire series is {max_value}, and the minimum value is {min_value}, the start value is {float(time_series[0]):.2f}, the end value if {float(time_series[-1]):.2f}. "
        
        # Random choose some points
        for _ in range(5):
            cur_pos = random.choice(list(range(seq_len)))
            detailed_description += f"The value of point {cur_pos} is {float(time_series[cur_pos]):.2f}. "

    return detailed_description

def prompt_to_inference(timeseries: np.ndarray, prompt: str) -> str:
    prompt_list = prompt.split("<ts><ts/>")
    result = prompt_list[0]

    for i in range(len(prompt_list) - 1):
        cur_ts = timeseries[i]
        if type(cur_ts) == np.ndarray:
            cur_ts = cur_ts.tolist()
        cur_ts = [[round(float(v), 4) for v in item] for item in cur_ts]
        result += f"<ts>{cur_ts}<ts/>" + prompt_list[i + 1]

    return result
