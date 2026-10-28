"""
Multivariate time series generation module.

This module provides functions to generate multivariate time series data
with configurable anomaly patterns, dependencies between features, and
activation functions.
"""

import networkx as nx
import random
import json
import numpy as np
import re
from typing import Optional, List, Tuple, Dict

from ts_generator import generate_controlled_attributes, generate_time_series, attribute_to_text
from attribute_utils import metric_to_controlled_attributes
from config import CONFIG_PATH, ALL_ATTRIBUTE_SET

# ==============================================================================
# Constants and Configuration
# ==============================================================================

# Multivariate Generation Constants
MIN_NUM_NODES = 2
MAX_NUM_NODES = 50
MAX_DAG_PATH_LENGTH = 3

# Time Series Scaling Constants
SCALE_THRESHOLD = 3.0

# Lag and Dependency Constants
MAX_LAG_RATIO = 0.1
MIN_LAG = 0
MAX_LAG_ABSOLUTE = 8

# Linear Combination Coefficients
COEFF_A_MIN = -0.8
COEFF_A_MAX = 0.8
COEFF_B_MIN = -2
COEFF_B_MAX = 2
COEFF_C_MIN = -3
COEFF_C_MAX = 3
ALPHA_MIN = 0.2
ALPHA_MAX = 0.6

# Anomaly Configuration
MAX_ANOMALY_RATIO = 0.35
DEFAULT_NUM_LOCAL_ANOMALIES = 1
DEFAULT_NUM_SEASONAL_ANOMALIES = 0

# Sequence Length Configuration
DEFAULT_SEQ_LEN = 512
SEQ_LEN_MIN = 100
SEQ_LEN_MAX = 10000
GEOMETRIC_PROB = 0.0004


def convert_str_to_float(obj):
    """
    Recursively convert string representations of floats to actual floats.
    
    Parameters
    ----------
    obj : any
        Object to convert (can be dict, list, str, or other types).
    
    Returns
    -------
    any
        Converted object with strings parsed to floats where applicable.
    """
    if isinstance(obj, dict):
        return {k: convert_str_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_str_to_float(item) for item in obj]
    elif isinstance(obj, str):
        s = obj.strip()
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s):
            return float(s)
        return obj
    else:
        return obj


# Load metric configuration
metric_config = json.load(open(CONFIG_PATH, 'rt'))
metric_config = convert_str_to_float(metric_config)


def process_timeseries(time_series, attribute_pool, max_anomaly_ratio=MAX_ANOMALY_RATIO):
    """
    Process time series and attribute pool to generate labels.
    
    Parameters
    ----------
    time_series : ndarray
        Time series data.
    attribute_pool : dict
        Attribute pool containing anomalies and their types.
    max_anomaly_ratio : float, default=0.35
        Maximum allowed ratio of anomalous points.
    
    Returns
    -------
    dict or False
        Sample dictionary if successful, False if validation fails.
    """
    seasonal_anomalies = attribute_pool.get('seasonal_anomalies', [])
    local_anomalies = attribute_pool.get('local', [])
    attribute_type = {
        'seasonal': attribute_pool.get('seasonal', {}).get('type', ''),
        'trend': attribute_pool.get('trend', {}).get('type', ''),
        'frequency': attribute_pool.get('frequency', {}).get('type', ''),
        'noise': attribute_pool.get('noise', {}).get('type', '')
    }
    all_anomalies = local_anomalies + seasonal_anomalies

    # Validate anomalies
    valid_anomalies = []
    for anomaly in all_anomalies:
        has_positions = ('position_start' in anomaly and 'position_end' in anomaly) or \
                        ('start' in anomaly and 'end' in anomaly)
        if has_positions:
            valid_anomalies.append(anomaly)
        else:
            return False

    n = len(time_series)
    labels_ts = np.zeros(n, dtype=int)

    # Assign labels to positions
    anomaly_dict = {}
    for j, anomaly in enumerate(valid_anomalies):
        if 'position_start' in anomaly:
            start = anomaly['position_start']
            end = anomaly['position_end']
        else:
            start = anomaly['start']
            end = anomaly['end']
        anomaly_type = anomaly['type']
        anomaly_dict[f'{j}_{anomaly_type}'] = (start, end)

        # Validate positions
        if start < 0 or end >= n or start > end:
            return False

        # Check for overlaps
        positions = np.arange(start, end)
        if np.any(np.array(labels_ts)[positions] != 0):
            return False
        labels_ts[positions] = 1

    # Check anomaly ratio
    anomaly_ratio = np.sum(labels_ts) / n
    if anomaly_ratio > max_anomaly_ratio:
        return False
    
    attribute_type['anomalies'] = anomaly_dict
    sample = {
        'time_series': time_series,
        'labels': labels_ts,
        'attribute_pool': attribute_type
    }
    return sample


def get_num_edges(n_nodes):
    """
    Calculate reasonable number of edges for a DAG.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    
    Returns
    -------
    int
        Reasonable number of edges for the DAG.
    """
    if n_nodes <= 1:
        raise ValueError("Number of nodes must be greater than 1.")
    elif n_nodes == 2:
        return 1
    else:
        edges = n_nodes
        if 4 <= n_nodes <= 6:
            num_edges = random.randint(edges - 1, edges + 1)
        else:
            num_edges = random.randint(int(0.8 * edges), int(1.2 * edges))
    return num_edges


def ts_lag(time_series, lag=0):
    """
    Apply lag transformation to a time series.
    
    Parameters
    ----------
    time_series : ndarray
        Original time series.
    lag : int, default=0
        Lag value to apply.
    
    Returns
    -------
    ndarray
        Lagged time series.
    """
    if lag < 0:
        raise ValueError("Lag must be non-negative.")
    elif lag == 0:
        return time_series
    elif lag >= len(time_series):
        raise ValueError("Lag is greater than or equal to time series length.")
    
    fun = random.choice([lambda x: x])
    transformed_series = fun(time_series)
    return np.concatenate(([transformed_series[0]] * lag, transformed_series[:-lag]))


def extend_labels(labels, lag=0):
    """
    Expand anomaly labels by applying a lag.
    
    Parameters
    ----------
    labels : ndarray
        Original binary labels.
    lag : int, default=0
        Lag value to apply.
    
    Returns
    -------
    ndarray
        Extended labels with lag applied.
    """
    if lag < 0:
        raise ValueError("Lag must be non-negative.")
    elif lag == 0:
        return labels
    elif lag >= len(labels):
        raise ValueError("Lag is greater than or equal to labels length.")
    
    extended_label = np.zeros_like(labels, dtype=int)
    
    padded_label = np.concatenate(([0], labels, [0]))
    diff = np.diff(padded_label)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return extended_label

    for i in range(len(starts)):
        start = starts[i]
        original_end = ends[i]
        extended_end = min(original_end + lag, len(labels))
        
        if i + 1 < len(starts) and extended_end > starts[i+1]:
            print(f"Warning: Anomaly regions [{start}, {original_end}) and [{starts[i+1]}, {ends[i+1]}) overlap after extension.")
        
        extended_label[start:extended_end] = 1
        
    return extended_label


def activate(time_series, normal_time_series):
    """
    Apply activation function and standardize time series.
    
    Parameters
    ----------
    time_series : ndarray
        Time series with anomalies.
    normal_time_series : ndarray
        Time series without anomalies.
    
    Returns
    -------
    tuple of ndarray
        Activated and standardized (time_series, normal_time_series).
    """
    time_series = np.asarray(time_series, dtype=float)
    normal_time_series = np.asarray(normal_time_series, dtype=float)
    
    probs = np.array([0.9, 0.028, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012])
    funcs = {
        'Identity': lambda t: t,
        'Sign': np.sign,
        'ReLU': lambda t: np.maximum(0, t),
        'LeakyReLU': lambda t: np.where(t > 0, t, 0.01*t),
        'ELU': lambda t: np.where(t > 0, t, np.expm1(t)),
        'GELU': lambda t: 0.5 * t * (1 + np.tanh(np.sqrt(2/np.pi) * (t + 0.044715 * t**3))),
        'Sigmoid': lambda t: 1 / (1 + np.exp(-t)),
        'Tanh': np.tanh
    }
    names = list(funcs.keys())
    if not np.isclose(probs.sum(), 1.0) or len(probs) != len(names):
        raise ValueError("Probabilities must sum to 1 and match number of functions.")
    
    idx = np.random.choice(len(names), p=probs)
    act_time_series = funcs[names[idx]](time_series)
    act_normal_time_series = funcs[names[idx]](normal_time_series)
    
    mean = np.mean(act_time_series)
    act_time_series -= mean
    act_normal_time_series -= mean
    
    scale_factor = 1.0
    if np.any(np.abs(act_time_series) >= SCALE_THRESHOLD):
        scale_factor = np.max(np.abs(act_time_series)) / SCALE_THRESHOLD
        act_time_series /= scale_factor
        act_normal_time_series /= scale_factor
    
    return act_time_series, act_normal_time_series


def generate_random_dag(n_nodes, n_edges):
    """
    Generate a random directed acyclic graph (DAG).
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    n_edges : int
        Number of edges.
    
    Returns
    -------
    networkx.DiGraph
        Random DAG.
    """
    if n_nodes <= 1:
        raise ValueError("Number of nodes must be greater than 1.")
    
    max_edges = n_nodes * (n_nodes - 1) // 2
    if n_edges > max_edges:
        raise ValueError(f"Number of edges cannot exceed {max_edges} for {n_nodes} nodes.")

    while True:
        G = nx.DiGraph()
        nodes = list(range(n_nodes))
        G.add_nodes_from(nodes)

        possible_edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                possible_edges.append((i, j))
        
        edges_to_add = random.sample(possible_edges, n_edges)
        G.add_edges_from(edges_to_add)
        if nx.is_directed_acyclic_graph(G):
            break
    
    return G


def generate_attributes_from_set(
    change_positions: Optional[List[Tuple[Optional[int], Optional[float]]]] = None,
    num_seasonal_anomalies: Optional[int] = None,
    enable_multiple_trend: bool = True,
    amplitude_range: Tuple[float, float] = (0.1, 10.0),
    period_range: Tuple[float, float] = (10.0, 100.0),
    start_range: Tuple[float, float] = (-5.0, 5.0),
    seq_len: Optional[int] = None
) -> dict:
    """
    Generate time series attributes directly from ALL_ATTRIBUTE_SET.
    
    This function generates attributes based on weighted random selection
    from the configuration set without relying on synthetic.json.
    
    Parameters
    ----------
    change_positions : list of tuple, optional
        Local change positions and amplitudes. If None, randomly generates 0-3 positions.
    num_seasonal_anomalies : int, optional
        Number of seasonal anomalies. If None, automatically determined.
    enable_multiple_trend : bool, default=True
        Whether to allow multiple trend segments.
    amplitude_range : tuple of float, default=(0.1, 10.0)
        Range for amplitude values (min, max).
    period_range : tuple of float, default=(10.0, 100.0)
        Range for period values (min, max).
    start_range : tuple of float, default=(-5.0, 5.0)
        Range for start values (min, max).
    seq_len : int, optional
        Length of the time series. Used for calculating low frequency periods.
    
    Returns
    -------
    dict
        Attribute dictionary with keys:
        - 'seasonal': Seasonal attributes
        - 'trend': Trend attributes
        - 'local': Local change list
        - 'frequency': Frequency attributes
        - 'noise': Noise attributes
        - 'num_seasonal_anomalies': Number of seasonal anomalies
    """
    if change_positions is None:
        change_positions = [(None, None) for _ in range(random.randint(0, 3))]
    
    description = {}
    
    # Generate seasonal attributes
    seasonal_options = ALL_ATTRIBUTE_SET['overall_attribute']['seasonal']
    seasonal_types = list(seasonal_options.keys())
    seasonal_weights = list(seasonal_options.values())
    seasonal_type = np.random.choice(seasonal_types, p=np.array(seasonal_weights) / sum(seasonal_weights))
    
    description["seasonal"] = {
        "type": seasonal_type,
        "amplitude": random.uniform(amplitude_range[0], amplitude_range[1])
    }
    
    # Generate trend attributes
    trend_options = ALL_ATTRIBUTE_SET['overall_attribute']['trend'].copy()
    if not enable_multiple_trend and "multiple" in trend_options:
        trend_options.pop("multiple")
        if not trend_options:
            trend_options = {"increase": 0.33, "decrease": 0.33, "keep steady": 0.34}
    
    trend_types = list(trend_options.keys())
    trend_weights = list(trend_options.values())
    trend_type = np.random.choice(trend_types, p=np.array(trend_weights) / sum(trend_weights))
    
    description["trend"] = {
        "type": trend_type,
        "start": random.uniform(start_range[0], start_range[1]),
        "amplitude": random.uniform(amplitude_range[0], amplitude_range[1])
    }
    
    # Force no periodicity and no noise for ARIMA
    if description["trend"]["type"] == "arima":
        description["seasonal"]['type'] = 'no periodic fluctuation'
        description["seasonal"]['amplitude'] = 0.0
    
    # Generate local change attributes
    num_local_chars = len(change_positions)
    change_options = ALL_ATTRIBUTE_SET['change']
    change_types = list(change_options.keys())
    change_weights = list(change_options.values())
    
    local_chars = list(np.random.choice(
        change_types, 
        size=num_local_chars, 
        p=np.array(change_weights) / sum(change_weights)
    ))
    
    description["local"] = []
    for char, (pos, amp) in zip(local_chars, change_positions):
        description["local"].append({
            "type": char,
            "position_start": pos,
            "amplitude": amp
        })
    
    # Generate frequency attributes
    if 'no periodic fluctuation' not in description["seasonal"]['type']:
        freq_options = ALL_ATTRIBUTE_SET['overall_attribute']['frequency']
        freq_types = list(freq_options.keys())
        freq_weights = list(freq_options.values())
        freq_type = np.random.choice(freq_types, p=np.array(freq_weights) / sum(freq_weights))

        if freq_type == 'high frequency':
            period = random.uniform(10.0, 30.0)
        elif freq_type == 'moderate frequency':
            period = random.uniform(30.0, 100.0)
        elif freq_type == 'low frequency':
            if seq_len is not None:
                num_periods = random.uniform(4, 20)
                raw_period = seq_len / num_periods
                if 100 <= raw_period <= 1500:
                    period = raw_period
                else:
                    if raw_period < 100:
                        period = 100.0
                    else:
                        period = 1500.0
            else:
                # Fallback if seq_len is missing
                period = random.uniform(100.0, 1500.0)
        else:
            # Fallback for unknown types
            period = random.uniform(period_range[0], period_range[1])

        description["frequency"] = {'type': freq_type, 'period': round(period, 1)}
    else:
        description["frequency"] = {'type': 'no periodicity'}
    
    # Generate noise attributes
    noise_options = ALL_ATTRIBUTE_SET['overall_attribute']['noise']
    noise_types = list(noise_options.keys())
    noise_weights = list(noise_options.values())
    noise_type = np.random.choice(noise_types, p=np.array(noise_weights) / sum(noise_weights))
    
    description["noise"] = {'type': noise_type}
    
    # Reselect noise if incompatible with periodicity
    while True:
        if description["seasonal"]['type'] != "no periodic fluctuation" and \
           description["noise"]['type'] not in ['almost no noise', 'low noise']:
            noise_type = np.random.choice(noise_types, p=np.array(noise_weights) / sum(noise_weights))
            description["noise"] = {'type': noise_type}
        else:
            break
    
    # Force almost no noise for ARIMA
    if description["trend"]['type'] == "arima":
        description["noise"]["type"] = "almost no noise"
    
    # Determine number of seasonal anomalies
    if num_seasonal_anomalies is None:
        if description["seasonal"]['type'] != "no periodic fluctuation":
            num_seasonal_anomalies = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            num_seasonal_anomalies = 0
    
    description['num_seasonal_anomalies'] = num_seasonal_anomalies
    
    return description


def generate_univariate_timeseries(
    seq_len: int = DEFAULT_SEQ_LEN, 
    num_local_char: int = None, 
    num_seasonal_anomalies: int = None, 
    is_multi: bool = False,
    use_attribute_set: bool = False,
    metric: str = None
) -> tuple:
    """
    Generate a univariate time series with configurable anomaly patterns.
    
    Parameters
    ----------
    seq_len : int, default=512
        Length of the time series.
    num_local_char : int, optional
        Number of local anomalies. If None, randomly determined.
    num_seasonal_anomalies : int, optional
        Number of seasonal anomalies. If None, automatically determined.
    is_multi : bool, default=False
        Whether part of multivariate generation (affects scaling).
    use_attribute_set : bool, default=False
        If True, generate attributes from ALL_ATTRIBUTE_SET.
        If False, use synthetic.json metrics.
    metric : str, optional
        Metric to use when use_attribute_set=False.
    
    Returns
    -------
    tuple
        - normal_timeseries : ndarray
            Normal time series without anomalies.
        - timeseries : ndarray
            Time series with anomalies.
        - labels : ndarray
            Binary anomaly labels.
        - attributes : dict
            Attribute metadata.
    """
    if num_local_char is None:
        change_positions = None
    else:
        change_positions = [(None, None) for _ in range(num_local_char)]
    
    while True:
        if use_attribute_set:
            attribute_pool = generate_attributes_from_set(
                change_positions=change_positions,
                num_seasonal_anomalies=num_seasonal_anomalies,
                seq_len=seq_len
            )
        else:
            if metric is None:
                sample = random.choice(list(metric_config))
                metric = random.choice(sample['metrics'])
            attribute_pool = generate_controlled_attributes(
                metric_to_controlled_attributes(metric), 
                change_positions,
                num_seasonal_anomalies
            )
        
        # Optional: boost outlier count when only outliers exist (no seasonal anomalies and no other local types)
        # Condition: only outlier types present, no seasonal anomalies
        locals_list = attribute_pool.get("local", [])
        seasonal_count = attribute_pool.get("num_seasonal_anomalies", 0)
        if seasonal_count == 0 and locals_list and all(l.get("type") == "outlier" for l in locals_list):
            if random.random() < 0.2:
                max_extra = max(0, seq_len // 200)
                if max_extra > 0:
                    extra_num = min(random.randint(5, 30), max_extra)
                    for _ in range(extra_num):
                        attribute_pool["local"].append({"type": "outlier", "position_start": None, "amplitude": None})
        
        num_seasonal_anomalies_to_use = attribute_pool.get('num_seasonal_anomalies', 0)
        
        normal_timeseries, timeseries, attribute_pool = generate_time_series(
            attribute_pool, seq_len, num_seasonal_anomalies_to_use, contrast=False, is_multi=is_multi
        )
        uni_ts = process_timeseries(timeseries, attribute_pool)
        if uni_ts:
            break
    
    mean = np.mean(timeseries)
    scaled_normal_timeseries = normal_timeseries - mean
    scaled_timeseries = timeseries - mean
    scale_factor = 1.0
    if np.any(np.abs(scaled_timeseries) >= SCALE_THRESHOLD):
        scale_factor = np.max(np.abs(scaled_timeseries)) / SCALE_THRESHOLD
        scaled_timeseries /= scale_factor
        scaled_normal_timeseries /= scale_factor
    labels = uni_ts['labels']
    attribute = uni_ts['attribute_pool']
    # Preserve full attribute_pool for downstream description/text generation
    attribute['full_attribute_pool'] = attribute_pool

    return scaled_normal_timeseries, scaled_timeseries, labels, attribute


def generate_multivariate_timeseries(
    seq_len: int = DEFAULT_SEQ_LEN, 
    num_nodes: int = None, 
    activate_fun: bool = False,
    use_attribute_set: bool = False
) -> tuple:
    """
    Generate a multivariate time series with configurable anomaly patterns.
    
    Parameters
    ----------
    seq_len : int, default=512
        Length of the time series.
    num_nodes : int, optional
        Number of features/nodes. If None, randomly generated.
    activate_fun : bool, default=False
        Whether to apply activation functions.
    use_attribute_set : bool, default=False
        If True, generate attributes from ALL_ATTRIBUTE_SET.
        If False, use synthetic.json metrics.
    
    Returns
    -------
    tuple
        - normal_timeseries : ndarray of shape (seq_len, num_features)
            Normal time series without anomalies.
        - anomaly_timeseries : ndarray of shape (seq_len, num_features)
            Time series with anomalies.
        - labels : ndarray of shape (seq_len,)
            Binary anomaly labels.
        - anomaly_type : str
            'Endogenous' or 'Exogenous'.
        - attributes_list : list of dict
            List of attribute dictionaries for each feature.
        - dag : networkx.DiGraph
            DAG representing feature dependencies.
    """
    # Generate DAG
    if num_nodes is None:
        num_nodes = random.randint(MIN_NUM_NODES, MAX_NUM_NODES)
    num_edges = get_num_edges(num_nodes)
    while True:
        dag = generate_random_dag(num_nodes, num_edges)
        if nx.dag_longest_path_length(dag) <= MAX_DAG_PATH_LENGTH:
            break
    
    all_parents = []
    for node in sorted(dag.nodes()):
        parents = list(dag.predecessors(node))
        all_parents.append(parents)
    
    attributes_list = []
    is_endogenous = random.choice([True, False])
    anomaly_index = random.randint(0, num_nodes - 1)
    anomaly_timeseries = np.zeros(seq_len)
    labels = np.zeros(seq_len, dtype=int)
    all_timeseries = np.zeros((seq_len, num_nodes))
    all_normal_timeseries = np.zeros((seq_len, num_nodes))
    
    for i in range(num_nodes):
        if i == anomaly_index:
            num_local_char = DEFAULT_NUM_LOCAL_ANOMALIES
            num_seasonal_anomalies = DEFAULT_NUM_SEASONAL_ANOMALIES
            while True:
                normal_timeseries_i, timeseries_i, labels_i, attribute_i = generate_univariate_timeseries(
                    seq_len, num_local_char, num_seasonal_anomalies, 
                    is_multi=True, use_attribute_set=use_attribute_set
                )
                if len(attribute_i['anomalies']) > 0:
                    labels = labels | labels_i
                    anomaly_timeseries = timeseries_i
                    all_normal_timeseries[:, i] = normal_timeseries_i
                    if is_endogenous:
                        all_timeseries[:, i] = timeseries_i
                    else:
                        all_timeseries[:, i] = normal_timeseries_i
                    break       
        else:
            num_local_char = 0
            num_seasonal_anomalies = 0
            normal_timeseries_i, timeseries_i, _, attribute_i = generate_univariate_timeseries(
                seq_len, num_local_char, num_seasonal_anomalies, 
                is_multi=True, use_attribute_set=use_attribute_set
            )
            all_normal_timeseries[:, i] = normal_timeseries_i
            all_timeseries[:, i] = timeseries_i
        attributes_list.append(attribute_i)

        if len(all_parents[i]) == 0:
            if activate_fun and i != anomaly_index:
                all_timeseries[:, i], all_normal_timeseries[:, i] = activate(all_timeseries[:, i], all_normal_timeseries[:, i])
            continue
        else:
            lag = np.random.randint(MIN_LAG, min(MAX_LAG_ABSOLUTE, int(seq_len * MAX_LAG_RATIO)), size=len(all_parents[i]))
            parent_timeseries = []
            parent_normal_timeseries = []
            for idx, parent in enumerate(all_parents[i]):
                if parent == anomaly_index and is_endogenous:
                    labels = extend_labels(labels, lag[idx])
                parent_timeseries.append(ts_lag(all_timeseries[:, parent], lag[idx]))
                parent_normal_timeseries.append(ts_lag(all_normal_timeseries[:, parent], lag[idx]))
        
            a = random.uniform(COEFF_A_MIN, COEFF_A_MAX)
            b = np.random.uniform(COEFF_B_MIN, COEFF_B_MAX, size=len(parent_timeseries))
            c = random.uniform(COEFF_C_MIN, COEFF_C_MAX)
            y_0 = np.zeros(seq_len)
            y_0_normal = np.zeros(seq_len)
            for t in range(1, seq_len):
                y_0[t] = a * y_0[t - 1] + np.sum(b * np.array(parent_timeseries)[:, t]) + c
                y_0_normal[t] = a * y_0_normal[t - 1] + np.sum(b * np.array(parent_normal_timeseries)[:, t]) + c
            mean = np.mean(y_0)
            y_0 = y_0 - mean
            y_0_normal = y_0_normal - mean
            if np.any(np.abs(y_0) >= SCALE_THRESHOLD):
                scale_factor = np.max(np.abs(y_0)) / SCALE_THRESHOLD
                y_0 /= scale_factor
                y_0_normal /= scale_factor
            
            alpha = random.uniform(ALPHA_MIN, ALPHA_MAX)
            y = alpha * y_0 + (1 - alpha) * all_timeseries[:, i]
            y_normal = alpha * y_0_normal + (1 - alpha) * all_normal_timeseries[:, i]
            if anomaly_index == i and not is_endogenous:
                anomaly_timeseries = alpha * y_0 + (1 - alpha) * anomaly_timeseries
            
            if activate_fun and i != anomaly_index:
                y, y_normal = activate(y, y_normal)
            all_timeseries[:, i] = y
            all_normal_timeseries[:, i] = y_normal
    
    if not is_endogenous:
        all_timeseries[:, anomaly_index] = anomaly_timeseries
    
    return all_normal_timeseries, all_timeseries, labels, 'Endogenous' if is_endogenous else 'Exogenous', attributes_list, dag


def generate_control_univariate_timeseries(
    seq_len: int = DEFAULT_SEQ_LEN, 
    num_local_char: int = None, 
    num_seasonal_anomalies: int = None, 
    is_multi: bool = False
) -> tuple:
    """
    Generate a controlled univariate time series with specific anomaly patterns.
    
    This is a legacy function that uses synthetic.json for attribute generation.
    
    Parameters
    ----------
    seq_len : int, default=512
        Length of the time series.
    num_local_char : int, optional
        Number of local anomalies.
    num_seasonal_anomalies : int, optional
        Number of seasonal anomalies.
    is_multi : bool, default=False
        Whether part of multivariate generation.
    
    Returns
    -------
    tuple
        - normal_timeseries : ndarray
        - timeseries : ndarray
        - labels : ndarray
        - attributes : dict
    """
    sample = random.choice(list(metric_config))
    metric = random.choice(sample['metrics'])
    
    if num_local_char is None:
        change_positions = None
    else:
        change_positions = [(None, None) for _ in range(num_local_char)]
    
    while True:
        attribute_pool = generate_controlled_attributes(
            metric_to_controlled_attributes(metric), 
            change_positions,
            num_seasonal_anomalies
        )
        
        num_seasonal_anomalies_to_use = attribute_pool.get('num_seasonal_anomalies', 0)
        
        normal_timeseries, timeseries, attribute_pool = generate_time_series(
            attribute_pool, seq_len, num_seasonal_anomalies_to_use, contrast=False, is_multi=is_multi
        )
        uni_ts = process_timeseries(timeseries, attribute_pool)
        if uni_ts:
            break
    
    mean = np.mean(timeseries)
    scaled_normal_timeseries = normal_timeseries - mean
    scaled_timeseries = timeseries - mean
    scale_factor = 1.0
    if np.any(np.abs(scaled_timeseries) >= SCALE_THRESHOLD):
        scale_factor = np.max(np.abs(scaled_timeseries)) / SCALE_THRESHOLD
        scaled_timeseries /= scale_factor
        scaled_normal_timeseries /= scale_factor
    labels = uni_ts['labels']
    attribute = uni_ts['attribute_pool']

    return scaled_normal_timeseries, scaled_timeseries, labels, attribute
