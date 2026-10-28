"""
Dataset generation module for time series anomaly detection.

This module provides functions to generate synthetic time series datasets
with configurable anomaly patterns for both univariate and multivariate data.
"""

from ts_multi_generator import generate_univariate_timeseries, generate_multivariate_timeseries
import numpy as np
import os
import pickle
import random
import logging
from tqdm import tqdm
from typing import Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ==============================================================================
# Logging Configurationgit
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Constants and Configuration
# ==============================================================================

DEFAULT_SEQ_LEN_MIN = 1000
DEFAULT_SEQ_LEN_MAX = 10000
GEOMETRIC_DISTRIBUTION_P = 0.00025

DEFAULT_NUM_SAMPLES = 1000
DEFAULT_ANOMALY_RATIO = 0.5
DEFAULT_NUM_WORKERS = 80  # None means use CPU count


def random_seq_len(low: int = DEFAULT_SEQ_LEN_MIN, high: int = DEFAULT_SEQ_LEN_MAX) -> int:
    """
    Generate a random sequence length using geometric distribution.
    
    Parameters
    ----------
    low : int, default=100
        Minimum sequence length.
    high : int, default=10000
        Maximum sequence length.
    
    Returns
    -------
    int
        A random sequence length within the specified range.
    """
    while True:
        seq_len = np.random.geometric(GEOMETRIC_DISTRIBUTION_P)
        if low <= seq_len <= high:
            break
    return seq_len


def _create_sample_dict(
    normal_time_series: np.ndarray, 
    time_series: np.ndarray, 
    labels: np.ndarray, 
    attribute: Dict
) -> Dict:
    """
    Create a standardized sample dictionary.
    
    Parameters
    ----------
    normal_time_series : ndarray
        Time series without anomalies.
    time_series : ndarray
        Time series with anomalies.
    labels : ndarray
        Binary labels for anomaly positions.
    attribute : dict
        Metadata about the time series.
    
    Returns
    -------
    dict
        A standardized sample dictionary.
    """
    return {
        'normal_time_series': normal_time_series,
        'time_series': time_series,
        'labels': labels,
        'attribute': attribute
    }


def _generate_univariate_anomaly_sample(
    seq_len: int, 
    metrics: Optional[List] = None,
    use_attribute_set: bool = False
) -> Optional[Dict]:
    """
    Generate a single univariate sample with guaranteed anomalies.
    
    Parameters
    ----------
    seq_len : int
        Length of the time series.
    metrics : list, optional
        List of metrics to choose from.
    use_attribute_set : bool, default=False
        Whether to use ALL_ATTRIBUTE_SET directly instead of synthetic.json.
    
    Returns
    -------
    dict or None
        Sample dictionary if successful, None otherwise.
    """
    try:
        metric_to_use = random.choice(metrics) if metrics else None
        while True:
            normal_ts, timeseries, labels, attribute = generate_univariate_timeseries(
                seq_len, 
                metric=metric_to_use,
                use_attribute_set=use_attribute_set
            )
            if len(attribute['anomalies']) > 0:
                return _create_sample_dict(normal_ts, timeseries, labels, attribute)
    except Exception as e:
        logger.debug(f"Failed to generate univariate anomaly sample: {e}")
        return None


def _generate_univariate_mixed_sample(
    seq_len: int, 
    anomaly_sample_ratio: float, 
    metrics: Optional[List] = None,
    use_attribute_set: bool = False
) -> Optional[Dict]:
    """
    Generate a single univariate sample with probabilistic anomalies.
    
    Parameters
    ----------
    seq_len : int
        Length of the time series.
    anomaly_sample_ratio : float
        Probability of including anomalies.
    metrics : list, optional
        List of metrics to choose from.
    use_attribute_set : bool, default=False
        Whether to use ALL_ATTRIBUTE_SET directly instead of synthetic.json.
    
    Returns
    -------
    dict or None
        Sample dictionary if successful, None otherwise.
    """
    try:
        metric_to_use = random.choice(metrics) if metrics else None
        normal_timeseries, timeseries, labels, attribute = generate_univariate_timeseries(
            seq_len, 
            metric=metric_to_use,
            use_attribute_set=use_attribute_set
        )
        
        if random.random() < anomaly_sample_ratio:
            return _create_sample_dict(normal_timeseries, timeseries, labels, attribute)
        else:
            return _create_sample_dict(normal_timeseries, normal_timeseries, np.zeros(seq_len), attribute)
    except Exception as e:
        logger.debug(f"Failed to generate univariate mixed sample: {e}")
        return None


def _generate_multivariate_sample(
    seq_len: int, 
    num_features: Optional[int], 
    anomaly_sample_ratio: float, 
    activate_function: bool,
    use_attribute_set: bool = False
) -> Optional[Dict]:
    """
    Generate a single multivariate sample.
    
    Parameters
    ----------
    seq_len : int
        Length of the time series.
    num_features : int, optional
        Number of features. If None, randomly determined.
    anomaly_sample_ratio : float
        Probability of including anomalies.
    activate_function : bool
        Whether to apply activation functions.
    use_attribute_set : bool, default=False
        Whether to use ALL_ATTRIBUTE_SET directly instead of synthetic.json.
    
    Returns
    -------
    dict or None
        Sample dictionary if successful, None otherwise.
    """
    try:
        normal_ts, abnormal_ts, labels, is_endogenous, attributes_list, dag = generate_multivariate_timeseries(
            seq_len, 
            num_nodes=num_features, 
            activate_fun=activate_function,
            use_attribute_set=use_attribute_set
        )
        
        attribute = {
            'attribute_list': attributes_list,
            'num_features': dag.number_of_nodes(),
            'is_endogenous': is_endogenous,
            'dag': str(dag.edges())
        }
        
        if random.random() < anomaly_sample_ratio:
            return _create_sample_dict(normal_ts, abnormal_ts, labels, attribute)
        else:
            attribute['is_endogenous'] = None
            return _create_sample_dict(normal_ts, normal_ts, np.zeros(seq_len), attribute)
    except Exception as e:
        logger.debug(f"Failed to generate multivariate sample: {e}")
        return None


def _generate_single_sample_worker(args):
    """
    Worker function for multiprocessing to generate a single sample.
    This function is called in separate processes.
    
    Parameters
    ----------
    args : tuple
        Tuple containing all parameters needed to generate a sample:
        (seq_len, is_multivariate, num_features, anomaly_sample_ratio, 
         activate_function, metrics, use_attribute_set, seed)
    
    Returns
    -------
    dict or None
        Sample dictionary if successful, None otherwise.
    """
    (seq_len, is_multivariate, num_features, anomaly_sample_ratio, 
     activate_function, metrics, use_attribute_set, seed) = args
    
    # Set random seed for this worker to ensure reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    try:
        if is_multivariate:
            return _generate_multivariate_sample(
                seq_len, num_features, anomaly_sample_ratio, activate_function, use_attribute_set
            )
        elif anomaly_sample_ratio == 1:
            return _generate_univariate_anomaly_sample(seq_len, metrics, use_attribute_set)
        else:
            return _generate_univariate_mixed_sample(seq_len, anomaly_sample_ratio, metrics, use_attribute_set)
    except Exception as e:
        logger.debug(f"Worker failed to generate sample: {e}")
        return None


def generate_dataset(
    num_samples: int = DEFAULT_NUM_SAMPLES, 
    seq_len: Optional[int] = None, 
    anomaly_sample_ratio: float = DEFAULT_ANOMALY_RATIO, 
    is_multivariate: bool = False, 
    num_features: Optional[int] = None, 
    activate_function: bool = False, 
    metrics: Optional[List] = None,
    use_attribute_set: bool = False,
    num_workers: Optional[int] = DEFAULT_NUM_WORKERS
) -> List[Dict]:
    """
    Generate a synthetic time series dataset for anomaly detection.

    Parameters
    ----------
    num_samples : int, default=1000
        Number of samples to generate.
    seq_len : int, optional
        Length of each time series sequence. If None, randomly generated.
    anomaly_sample_ratio : float, default=0.5
        Ratio of samples that will contain anomalies (0.0 to 1.0).
    is_multivariate : bool, default=False
        If True, generates multivariate time series data.
    num_features : int, optional
        Number of features per sample for multivariate data. 
        If None, different samples may have different number of features.
    activate_function : bool, default=False
        Whether to apply activation functions in multivariate generation.
    metrics : list, optional
        Specific metrics to use for generating anomalies.
        Only effective when use_attribute_set=False.
    use_attribute_set : bool, default=False
        If True, generate attributes directly from ALL_ATTRIBUTE_SET by weight.
        If False, select metric from synthetic.json to generate attributes.
        Using True is recommended for more flexible generation without JSON constraints.
    num_workers : int, optional
        Number of worker processes to use for parallel generation.
        If None, uses the number of CPU cores available.
        Set to 1 to disable multiprocessing.

    Returns
    -------
    list of dict
        A list of generated samples, each containing:
        - 'normal_time_series': Time series without anomalies
        - 'time_series': Time series with anomalies
        - 'labels': Binary labels indicating anomaly positions
        - 'attribute': Metadata about the time series attributes
    """
    # Input validation
    if num_samples <= 0:
        raise ValueError("Number of samples must be greater than 0")
    if seq_len is not None and seq_len <= 0:
        raise ValueError("Sequence length must be greater than 0")
    if not 0 <= anomaly_sample_ratio <= 1:
        raise ValueError("Anomaly sample ratio must be between 0 and 1")
    if num_workers is not None and num_workers <= 0:
        raise ValueError("Number of workers must be greater than 0 or None")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = mp.cpu_count()
    elif num_workers == 1:
        num_workers = None  # Disable multiprocessing
    
    # Log configuration
    logger.info(f"Starting dataset generation with {num_samples} samples")
    logger.info(f"Configuration: multivariate={is_multivariate}, seq_len={seq_len}, anomaly_ratio={anomaly_sample_ratio}")
    if num_workers:
        logger.info(f"Using {num_workers} worker processes for parallel generation")
    else:
        logger.info("Using single-threaded generation")
    
    if is_multivariate and num_features is None:
        logger.warning("Multivariate mode enabled without fixed num_features - samples may have different dimensions")

    dataset = []
    
    # Determine description for progress bar
    if is_multivariate:
        desc = f"Generating multivariate samples with {num_features} features" if num_features else "Generating multivariate samples"
    elif anomaly_sample_ratio == 1:
        desc = "Generating univariate anomaly samples"
    else:
        desc = "Generating univariate mixed samples"
    
    # Generate samples
    if num_workers is None:
        # Single-threaded generation (original method)
        count = 0
        pbar = tqdm(total=num_samples, desc=desc)
        
        while count < num_samples:
            current_seq_len = seq_len if seq_len is not None else random_seq_len()
            
            # Generate sample based on type
            if is_multivariate:
                sample = _generate_multivariate_sample(
                    current_seq_len, num_features, anomaly_sample_ratio, activate_function, use_attribute_set
                )
            elif anomaly_sample_ratio == 1:
                sample = _generate_univariate_anomaly_sample(current_seq_len, metrics, use_attribute_set)
            else:
                sample = _generate_univariate_mixed_sample(current_seq_len, anomaly_sample_ratio, metrics, use_attribute_set)
            
            # Add to dataset if generation succeeded
            if sample is not None:
                dataset.append(sample)
                count += 1
                pbar.update(1)
        
        pbar.close()
    else:
        # Multi-process generation with batched task submission
        # Key fix: submit tasks in batches instead of all at once to avoid memory/queue issues
        pbar = tqdm(total=num_samples, desc=desc)
        np.random.seed()  # Initialize random state
        
        # Batch size: submit tasks in chunks to avoid overwhelming the queue
        # Use a reasonable batch size based on number of workers
        batch_size = num_workers * 10  # 10 tasks per worker per batch
        
        completed = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            while completed < num_samples:
                # Prepare a batch of tasks
                current_batch_size = min(batch_size, (num_samples - completed) * 2)  # 2x buffer for failures
                batch_tasks = []
                
                for _ in range(current_batch_size):
                    current_seq_len = seq_len if seq_len is not None else random_seq_len()
                    seed = np.random.randint(0, 2**31)
                    task_args = (
                        current_seq_len, is_multivariate, num_features, anomaly_sample_ratio,
                        activate_function, metrics, use_attribute_set, seed
                    )
                    batch_tasks.append(task_args)
                
                # Submit batch and collect results
                futures = [executor.submit(_generate_single_sample_worker, task) for task in batch_tasks]
                
                for future in as_completed(futures):
                    if completed >= num_samples:
                        break
                    try:
                        sample = future.result()
                        if sample is not None:
                            dataset.append(sample)
                            completed += 1
                            pbar.update(1)
                    except Exception as e:
                        logger.debug(f"Task failed with exception: {e}")
        
        pbar.close()
    
    logger.info(f"Successfully generated {len(dataset)} samples")
    return dataset


if __name__ == "__main__":
    OUTPUT_DIR = '/mnt/hdd_data/anomaly_detection/RCD_data/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Dataset configuration
    num_samples = 400000
    seq_len = None
    anomaly_sample_ratio = 1.0
    is_multivariate = False
    num_features = 1
    use_attribute_set = True  # Recommended: use ALL_ATTRIBUTE_SET directly

    dataset = generate_dataset(
        num_samples=num_samples, 
        seq_len=seq_len, 
        anomaly_sample_ratio=anomaly_sample_ratio, 
        is_multivariate=is_multivariate, 
        num_features=num_features,
        use_attribute_set=use_attribute_set
    )
    
    mode_str = "attr_set" if use_attribute_set else "json"
    num_variate = f'{num_features if num_features is not None else "random_"}features' if is_multivariate else 'univariate'
    output_path = os.path.join(OUTPUT_DIR, f'dataset_{num_samples}samples_{seq_len}len_{anomaly_sample_ratio}ratio_{num_variate}_{mode_str}.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset generated and saved to '{output_path}'.")
    print(f"Generated {len(dataset)} samples with sequence length {seq_len} and anomaly sample ratio {anomaly_sample_ratio}.")
    print(f"Generation mode: {'ALL_ATTRIBUTE_SET (recommended)' if use_attribute_set else 'synthetic.json'}")
