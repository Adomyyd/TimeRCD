# Time Series Anomaly Detection Dataset Generator

A synthetic dataset generator for time series anomaly detection tasks, supporting both univariate and multivariate time series with configurable anomaly patterns.

## Features

- **Univariate & Multivariate Support**: Generate single or multi-dimensional time series data
- **Flexible Anomaly Injection**: Control the ratio of anomalous samples in your dataset
- **Customizable Sequence Length**: Fixed or randomly distributed sequence lengths
- **Rich Metadata**: Each sample includes detailed attributes about the time series and anomalies
- **Two Generation Modes**: 
  - Attribute-based generation (recommended)
  - Metric-based generation using synthetic.json

## Installation

### Requirements

```bash
numpy
tqdm
networkx  # for multivariate generation
```

### Setup

```bash
pip install numpy tqdm networkx
```

## Quick Start

### Basic Usage

```python
from src.generate_dataset import generate_dataset
import pickle

# Generate 100 univariate samples with anomalies
dataset = generate_dataset(
    num_samples=100,
    seq_len=1000,
    anomaly_sample_ratio=1.0,
    is_multivariate=False
)

# Save the dataset
with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
```

### Multivariate Time Series

```python
# Generate multivariate time series with 5 features
dataset = generate_dataset(
    num_samples=100,
    seq_len=2500,
    anomaly_sample_ratio=1.0,
    is_multivariate=True,
    num_features=5,
    use_attribute_set=True  # Recommended
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_samples` | int | 1000 | Number of samples to generate |
| `seq_len` | int, optional | None | Length of each time series. If None, randomly generated (100-10000) |
| `anomaly_sample_ratio` | float | 0.5 | Ratio of samples containing anomalies (0.0-1.0) |
| `is_multivariate` | bool | False | Whether to generate multivariate time series |
| `num_features` | int, optional | None | Number of features for multivariate data |
| `activate_function` | bool | False | Apply activation functions in multivariate generation |
| `metrics` | list, optional | None | Specific metrics for anomaly generation |
| `use_attribute_set` | bool | False | Use ALL_ATTRIBUTE_SET for generation (recommended) |

## Output Format

Each sample in the dataset is a dictionary containing:

```python
{
    'normal_time_series': np.ndarray,  # Time series without anomalies
    'time_series': np.ndarray,         # Time series with anomalies injected
    'labels': np.ndarray,              # Binary labels (1 = anomaly, 0 = normal)
    'attribute': dict                  # Metadata about the time series
}
```

### Univariate Attribute Structure

```python
{
    'metric': str,           # Metric type
    'anomalies': list,       # List of anomaly descriptions
    'pattern': str,          # Base pattern type
    # ... other attributes
}
```

### Multivariate Attribute Structure

```python
{
    'attribute_list': list,  # Attributes for each feature
    'num_features': int,     # Number of features
    'is_endogenous': list,   # Which features have endogenous anomalies
    'dag': str              # Directed Acyclic Graph structure
}
```

## Examples

### Example 1: Anomaly-Only Dataset

```python
# Generate dataset with only anomalous samples
dataset = generate_dataset(
    num_samples=500,
    seq_len=1000,
    anomaly_sample_ratio=1.0,  # 100% anomalies
    is_multivariate=False,
    use_attribute_set=True
)
```

### Example 2: Mixed Dataset

```python
# Generate dataset with 50% anomalous and 50% normal samples
dataset = generate_dataset(
    num_samples=1000,
    anomaly_sample_ratio=0.5,
    is_multivariate=False
)
```

### Example 3: Variable-Length Sequences

```python
# Generate samples with random sequence lengths (100-10000)
dataset = generate_dataset(
    num_samples=200,
    seq_len=None,  # Random length
    anomaly_sample_ratio=0.8
)
```

## Running from Command Line

You can modify the configuration at the bottom of `generate_dataset.py` and run:

```bash
cd src
python generate_dataset.py
```

This will generate a dataset with the configured parameters and save it to `data_generate/samples/`.

## Project Structure

```
TSAD_dataset_gen/
├── src/
│   ├── generate_dataset.py      # Main dataset generation script
│   ├── ts_generator.py           # Univariate time series generator
│   └── ts_multi_generator.py    # Multivariate time series generator
├── data_generate/
│   └── samples/                  # Output directory for datasets
└── README.md
```

## Notes

- **Recommended Mode**: Use `use_attribute_set=True` for more flexible generation without JSON constraints
- **Memory Usage**: Large datasets with long sequences can consume significant memory
- **Generation Time**: Multivariate generation with many features may take longer
- **Anomaly Guarantee**: When `anomaly_sample_ratio=1.0`, the generator ensures all samples contain at least one anomaly

## Acknowledgements

This project references and adapts parts of the code from [ChatTS](https://github.com/NetManAIOps/ChatTS).

## License

This project is provided as-is for research and educational purposes.

