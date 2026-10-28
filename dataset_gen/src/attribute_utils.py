"""
Attribute utilities for time series generation.

This module provides functions to load and map metrics to their corresponding
controlled attributes from the configuration file.

Copyright 2024 Tsinghua University and ByteDance.
Licensed under the MIT License.
"""

import json
from config import CONFIG_PATH

# Load metric configuration from JSON file
control_attribute_data = json.load(open(CONFIG_PATH))
metric_to_attributes = {}

# Build metric to attributes mapping
for category in control_attribute_data:
    for k, v in category['attributes'].items():
        metric_to_attributes[k] = v


def metric_to_controlled_attributes(metric: str):
    """
    Retrieve controlled attributes for a given metric.
    
    Parameters
    ----------
    metric : str
        The metric name to look up.
    
    Returns
    -------
    dict or None
        Attribute configuration for the specified metric, or None if not found.
    """
    return metric_to_attributes.get(metric, None)
