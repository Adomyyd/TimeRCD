"""
Trend generation utilities for time series.

This module provides functions to generate various trend patterns including
multi-point curves, ARIMA processes, and trend descriptions.

Copyright 2024 Tsinghua University and ByteDance.
Licensed under the MIT License.
"""

import math
import random
import numpy as np
from scipy.interpolate import PchipInterpolator
from statsmodels.tsa.arima_process import ArmaProcess


def generate_random_points(seq_len):
    """
    Generate random key points for curve interpolation.

    Parameters
    ----------
    seq_len : int
        Total length of the sequence.

    Returns
    -------
    points : list of tuple
        List of (x, y) coordinate tuples representing key points.
    curve_type : str
        Type of curve to use ('Bezier' or 'Straight Line').
    """
    min_distance = math.ceil(seq_len / 8)
    num_turning_points = random.randint(1, 5)
    total_key_points = 2 + num_turning_points
    total_min_distance = (total_key_points - 1) * min_distance
    total_distance = seq_len - 1
    extra_distance = total_distance - total_min_distance
    
    # Adjust number of turning points if sequence is too short
    while extra_distance < 0 and num_turning_points > 0:
        num_turning_points -= 1
        total_key_points = 2 + num_turning_points
        total_min_distance = (total_key_points - 1) * min_distance
        extra_distance = total_distance - total_min_distance
    
    if extra_distance < 0:
        raise ValueError("Sequence length is too small for trend generation")
    
    # Distribute gaps between key points
    gaps = [min_distance] * (total_key_points - 1)
    for _ in range(extra_distance):
        idx = random.randint(0, total_key_points - 2)
        gaps[idx] += 1
    
    # Generate x coordinates
    key_x = [0]
    for gap in gaps:
        key_x.append(key_x[-1] + gap)
    
    # Generate y coordinates with random distribution
    y_positions = np.random.uniform(-1, 1, total_key_points)
    if random.random() < 0.5:
        y_positions = np.sqrt(np.abs(y_positions)) * np.sign(y_positions)
    
    points = list(zip(key_x, y_positions))
    curve_type = "Bezier" if random.random() < 0.99 else "Straight Line"
    
    return points, curve_type


def generate_trend_curve(seq_len, points):
    """
    Generate a smooth curve through key points using interpolation.

    Parameters
    ----------
    seq_len : int
        Total length of the sequence.
    points : list of tuple
        List of (x, y) coordinate tuples.

    Returns
    -------
    curve_x : ndarray
        X-coordinates of the curve (0 to seq_len-1).
    curve_y : ndarray
        Y-coordinates of the curve.
    curve_type : str
        Type of interpolation used ('Bezier' or 'Straight Line').
    """
    key_x = [point[0] for point in points]
    key_y = [point[1] for point in points]
    
    curve_x = np.arange(seq_len)
    if random.random() < 0.99:
        curve_type = "Bezier"
        interpolator = PchipInterpolator(key_x, key_y)
        curve_y = interpolator(curve_x)
    else:
        curve_type = "Straight Line"
        curve_y = np.interp(curve_x, key_x, key_y)
    
    return curve_x, curve_y, curve_type


def generate_trend_prompt(points):
    """
    Generate an English description of the trend between key points.

    Parameters
    ----------
    points : list of tuple
        List of (x, y) coordinate tuples.

    Returns
    -------
    prompt : str
        English description of the trend pattern.
    """
    if not points or len(points) < 2:
        return "Insufficient points to determine trends."

    y_values = [y for _, y in points]
    curve_range = max(y_values) - min(y_values)
    
    if curve_range == 0:
        curve_range = 1

    # Determine trends between consecutive points
    trends = []
    for i in range(len(points) - 1):
        _, y_left = points[i]
        _, y_right = points[i + 1]
        delta_y = y_right - y_left

        if delta_y > 0.1 * curve_range:
            trend = "increasing"
        elif delta_y < -0.1 * curve_range:
            trend = "decreasing"
        else:
            trend = "stable"

        trends.append(trend)

    # Merge consecutive identical trends
    merged_trends = []
    current_trend = trends[0]
    start_idx = 0

    for i in range(1, len(trends)):
        if trends[i] != current_trend:
            merged_trends.append((current_trend, start_idx, i))
            current_trend = trends[i]
            start_idx = i
    merged_trends.append((current_trend, start_idx, len(trends)))

    # Generate prompt segments
    prompt_segments = []
    for trend, start, end in merged_trends:
        point_start = points[start]
        point_end = points[end]

        if trend == "increasing":
            article = "an increasing trend"
        elif trend == "decreasing":
            article = "a decreasing trend"
        else:
            article = "a stable trend"

        if end - start > 1:
            variation_note = "with some variation in slope"
        else:
            variation_note = ""

        if variation_note:
            sentence = (f"From point {point_start[0]} to point {point_end[0]}, "
                       f"there is {article} {variation_note}.")
        else:
            sentence = (f"From point {point_start[0]} to point {point_end[0]}, "
                       f"there is {article}.")

        prompt_segments.append(sentence)

    prompt = " ".join(prompt_segments)
    return prompt


def generate_trend_list(points, seq_len):
    """
    Generate a structured list of trend segments.

    Parameters
    ----------
    points : list of tuple
        List of (x, y) coordinate tuples.
    seq_len : int
        Total length of the sequence.

    Returns
    -------
    list of tuple
        Each tuple contains (trend_type, start_point, end_point).
        trend_type is one of 'increase', 'decrease', or 'steady'.
    """
    if not points or len(points) < 2:
        return []

    y_values = [y for _, y in points]
    curve_range = max(y_values) - min(y_values)
    
    if curve_range == 0:
        curve_range = 1

    # Determine trends between consecutive points
    trends = []
    for i in range(len(points) - 1):
        _, y_left = points[i]
        _, y_right = points[i + 1]
        delta_y = y_right - y_left

        if delta_y > 0.1 * curve_range:
            trend = "increase"
        elif delta_y < -0.1 * curve_range:
            trend = "decrease"
        else:
            trend = "steady"

        trends.append(trend)

    # Merge consecutive identical trends
    merged_trends = []
    current_trend = trends[0]
    start_idx = 0

    for i in range(1, len(trends)):
        if trends[i] != current_trend:
            merged_trends.append((current_trend, points[start_idx][0], points[i][0]))
            current_trend = trends[i]
            start_idx = i
    merged_trends.append((current_trend, points[start_idx][0], seq_len - 1))

    return merged_trends


def generate_arima_series(length: int, noise_std: float = 1):
    """
    Generate a time series using randomly selected ARIMA parameters.

    Parameters
    ----------
    length : int
        Length of the time series to generate.
    noise_std : float, default=1
        Standard deviation of the noise component.

    Returns
    -------
    integrated_series : ndarray
        Generated ARIMA time series.
    params : dict
        Dictionary containing the ARIMA parameters (p, d, q) used.
    """
    p_choices = [0, 1, 2]
    d_choices = [0, 1, 2]
    q_choices = [0, 1, 2]

    # Ensure at least one parameter is non-zero
    while True:
        p = random.choice(p_choices)
        d = random.choice(d_choices)
        q = random.choice(q_choices)
        if p != 0 or d != 0 or q != 0:
            break

    params = {'p': p, 'd': d, 'q': q}

    # Generate stationary and invertible ARMA process
    while True:
        ar_params = np.random.uniform(low=-2, high=2, size=2)
        ar = np.r_[1, -ar_params]

        ma_params = np.random.uniform(low=-2, high=2, size=2)
        ma = np.r_[1, ma_params]

        arma_process = ArmaProcess(ar=ar, ma=ma)
        if arma_process.isstationary and arma_process.isinvertible:
            # Adjust noise based on integration order
            if d == 0:
                noise_std = 1.0
            elif d == 1:
                noise_std = 1.5
            else:
                noise_std = 100.0
            arma_series = arma_process.generate_sample(nsample=length, scale=noise_std, burnin=100)
            break

    # Apply integration
    integrated_series = arma_series
    if d > 0:
        for _ in range(d):
            integrated_series = np.cumsum(integrated_series)

    return integrated_series, params
