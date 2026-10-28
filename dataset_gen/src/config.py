"""
Configuration module for time series generation.

This module defines configuration parameters and attribute sets used for 
generating synthetic time series data with various patterns and anomalies.
"""

import os

# ==============================================================================
# Path Configuration
# ==============================================================================

DEFAULT_CONFIG_PATH = 'config/synthetic.json'
CONFIG_PATH = os.getenv('CHATTS_CONFIG_PATH', DEFAULT_CONFIG_PATH)

# ==============================================================================
# Attribute Set Configuration
# ==============================================================================

SEASONAL_FREQUENCY_WEIGHT = 10
ALL_ATTRIBUTE_SET = {
    # Overall time series characteristics
    "overall_attribute": {
        "seasonal": {
            "no periodic fluctuation": 0.1,
            "sin periodic fluctuation": 0.3,
            "square periodic fluctuation": 0.05,
            "triangle periodic fluctuation": 0.05,
            "wavelet periodic fluctuation": 0.3,
        },
        "trend": {
            "decrease": 0.2,
            "increase": 0.2,
            "keep steady": 0.2,
            "multiple": 0.3,
            "arima": 0.2
        },
        "frequency": {
            "high frequency": 0.3,
            "moderate frequency": 0.3,
            "low frequency": 0.4
        },
        "noise": {
            "low noise": 0.25,
            "moderate noise": 0.25,
            "high noise": 0.25,
            "almost no noise": 0.25,
        }
    },
    
    # Seasonal anomaly types
    "seasonal_anomalies": {
        "common": {
            "none": 10,
            "waveform_inversion": 5 * SEASONAL_FREQUENCY_WEIGHT,
            "amplitude_scaling": 5 * SEASONAL_FREQUENCY_WEIGHT,
            "noise_injection": 5 * SEASONAL_FREQUENCY_WEIGHT,
            "frequency_change": 5 * SEASONAL_FREQUENCY_WEIGHT
        },
        "sin": {
            "waveform_change": 5 * SEASONAL_FREQUENCY_WEIGHT,
            "phase_shift": 5 * SEASONAL_FREQUENCY_WEIGHT,
            "add_harmonic": 3 * SEASONAL_FREQUENCY_WEIGHT,
            "remove_harmonic": 3 * SEASONAL_FREQUENCY_WEIGHT,
            "modify_harmonic_phase": 3 * SEASONAL_FREQUENCY_WEIGHT,
            "modify_harmonic_amp_mod": 3 * SEASONAL_FREQUENCY_WEIGHT,
            "modify_harmonic_mod_freq": 3 * SEASONAL_FREQUENCY_WEIGHT,
            "modify_harmonic_mod_phase": 3 * SEASONAL_FREQUENCY_WEIGHT
        },
        "square": {
            "pulse_shift": 5 * SEASONAL_FREQUENCY_WEIGHT,
            "pulse_width_modulation": 5 * SEASONAL_FREQUENCY_WEIGHT
        },
        "triangle": {
            "pulse_shift": 5 * SEASONAL_FREQUENCY_WEIGHT,
            "pulse_width_modulation": 5 * SEASONAL_FREQUENCY_WEIGHT
        },
        "wavelet": {
            "family_change": 5 * SEASONAL_FREQUENCY_WEIGHT,
            "scale_change": 5 * SEASONAL_FREQUENCY_WEIGHT,
            "shift_change": 5 * SEASONAL_FREQUENCY_WEIGHT,
            "amplitude_change": 5 * SEASONAL_FREQUENCY_WEIGHT,
            "add_wavelet": 3 * SEASONAL_FREQUENCY_WEIGHT,
            "remove_wavelet": 3 * SEASONAL_FREQUENCY_WEIGHT
        }
    },

    # Local/point anomaly types
    "change": {
        "shake": 4,
        "upward spike": 12,
        "downward spike": 10,
        "continuous upward spike": 4,
        "continuous downward spike": 2,
        "upward convex": 2,
        "downward convex": 2,
        "sudden increase": 2,
        "sudden decrease": 2,
        "rapid rise followed by slow decline": 2,
        "slow rise followed by rapid decline": 2,
        "rapid decline followed by slow rise": 2,
        "slow decline followed by rapid rise": 2,
        "decrease after upward spike": 3,
        "increase after downward spike": 3,
        "increase after upward spike": 3,
        "decrease after downward spike": 3,
        "wide upward spike": 3,
        "wide downward spike": 3,
        "outlier": 40
    }
}
