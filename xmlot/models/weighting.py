"""
Utilities for computing sample weights to address underrepresented subgroups
in survival analysis datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


def compute_subgroup_weights(
    df: pd.DataFrame,
    subgroup_columns: List[str],
    weight_strategy: str = "inverse_frequency",
    min_weight: float = 0.1,
    max_weight: float = 10.0,
    **kwargs
) -> np.ndarray:
    """
    Compute sample weights to address underrepresented subgroups.
    
    Args:
        df: DataFrame containing the data
        subgroup_columns: List of column names that define subgroups
        weight_strategy: Strategy for computing weights
            - "inverse_frequency": Weight inversely proportional to subgroup frequency
            - "balanced": Equal weights for all subgroups
            - "custom": Use custom weights provided in kwargs
        min_weight: Minimum weight to assign (prevents extreme weights)
        max_weight: Maximum weight to assign (prevents extreme weights)
        **kwargs: Additional arguments for custom weight strategies
        
    Returns:
        Array of sample weights
    """
    if not subgroup_columns:
        raise ValueError("subgroup_columns cannot be empty")
    
    # Create subgroup identifier
    subgroup_id = df[subgroup_columns].astype(str).agg('_'.join, axis=1)
    
    if weight_strategy == "inverse_frequency":
        return _compute_inverse_frequency_weights(
            subgroup_id, min_weight, max_weight
        )
    elif weight_strategy == "balanced":
        return _compute_balanced_weights(
            subgroup_id, min_weight, max_weight
        )
    elif weight_strategy == "custom":
        return _compute_custom_weights(
            subgroup_id, kwargs.get("custom_weights", {}), min_weight, max_weight
        )
    else:
        raise ValueError(f"Unknown weight_strategy: {weight_strategy}")


def _compute_inverse_frequency_weights(
    subgroup_id: pd.Series,
    min_weight: float,
    max_weight: float
) -> np.ndarray:
    """
    Compute weights inversely proportional to subgroup frequency.
    """
    # Count occurrences of each subgroup
    subgroup_counts = subgroup_id.value_counts()
    total_samples = len(subgroup_id)
    
    # Compute inverse frequency weights
    weights = np.zeros(len(subgroup_id))
    for subgroup, count in subgroup_counts.items():
        mask = subgroup_id == subgroup
        weights[mask] = total_samples / count
    
    # Normalize weights
    weights = weights / weights.mean()
    
    # Clip weights to specified range
    weights = np.clip(weights, min_weight, max_weight)
    
    return weights


def _compute_balanced_weights(
    subgroup_id: pd.Series,
    min_weight: float,
    max_weight: float
) -> np.ndarray:
    """
    Compute equal weights for all subgroups.
    """
    # Count occurrences of each subgroup
    subgroup_counts = subgroup_id.value_counts()
    max_count = subgroup_counts.max()
    
    # Compute balanced weights
    weights = np.zeros(len(subgroup_id))
    for subgroup, count in subgroup_counts.items():
        mask = subgroup_id == subgroup
        weights[mask] = max_count / count
    
    # Normalize weights
    weights = weights / weights.mean()
    
    # Clip weights to specified range
    weights = np.clip(weights, min_weight, max_weight)
    
    return weights


def _compute_custom_weights(
    subgroup_id: pd.Series,
    custom_weights: Dict[str, float],
    min_weight: float,
    max_weight: float
) -> np.ndarray:
    """
    Compute weights using custom weights provided by the user.
    """
    weights = np.ones(len(subgroup_id))
    
    for subgroup, weight in custom_weights.items():
        mask = subgroup_id == subgroup
        weights[mask] = weight
    
    # Normalize weights
    weights = weights / weights.mean()
    
    # Clip weights to specified range
    weights = np.clip(weights, min_weight, max_weight)
    
    return weights


def compute_survival_weights(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    subgroup_columns: Optional[List[str]] = None,
    weight_strategy: str = "inverse_frequency",
    time_bins: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Compute sample weights specifically for survival analysis, considering
    both subgroup representation and event rarity.
    
    Args:
        df: DataFrame containing the data
        duration_col: Name of the duration column
        event_col: Name of the event column
        subgroup_columns: List of column names that define subgroups
        weight_strategy: Strategy for computing weights
        time_bins: Number of time bins for temporal weighting (optional)
        **kwargs: Additional arguments for weight computation
        
    Returns:
        Array of sample weights
    """
    # Start with subgroup weights
    if subgroup_columns:
        subgroup_weights = compute_subgroup_weights(
            df, subgroup_columns, weight_strategy, **kwargs
        )
    else:
        subgroup_weights = np.ones(len(df))
    
    # Add event-based weighting
    event_weights = _compute_event_weights(df, duration_col, event_col, time_bins)
    
    # Combine weights
    combined_weights = subgroup_weights * event_weights
    
    # Normalize final weights
    combined_weights = combined_weights / combined_weights.mean()
    
    return combined_weights


def _compute_event_weights(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    time_bins: Optional[int] = None
) -> np.ndarray:
    """
    Compute weights based on event rarity and temporal distribution.
    """
    weights = np.ones(len(df))
    
    # Weight by event rarity
    event_counts = df[event_col].value_counts()
    total_samples = len(df)
    
    for event_type, count in event_counts.items():
        mask = df[event_col] == event_type
        weights[mask] *= total_samples / count
    
    # Add temporal weighting if requested
    if time_bins is not None:
        temporal_weights = _compute_temporal_weights(df, duration_col, event_col, time_bins)
        weights *= temporal_weights
    
    return weights


def _compute_temporal_weights(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    time_bins: int
) -> np.ndarray:
    """
    Compute weights based on temporal distribution of events.
    """
    # Create time bins
    duration_values = df[duration_col].values
    bins = np.linspace(duration_values.min(), duration_values.max(), time_bins + 1)
    bin_indices = np.digitize(duration_values, bins) - 1
    
    weights = np.ones(len(df))
    
    # Count events in each time bin
    for bin_idx in range(time_bins):
        bin_mask = bin_indices == bin_idx
        event_mask = df[event_col] != 0  # Non-censored events
        combined_mask = bin_mask & event_mask
        
        if combined_mask.sum() > 0:
            # Weight inversely to event frequency in this time bin
            bin_weight = len(df) / combined_mask.sum()
            weights[combined_mask] *= bin_weight
    
    return weights


def add_weights_to_dataframe(
    df: pd.DataFrame,
    weights: np.ndarray,
    weight_column_name: str = "sample_weights"
) -> pd.DataFrame:
    """
    Add computed weights to the DataFrame.
    
    Args:
        df: Original DataFrame
        weights: Array of sample weights
        weight_column_name: Name for the weight column
        
    Returns:
        DataFrame with weights added
    """
    df_with_weights = df.copy()
    df_with_weights[weight_column_name] = weights
    return df_with_weights


def analyze_subgroup_distribution(
    df: pd.DataFrame,
    subgroup_columns: List[str],
    duration_col: str,
    event_col: str
) -> pd.DataFrame:
    """
    Analyze the distribution of subgroups and events to help determine
    appropriate weighting strategies.
    
    Args:
        df: DataFrame containing the data
        subgroup_columns: List of column names that define subgroups
        duration_col: Name of the duration column
        event_col: Name of the event column
        
    Returns:
        DataFrame with subgroup analysis
    """
    # Create subgroup identifier
    subgroup_id = df[subgroup_columns].astype(str).agg('_'.join, axis=1)
    
    # Analyze each subgroup
    analysis_data = []
    for subgroup in subgroup_id.unique():
        mask = subgroup_id == subgroup
        subgroup_df = df[mask]
        
        total_count = len(subgroup_df)
        event_count = (subgroup_df[event_col] != 0).sum()
        censored_count = total_count - event_count
        mean_duration = subgroup_df[duration_col].mean()
        median_duration = subgroup_df[duration_col].median()
        
        analysis_data.append({
            'subgroup': subgroup,
            'total_count': total_count,
            'event_count': event_count,
            'censored_count': censored_count,
            'event_rate': event_count / total_count,
            'mean_duration': mean_duration,
            'median_duration': median_duration,
            'percentage_of_total': total_count / len(df) * 100
        })
    
    return pd.DataFrame(analysis_data).sort_values('total_count', ascending=False) 