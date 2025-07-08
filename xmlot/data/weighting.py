"""
Weighting utilities for survival analysis data.
This module re-exports weighting functions from xmlot.models.weighting for convenience.
"""

# Import all weighting functions from the models package
from xmlot.models.weighting import (
    compute_subgroup_weights,
    compute_survival_weights,
    add_weights_to_dataframe,
    analyze_subgroup_distribution
)

# Add a convenience function for adding sample weights to a DataFrame
def add_sample_weights(df, weights, weight_column_name="sample_weights"):
    """
    Add sample weights to a DataFrame.
    
    Args:
        df: DataFrame to add weights to
        weights: Array of weights to add
        weight_column_name: Name of the weight column
        
    Returns:
        DataFrame with weights added
    """
    df_copy = df.copy()
    df_copy[weight_column_name] = weights
    return df_copy

# Re-export all functions
__all__ = [
    'compute_subgroup_weights',
    'compute_survival_weights', 
    'add_weights_to_dataframe',
    'analyze_subgroup_distribution',
    'add_sample_weights'
] 