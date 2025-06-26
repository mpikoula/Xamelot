"""
Test the weighted DeepHit implementation.
"""

import pytest
import pandas as pd
import numpy as np
from xmlot.data.dataframes import build_weighted_survival_accessor
from xmlot.models.pycox import DeepHit
from xmlot.models.weighting import (
    compute_subgroup_weights,
    compute_survival_weights,
    add_weights_to_dataframe,
    analyze_subgroup_distribution
)


def create_test_data():
    """Create a small test dataset with underrepresented subgroups."""
    np.random.seed(42)
    
    # Create imbalanced data
    n_samples = 100
    race = np.random.choice(['White', 'Black', 'Asian'], n_samples, p=[0.7, 0.2, 0.1])
    stage = np.random.choice(['I', 'II', 'III'], n_samples, p=[0.5, 0.3, 0.2])
    
    # Create features
    age = np.random.normal(65, 15, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    
    # Create survival data
    durations = np.random.exponential(100, n_samples)
    events = np.random.binomial(1, 0.7, n_samples)
    
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'race': race,
        'stage': stage,
        'duration': durations,
        'event': events
    })
    
    return df


def test_weight_computation():
    """Test that sample weights are computed correctly."""
    df = create_test_data()
    
    # Test inverse frequency weighting
    weights = compute_subgroup_weights(
        df, 
        subgroup_columns=['race'],
        weight_strategy="inverse_frequency",
        min_weight=0.1,
        max_weight=10.0
    )
    
    assert len(weights) == len(df)
    assert weights.min() >= 0.1
    assert weights.max() <= 10.0
    
    # Test that underrepresented groups get higher weights
    asian_mask = df['race'] == 'Asian'
    white_mask = df['race'] == 'White'
    
    if asian_mask.sum() > 0 and white_mask.sum() > 0:
        assert weights[asian_mask].mean() > weights[white_mask].mean()


def test_survival_weight_computation():
    """Test survival-specific weight computation."""
    df = create_test_data()
    
    weights = compute_survival_weights(
        df,
        duration_col='duration',
        event_col='event',
        subgroup_columns=['race'],
        weight_strategy="inverse_frequency",
        time_bins=5
    )
    
    assert len(weights) == len(df)
    assert weights.min() > 0
    assert weights.max() > 1  # Should have some weights > 1


def test_weighted_accessor():
    """Test the weighted survival accessor."""
    df = create_test_data()
    weights = compute_subgroup_weights(df, ['race'])
    df_with_weights = add_weights_to_dataframe(df, weights, "sample_weights")
    
    # Build accessor
    build_weighted_survival_accessor(
        event="event",
        duration="duration",
        weight_column="sample_weights",
        accessor_code="surv",
        exceptions=[]
    )
    
    # Test accessor properties
    accessor = df_with_weights.surv
    
    assert accessor.event == "event"
    assert accessor.duration == "duration"
    assert accessor.weights is not None
    assert len(accessor.weights) == len(df)
    assert "sample_weights" not in accessor.features.columns  # Should be excluded from features


def test_subgroup_analysis():
    """Test subgroup distribution analysis."""
    df = create_test_data()
    
    analysis = analyze_subgroup_distribution(
        df, 
        subgroup_columns=['race'], 
        duration_col='duration', 
        event_col='event'
    )
    
    assert len(analysis) > 0
    assert 'subgroup' in analysis.columns
    assert 'total_count' in analysis.columns
    assert 'event_rate' in analysis.columns
    
    # Check that all races are included
    expected_races = ['White', 'Black', 'Asian']
    for race in expected_races:
        assert race in analysis['subgroup'].values


def test_weighted_deephit_initialization():
    """Test that weighted DeepHit can be initialized correctly."""
    # This is a basic test to ensure the model can be created
    # Full training would require more setup and time
    
    hyperparameters = {
        "in_features": 4,
        "num_nodes_shared": [32],
        "num_nodes_indiv": [16],
        "num_risks": 1,
        "out_features": 10,
        "cuts": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "batch_norm": True,
        "dropout": 0.1,
        "alpha": 0.2,
        "sigma": 0.1,
        "seed": 42,
        "use_weights": True
    }
    
    # Should not raise an error
    model = DeepHit(
        accessor_code="surv",
        hyperparameters=hyperparameters
    )
    
    assert model.use_weights == True
    assert hasattr(model, 'm_model')


def test_weighted_deephit_backward_compatibility():
    """Test that standard DeepHit still works without weights."""
    hyperparameters = {
        "in_features": 4,
        "num_nodes_shared": [32],
        "num_nodes_indiv": [16],
        "num_risks": 1,
        "out_features": 10,
        "cuts": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "batch_norm": True,
        "dropout": 0.1,
        "alpha": 0.2,
        "sigma": 0.1,
        "seed": 42,
        "use_weights": False  # Standard mode
    }
    
    model = DeepHit(
        accessor_code="surv",
        hyperparameters=hyperparameters
    )
    
    assert model.use_weights == False
    assert hasattr(model, 'm_model')


if __name__ == "__main__":
    # Run tests
    test_weight_computation()
    test_survival_weight_computation()
    test_weighted_accessor()
    test_subgroup_analysis()
    test_weighted_deephit_initialization()
    test_weighted_deephit_backward_compatibility()
    print("All tests passed!") 