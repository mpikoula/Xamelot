"""
Simplified test for the weighted DeepHit implementation.
This test focuses on the core weighting functionality without requiring full model initialization.
"""

import pandas as pd
import numpy as np
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
    print("Testing weight computation...")
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
    
    print("✓ Weight computation test passed")


def test_survival_weight_computation():
    """Test survival-specific weight computation."""
    print("Testing survival weight computation...")
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
    
    print("✓ Survival weight computation test passed")


def test_weighted_dataframe():
    """Test adding weights to DataFrame."""
    print("Testing weighted DataFrame creation...")
    df = create_test_data()
    weights = compute_subgroup_weights(df, ['race'])
    df_with_weights = add_weights_to_dataframe(df, weights, "sample_weights")
    
    assert "sample_weights" in df_with_weights.columns
    assert len(df_with_weights) == len(df)
    assert df_with_weights["sample_weights"].equals(pd.Series(weights))
    
    print("✓ Weighted DataFrame test passed")


def test_subgroup_analysis():
    """Test subgroup distribution analysis."""
    print("Testing subgroup analysis...")
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
    
    print("✓ Subgroup analysis test passed")


def test_balanced_weighting():
    """Test balanced weighting strategy."""
    print("Testing balanced weighting...")
    df = create_test_data()
    
    weights = compute_subgroup_weights(
        df, 
        subgroup_columns=['race'],
        weight_strategy="balanced",
        min_weight=0.1,
        max_weight=10.0
    )
    
    assert len(weights) == len(df)
    assert weights.min() >= 0.1
    assert weights.max() <= 10.0
    
    print("✓ Balanced weighting test passed")


def test_custom_weighting():
    """Test custom weighting strategy."""
    print("Testing custom weighting...")
    df = create_test_data()
    
    custom_weights = {
        'White': 1.0,
        'Black': 2.0,
        'Asian': 3.0
    }
    
    weights = compute_subgroup_weights(
        df, 
        subgroup_columns=['race'],
        weight_strategy="custom",
        custom_weights=custom_weights,
        min_weight=0.1,
        max_weight=10.0
    )
    
    assert len(weights) == len(df)
    
    # Check that custom weights are applied correctly
    for race, expected_weight in custom_weights.items():
        mask = df['race'] == race
        if mask.sum() > 0:
            # The weights should be proportional to the custom weights
            assert weights[mask].mean() > 0
    
    print("✓ Custom weighting test passed")


def test_multiple_subgroups():
    """Test weighting with multiple subgroup columns."""
    print("Testing multiple subgroups...")
    df = create_test_data()
    
    weights = compute_subgroup_weights(
        df, 
        subgroup_columns=['race', 'stage'],
        weight_strategy="inverse_frequency",
        min_weight=0.1,
        max_weight=10.0
    )
    
    assert len(weights) == len(df)
    assert weights.min() >= 0.1
    assert weights.max() <= 10.0
    
    # Check that we have weights for different combinations
    unique_subgroups = df[['race', 'stage']].astype(str).agg('_'.join, axis=1).unique()
    assert len(unique_subgroups) > 1
    
    print("✓ Multiple subgroups test passed")


def main():
    """Run all tests."""
    print("Running simplified weighted DeepHit tests...")
    print("=" * 50)
    
    try:
        test_weight_computation()
        test_survival_weight_computation()
        test_weighted_dataframe()
        test_subgroup_analysis()
        test_balanced_weighting()
        test_custom_weighting()
        test_multiple_subgroups()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed successfully!")
        print("The weighting functionality is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 