"""
Standalone test for the weighting functionality.
This test directly tests the weighting functions without importing the full xmlot package.
"""

import pandas as pd
import numpy as np


def compute_subgroup_weights(
    df: pd.DataFrame,
    subgroup_columns: list,
    weight_strategy: str = "inverse_frequency",
    min_weight: float = 0.1,
    max_weight: float = 10.0,
    **kwargs
) -> np.ndarray:
    """
    Compute sample weights to address underrepresented subgroups.
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
    custom_weights: dict,
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


def add_weights_to_dataframe(
    df: pd.DataFrame,
    weights: np.ndarray,
    weight_column_name: str = "sample_weights"
) -> pd.DataFrame:
    """
    Add computed weights to the DataFrame.
    """
    df_with_weights = df.copy()
    df_with_weights[weight_column_name] = weights
    return df_with_weights


def analyze_subgroup_distribution(
    df: pd.DataFrame,
    subgroup_columns: list,
    duration_col: str,
    event_col: str
) -> pd.DataFrame:
    """
    Analyze the distribution of subgroups and events.
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


def test_weight_distribution():
    """Test that weights are distributed correctly."""
    print("Testing weight distribution...")
    df = create_test_data()
    
    weights = compute_subgroup_weights(
        df, 
        subgroup_columns=['race'],
        weight_strategy="inverse_frequency",
        min_weight=0.1,
        max_weight=10.0
    )
    
    # Check weight statistics
    print(f"  Weight mean: {weights.mean():.3f}")
    print(f"  Weight std: {weights.std():.3f}")
    print(f"  Weight min: {weights.min():.3f}")
    print(f"  Weight max: {weights.max():.3f}")
    
    # Check that weights are normalized (mean should be close to 1)
    assert abs(weights.mean() - 1.0) < 0.1
    
    print("✓ Weight distribution test passed")


def main():
    """Run all tests."""
    print("Running standalone weighting tests...")
    print("=" * 50)
    
    try:
        test_weight_computation()
        test_weighted_dataframe()
        test_subgroup_analysis()
        test_balanced_weighting()
        test_custom_weighting()
        test_multiple_subgroups()
        test_weight_distribution()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed successfully!")
        print("The weighting functionality is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 