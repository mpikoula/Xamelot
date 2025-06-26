#!/usr/bin/env python3
"""
Verification script to ensure sample weights are NOT used as training features.
This script helps you verify that the weighted survival accessor is working correctly.
"""

import pandas as pd
import numpy as np
from xmlot.data.dataframes import build_weighted_survival_accessor
from xmlot.models.weighting import compute_subgroup_weights, add_weights_to_dataframe

def verify_weight_exclusion():
    """
    Verify that sample weights are excluded from training features.
    """
    print("=== Weight Exclusion Verification ===\n")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    # Create features
    df = pd.DataFrame({
        'age': np.random.normal(65, 15, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian'], n_samples, p=[0.7, 0.2, 0.1]),
        'stage': np.random.choice(['I', 'II', 'III'], n_samples),
        'duration': np.random.exponential(100, n_samples),
        'event': np.random.binomial(1, 0.7, n_samples)
    })
    
    print("1. Original DataFrame columns:")
    print(f"   {list(df.columns)}")
    print(f"   Total columns: {len(df.columns)}")
    
    # Add sample weights
    weights = compute_subgroup_weights(df, ['race'], weight_strategy="inverse_frequency")
    df_with_weights = add_weights_to_dataframe(df, weights, "sample_weights")
    
    print("\n2. DataFrame with weights added:")
    print(f"   {list(df_with_weights.columns)}")
    print(f"   Total columns: {len(df_with_weights.columns)}")
    print(f"   Weight column present: {'sample_weights' in df_with_weights.columns}")
    
    # Set up weighted survival accessor
    build_weighted_survival_accessor(
        event="event",
        duration="duration",
        weight_column="sample_weights",
        accessor_code="surv",
        exceptions=[]
    )
    
    # Get accessor
    accessor = df_with_weights.surv
    
    print("\n3. Accessor properties:")
    print(f"   Event column: {accessor.event}")
    print(f"   Duration column: {accessor.duration}")
    print(f"   Weight column: {accessor.m_weight_column}")
    print(f"   Weights available: {accessor.weights is not None}")
    
    # Check features
    print("\n4. Feature analysis:")
    print(f"   All DataFrame columns: {list(df_with_weights.columns)}")
    print(f"   Feature columns: {list(accessor.features.columns)}")
    print(f"   Target columns: {accessor.target}")
    
    # Verify weight exclusion
    weight_in_features = 'sample_weights' in accessor.features.columns
    print(f"\n5. Weight exclusion check:")
    print(f"   Weight column in features: {weight_in_features}")
    
    if not weight_in_features:
        print("   ✅ SUCCESS: Weight column is correctly excluded from features!")
    else:
        print("   ❌ ERROR: Weight column is incorrectly included in features!")
    
    # Check feature counts
    print(f"\n6. Feature counts:")
    print(f"   Total DataFrame columns: {len(df_with_weights.columns)}")
    print(f"   Feature columns: {len(accessor.features.columns)}")
    print(f"   Target columns: {len(accessor.target)}")
    print(f"   Weight column: 1")
    
    expected_features = len(df_with_weights.columns) - len(accessor.target) - 1  # -1 for weight column
    actual_features = len(accessor.features.columns)
    
    print(f"\n7. Feature count verification:")
    print(f"   Expected features: {expected_features} (total - targets - weight)")
    print(f"   Actual features: {actual_features}")
    
    if expected_features == actual_features:
        print("   ✅ SUCCESS: Feature count is correct!")
    else:
        print("   ❌ ERROR: Feature count mismatch!")
    
    # Show what's being excluded
    excluded_columns = set(df_with_weights.columns) - set(accessor.features.columns)
    print(f"\n8. Excluded columns:")
    print(f"   {list(excluded_columns)}")
    
    # Test the actual data extraction that the model uses
    print(f"\n9. Model data extraction test:")
    print("   Simulating what the model sees during training...")
    
    # This is what the model's _df_to_xyw_ method does
    x = accessor.features.to_numpy()
    y = (accessor.durations.values, accessor.events.values)
    w = accessor.weights.values if accessor.weights is not None else None
    
    print(f"   Features shape: {x.shape}")
    print(f"   Features columns: {list(accessor.features.columns)}")
    print(f"   Weights shape: {w.shape if w is not None else 'None'}")
    print(f"   Weights available: {w is not None}")
    
    # Final verification
    print(f"\n10. Final verification:")
    if not weight_in_features and expected_features == actual_features and w is not None:
        print("   ✅ ALL CHECKS PASSED: Weights are correctly excluded from features!")
        print("   ✅ Weights are available for training but not used as features!")
        return True
    else:
        print("   ❌ SOME CHECKS FAILED: Please review the setup!")
        return False

def test_with_your_data(df_train, weight_column="sample_weights"):
    """
    Test the verification with your actual training data.
    
    Args:
        df_train: Your training DataFrame
        weight_column: Name of your weight column
    """
    print(f"\n=== Testing with Your Data ===")
    print(f"DataFrame shape: {df_train.shape}")
    print(f"Columns: {list(df_train.columns)}")
    
    # Check if weight column exists
    if weight_column not in df_train.columns:
        print(f"❌ Weight column '{weight_column}' not found in DataFrame!")
        return False
    
    # Get accessor
    accessor = df_train.surv
    
    # Check features
    weight_in_features = weight_column in accessor.features.columns
    print(f"\nWeight column in features: {weight_in_features}")
    
    if not weight_in_features:
        print("✅ SUCCESS: Weight column is correctly excluded from features!")
        print(f"Feature columns: {list(accessor.features.columns)}")
        print(f"Feature count: {len(accessor.features.columns)}")
        return True
    else:
        print("❌ ERROR: Weight column is incorrectly included in features!")
        return False

if __name__ == "__main__":
    # Run the verification
    success = verify_weight_exclusion()
    
    if success:
        print("\n" + "="*50)
        print("VERIFICATION COMPLETE: Weights are correctly excluded!")
        print("Your weighted accessor is working properly.")
    else:
        print("\n" + "="*50)
        print("VERIFICATION FAILED: Please check your setup!") 