# Example of proper cross-validation with DeepHit models
# This avoids the feature count mismatch issue

from copy import deepcopy
import numpy as np
import pandas as pd
from xmlot.models.pycox import DeepHit
from xmlot.data.weighting import add_sample_weights, compute_survival_weights

# Example hyperparameters for DeepHit
DEEPHIT_HYPERPARAMETERS = {
    "in_features": None,  # Will be set dynamically based on data
    "num_nodes_shared": [64, 64],
    "num_nodes_indiv": [32],
    "num_risks": 1,
    "out_features": 50,  # Number of time bins
    "batch_norm": True,
    "dropout": 0.1,
    "alpha": 0.2,
    "sigma": 0.1,
    "cuts": np.linspace(0, 100, 51),  # Time discretization
    "use_weights": True,  # Enable sample weighting
    "seed": 42
}

# Training parameters
TRAINING_PARAMETERS = {
    "batch_size": 64,
    "epochs": 100,
    "callbacks": [],
    "verbose": True,
    "lr": None,  # Will be found automatically
    "tolerance": 10
}

def create_fresh_model_for_fold(model_class, hyperparameters, fold_data):
    """
    Create a fresh model instance for each fold with the correct feature count.
    
    Args:
        model_class: The model class (e.g., DeepHit)
        hyperparameters: Model hyperparameters
        fold_data: Training data for this fold (to determine feature count)
    
    Returns:
        Fresh model instance with correct feature count
    """
    # Determine the actual feature count from the fold data
    accessor = getattr(fold_data, 'surv')  # Assuming 'surv' is the accessor code
    actual_features = fold_data.surv.features.shape[1]
    
    # Create a copy of hyperparameters with correct feature count
    fold_hyperparameters = hyperparameters.copy()
    fold_hyperparameters["in_features"] = actual_features
    
    # Create fresh model instance
    return model_class(
        accessor_code='surv',
        hyperparameters=fold_hyperparameters
    )

def cross_validation_example():
    """
    Example of proper cross-validation that avoids feature count mismatches.
    """
    # Simulate your data preparation
    # In practice, this would be your actual data loading and preprocessing
    print("Setting up cross-validation...")
    
    # Example: Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create synthetic features (some categorical, some continuous)
    features = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features - 5)
    })
    
    # Add some categorical features that might cause encoding differences
    features['categorical_1'] = np.random.choice(['A', 'B', 'C'], n_samples)
    features['categorical_2'] = np.random.choice(['X', 'Y'], n_samples)
    features['categorical_3'] = np.random.choice(['Low', 'Medium', 'High'], n_samples)
    features['binary_1'] = np.random.choice([0, 1], n_samples)
    features['binary_2'] = np.random.choice([0, 1], n_samples)
    
    # Create survival data
    durations = np.random.exponential(50, n_samples)
    events = np.random.binomial(1, 0.7, n_samples)
    
    # Create DataFrame with survival accessor
    df = pd.DataFrame({
        'duration': durations,
        'event': events
    })
    df = pd.concat([df, features], axis=1)
    
    # Add sample weights
    df = add_sample_weights(
        df, 
        duration_col='duration', 
        event_col='event',
        weight_strategy='survival_specific'
    )
    
    # Set up weighted survival accessor
    df.surv.setup(
        duration_col='duration',
        event_col='event',
        weight_col='sample_weights'
    )
    
    # Split data into folds (simplified for example)
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    
    fold_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df)):
        print(f"\n=== Fold {fold_idx + 1} ===")
        
        # Split data
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()
        
        # Create FRESH model for this fold
        # This is the key difference from using deepcopy!
        model = create_fresh_model_for_fold(
            DeepHit, 
            DEEPHIT_HYPERPARAMETERS, 
            df_train
        )
        
        # Set up training parameters
        fold_parameters = TRAINING_PARAMETERS.copy()
        fold_parameters["val_data"] = df_val
        fold_parameters["seed"] = 42 + fold_idx
        
        print(f"Training model with {df_train.surv.features.shape[1]} features...")
        
        try:
            # Train the model
            model = model.fit(df_train, fold_parameters)
            
            # Evaluate (you would use your actual validation metric)
            # For this example, we'll just print a dummy score
            validation_score = 0.75 + np.random.normal(0, 0.05)  # Dummy score
            print(f"Validation score: {validation_score:.4f}")
            
            fold_scores.append(validation_score)
            
        except Exception as e:
            print(f"Error in fold {fold_idx + 1}: {e}")
            continue
    
    print(f"\n=== Cross-validation Results ===")
    print(f"Mean score: {np.mean(fold_scores):.4f}")
    print(f"Std score: {np.std(fold_scores):.4f}")
    print(f"All scores: {[f'{s:.4f}' for s in fold_scores]}")

if __name__ == "__main__":
    cross_validation_example() 