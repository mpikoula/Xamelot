"""
Example: Using Weighted DeepHit for Addressing Underrepresented Subgroups

This example demonstrates how to use the modified DeepHit model with sample weights
to improve performance on underrepresented subgroups in survival analysis datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xmlot.data.dataframes import build_weighted_survival_accessor
from xmlot.data.discretise import BalancedDiscretiser
from xmlot.models.pycox import DeepHit
from xmlot.models.weighting import (
    compute_subgroup_weights,
    compute_survival_weights,
    add_weights_to_dataframe,
    analyze_subgroup_distribution
)
from xmlot.metrics.survival import concordance
import torchtuples as tt


def create_sample_data(n_samples=1000):
    """
    Create a sample survival dataset with underrepresented subgroups.
    """
    np.random.seed(42)
    
    # Create features
    age = np.random.normal(65, 15, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples, p=[0.7, 0.15, 0.1, 0.05])
    stage = np.random.choice(['I', 'II', 'III', 'IV'], n_samples, p=[0.3, 0.3, 0.25, 0.15])
    
    # Create survival times with different hazards for different subgroups
    # Underrepresented groups (e.g., Asian patients) will have different survival patterns
    base_hazard = 0.01
    hazard_multipliers = {
        'White': 1.0,
        'Black': 1.2,
        'Hispanic': 1.1,
        'Asian': 0.8  # Lower hazard for Asian patients
    }
    
    hazards = np.array([base_hazard * hazard_multipliers[r] for r in race])
    durations = np.random.exponential(1/hazards)
    
    # Add some censoring
    censoring_times = np.random.exponential(200, n_samples)
    events = (durations <= censoring_times).astype(int)
    durations = np.minimum(durations, censoring_times)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'race': race,
        'stage': stage,
        'duration': durations,
        'event': events
    })
    
    return df


def analyze_data_imbalance(df):
    """
    Analyze the distribution of subgroups in the dataset.
    """
    print("=== Subgroup Distribution Analysis ===")
    
    # Analyze by race
    race_analysis = analyze_subgroup_distribution(
        df, 
        subgroup_columns=['race'], 
        duration_col='duration', 
        event_col='event'
    )
    print("\nRace distribution:")
    print(race_analysis)
    
    # Analyze by stage
    stage_analysis = analyze_subgroup_distribution(
        df, 
        subgroup_columns=['stage'], 
        duration_col='duration', 
        event_col='event'
    )
    print("\nStage distribution:")
    print(stage_analysis)
    
    # Analyze by race and stage combination
    combined_analysis = analyze_subgroup_distribution(
        df, 
        subgroup_columns=['race', 'stage'], 
        duration_col='duration', 
        event_col='event'
    )
    print("\nRace + Stage distribution (top 10):")
    print(combined_analysis.head(10))
    
    return race_analysis, stage_analysis, combined_analysis


def compute_weights_for_subgroups(df, subgroup_columns=['race', 'stage']):
    """
    Compute sample weights to address underrepresented subgroups.
    """
    print("\n=== Computing Sample Weights ===")
    
    # Method 1: Inverse frequency weighting
    weights_inverse = compute_subgroup_weights(
        df, 
        subgroup_columns=subgroup_columns,
        weight_strategy="inverse_frequency",
        min_weight=0.1,
        max_weight=10.0
    )
    
    # Method 2: Balanced weighting
    weights_balanced = compute_subgroup_weights(
        df, 
        subgroup_columns=subgroup_columns,
        weight_strategy="balanced",
        min_weight=0.1,
        max_weight=10.0
    )
    
    # Method 3: Survival-specific weighting
    weights_survival = compute_survival_weights(
        df,
        duration_col='duration',
        event_col='event',
        subgroup_columns=subgroup_columns,
        weight_strategy="inverse_frequency",
        time_bins=10
    )
    
    # Add weights to DataFrame
    df_with_weights = add_weights_to_dataframe(df, weights_survival, "sample_weights")
    
    # Analyze weight distribution
    print(f"\nWeight statistics:")
    print(f"Mean weight: {weights_survival.mean():.3f}")
    print(f"Std weight: {weights_survival.std():.3f}")
    print(f"Min weight: {weights_survival.min():.3f}")
    print(f"Max weight: {weights_survival.max():.3f}")
    
    # Show weight distribution by race
    weight_by_race = df_with_weights.groupby('race')['sample_weights'].agg(['mean', 'std', 'count'])
    print(f"\nWeight distribution by race:")
    print(weight_by_race)
    
    return df_with_weights, weights_inverse, weights_balanced, weights_survival


def train_models(df_train, df_val, df_test):
    """
    Train both standard and weighted DeepHit models.
    """
    print("\n=== Training Models ===")
    
    # Set up discretization
    SIZE_GRID = 50
    DISCRETISER = BalancedDiscretiser(df_train, "surv", SIZE_GRID)
    
    # Prepare data
    data_train = DISCRETISER(df_train.copy())
    data_val = DISCRETISER(df_val.copy())
    data_test = DISCRETISER(df_test.copy())
    
    # Common hyperparameters
    common_params = {
        "in_features": data_train.surv.features.shape[1],
        "num_nodes_shared": [64, 64],
        "num_nodes_indiv": [32],
        "num_risks": 1,
        "out_features": SIZE_GRID,
        "cuts": DISCRETISER.grid,
        "batch_norm": True,
        "dropout": 0.1,
        "alpha": 0.2,
        "sigma": 0.1,
        "seed": 42,
    }
    
    # Training parameters
    train_params = {
        'batch_size': 256,
        'epochs': 50,
        'callbacks': [tt.callbacks.EarlyStopping(patience=10)],
        'verbose': True,
        'val_data': data_val,
        'lr': None,
        'tolerance': 10
    }
    
    # Model 1: Standard DeepHit (no weights)
    print("Training standard DeepHit model...")
    model_standard = DeepHit(
        accessor_code="surv",
        hyperparameters={**common_params, "use_weights": False}
    )
    model_standard.fit(data_train, train_params)
    
    # Model 2: Weighted DeepHit
    print("\nTraining weighted DeepHit model...")
    model_weighted = DeepHit(
        accessor_code="surv",
        hyperparameters={**common_params, "use_weights": True}
    )
    model_weighted.fit(data_train, train_params)
    
    return model_standard, model_weighted, data_test


def evaluate_models(model_standard, model_weighted, data_test):
    """
    Evaluate both models on the test set, with special attention to underrepresented groups.
    """
    print("\n=== Model Evaluation ===")
    
    # Overall performance
    concordance_standard = concordance(model_standard, data_test)
    concordance_weighted = concordance(model_weighted, data_test)
    
    print(f"Overall Concordance Index:")
    print(f"  Standard DeepHit: {concordance_standard:.4f}")
    print(f"  Weighted DeepHit: {concordance_weighted:.4f}")
    print(f"  Improvement: {concordance_weighted - concordance_standard:.4f}")
    
    # Performance by race
    races = data_test['race'].unique()
    print(f"\nPerformance by race:")
    for race in races:
        mask = data_test['race'] == race
        subset = data_test[mask]
        
        if len(subset) > 10:  # Only evaluate if enough samples
            concordance_std = concordance(model_standard, subset)
            concordance_wgt = concordance(model_weighted, subset)
            
            print(f"  {race}:")
            print(f"    Standard: {concordance_std:.4f}")
            print(f"    Weighted: {concordance_wgt:.4f}")
            print(f"    Improvement: {concordance_wgt - concordance_std:.4f}")
            print(f"    Sample size: {len(subset)}")
    
    # Performance by stage
    stages = data_test['stage'].unique()
    print(f"\nPerformance by stage:")
    for stage in stages:
        mask = data_test['stage'] == stage
        subset = data_test[mask]
        
        if len(subset) > 10:  # Only evaluate if enough samples
            concordance_std = concordance(model_standard, subset)
            concordance_wgt = concordance(model_weighted, subset)
            
            print(f"  Stage {stage}:")
            print(f"    Standard: {concordance_std:.4f}")
            print(f"    Weighted: {concordance_wgt:.4f}")
            print(f"    Improvement: {concordance_wgt - concordance_std:.4f}")
            print(f"    Sample size: {len(subset)}")
    
    return concordance_standard, concordance_weighted


def plot_results(df_with_weights, concordance_standard, concordance_weighted):
    """
    Create visualizations of the results.
    """
    print("\n=== Creating Visualizations ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Weight distribution by race
    weight_by_race = df_with_weights.groupby('race')['sample_weights'].mean()
    axes[0, 0].bar(weight_by_race.index, weight_by_race.values)
    axes[0, 0].set_title('Average Sample Weights by Race')
    axes[0, 0].set_ylabel('Average Weight')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Sample counts by race
    race_counts = df_with_weights['race'].value_counts()
    axes[0, 1].bar(race_counts.index, race_counts.values)
    axes[0, 1].set_title('Sample Counts by Race')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Weight distribution histogram
    axes[1, 0].hist(df_with_weights['sample_weights'], bins=30, alpha=0.7)
    axes[1, 0].set_title('Distribution of Sample Weights')
    axes[1, 0].set_xlabel('Weight')
    axes[1, 0].set_ylabel('Frequency')
    
    # Plot 4: Model performance comparison
    models = ['Standard\nDeepHit', 'Weighted\nDeepHit']
    concordances = [concordance_standard, concordance_weighted]
    colors = ['lightcoral', 'lightblue']
    
    bars = axes[1, 1].bar(models, concordances, color=colors)
    axes[1, 1].set_title('Model Performance Comparison')
    axes[1, 1].set_ylabel('Concordance Index')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, concordances):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the complete example.
    """
    print("Weighted DeepHit Example: Addressing Underrepresented Subgroups")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("Step 1: Creating sample data...")
    df = create_sample_data(n_samples=2000)
    
    # Step 2: Set up accessor
    build_weighted_survival_accessor(
        event="event",
        duration="duration",
        weight_column="sample_weights",
        accessor_code="surv",
        exceptions=[]
    )
    
    # Step 3: Analyze data imbalance
    race_analysis, stage_analysis, combined_analysis = analyze_data_imbalance(df)
    
    # Step 4: Compute sample weights
    df_with_weights, weights_inverse, weights_balanced, weights_survival = compute_weights_for_subgroups(df)
    
    # Step 5: Split data
    train_size = int(0.7 * len(df_with_weights))
    val_size = int(0.15 * len(df_with_weights))
    
    df_train = df_with_weights.iloc[:train_size]
    df_val = df_with_weights.iloc[train_size:train_size + val_size]
    df_test = df_with_weights.iloc[train_size + val_size:]
    
    print(f"\nData split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    
    # Step 6: Train models
    model_standard, model_weighted, data_test = train_models(df_train, df_val, df_test)
    
    # Step 7: Evaluate models
    concordance_standard, concordance_weighted = evaluate_models(model_standard, model_weighted, data_test)
    
    # Step 8: Plot results
    plot_results(df_with_weights, concordance_standard, concordance_weighted)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("The weighted DeepHit model should show improved performance on underrepresented subgroups.")


if __name__ == "__main__":
    main() 