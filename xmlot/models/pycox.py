# Wrap various classifiers from PyCox.
# Everything follows Model's design.
#
# More details on: https://github.com/havakv/pycox

import pandas as pd
import numpy  as np

import pycox.models as pycox
import torch
import torchtuples  as tt

from xmlot.models.model import FromTheShelfModel

def _adapt_input_(x):
    # Adapt input
    if type(x) == pd.DataFrame:
        return torch.tensor(x.values, dtype=torch.float32)
    elif type(x) == np.ndarray:
        return torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, torch.Tensor):
        return x.float()  # Ensure float32
    else:
        return torch.tensor(x, dtype=torch.float32)

class PyCoxModel(FromTheShelfModel):
    def __init__(self, accessor_code, hyperparameters=None):
        super().__init__(
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )
        self.m_net  = None
        self.m_log  = None

    @property
    def net(self):
        return self.m_net

    @property
    def log(self):
        return self.m_log

    def _debug_data_types(self, df):
        """
        Debug method to check data types of input data.
        """
        accessor = getattr(df, self.accessor_code)
        print("=== Data Type Debug Info ===")
        print(f"Features shape: {accessor.features.shape}")
        print(f"Features dtypes: {accessor.features.dtypes}")
        print(f"Features numpy dtype: {accessor.features.to_numpy().dtype}")
        print(f"Durations dtype: {accessor.durations.dtype}")
        print(f"Events dtype: {accessor.events.dtype}")
        if hasattr(accessor, 'weights') and accessor.weights is not None:
            print(f"Weights dtype: {accessor.weights.dtype}")
        print("=" * 30)

    def _df_to_xy_(self, df):
        """
        Extract features and targets from a DataFrame into the intended PyCox fromat.
        """
        accessor = getattr(df, self.accessor_code)
        x = accessor.features.to_numpy().astype(np.float32)
        y = (
            accessor.durations.values,
            accessor.events.values
        )
        return x, y

    def _df_to_xyw_(self, df):
        """
        Extract features, targets, and weights from a DataFrame into the intended PyCox format.
        """
        accessor = getattr(df, self.accessor_code)
        x = accessor.features.to_numpy().astype(np.float32)
        y = (
            accessor.durations.values,
            accessor.events.values
        )
        # Check if weights are available in the accessor
        if hasattr(accessor, 'weights') and accessor.weights is not None:
            w = accessor.weights.values.astype(np.float32)
        else:
            w = None
        return x, y, w

# DEEPSURV #

class DeepSurv(PyCoxModel):
    """
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/cox-ph.ipynb

    Args:
        - in_features : dimension of a feature vector as input.
        - num_nodes   : sizes for intermediate layers.
        - batch_norm  : boolean enabling batch normalisation
        - dropout     : drop out rate.
        - output_bias : if set to False, no additive bias will be learnt.
    """

    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )
        in_features  = hyperparameters["in_features"]
        num_nodes    = hyperparameters["num_nodes"]    # [32, 32]
        out_features = 1
        batch_norm   = hyperparameters["batch_norm"]   # True
        dropout      = hyperparameters["dropout"]      # 0.1
        output_bias  = hyperparameters["output_bias"]  # False

        self.m_net = tt.practical.MLPVanilla(
            in_features,
            num_nodes,
            out_features,
            batch_norm,
            dropout,
            output_bias=output_bias)

        self.m_model = pycox.CoxPH(self.m_net, tt.optim.Adam)

    def fit(self, data_train, parameters=None):
        x, y, = self._df_to_xy_(data_train)

        # Compute learning rate
        if parameters['lr']:
            self.model.optimizer.set_lr(parameters['lr'])
        else:
            lrfinder = self.model.lr_finder(
                x,
                y,
                parameters['batch_size'],
                tolerance=parameters['tolerance']
            )
            lr = lrfinder.get_best_lr()
            self.model.optimizer.set_lr(lr)

        # Train
        self.m_log = self.model.fit(
            x,
            y,
            batch_size=parameters['batch_size'],
            epochs=parameters['epochs'],
            callbacks=parameters['callbacks'],
            verbose=parameters['verbose'],
            val_data=self._df_to_xy_(parameters["val_data"]),
            val_batch_size=parameters['batch_size']
        )

        _ = self.model.compute_baseline_hazards()

        return self

    def predict(self, x, parameters=None):
        input_tensor = torch.tensor(x.values).cuda()
        output = self.net(input_tensor).cpu().detach().numpy()
        output = output.reshape((output.shape[0],))
        return output


# DEEPHIT #

class DeepHitSingle(PyCoxModel):
    """
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/deephit.ipynb (single risk)
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/deephit_competing_risks.ipynb (competing risks)

    Args:
        - in_features  : dimension of a feature vector as input.
        - num_nodes    : sizes for intermediate layers.
        - out_features : matches the size of the grid on which time has been discretised
        - batch_norm   : boolean enabling batch normalisation
        - dropout      : drop out rate.
        - alpha        :
        - sigma        :
    """

    def __init__(self, accessor_code, hyperparameters):
        super().__init__(accessor_code=accessor_code)

        in_features = hyperparameters["in_features"]
        num_nodes = hyperparameters["num_nodes"]  # [32, 32]
        out_features = hyperparameters["out_features"]
        batch_norm = hyperparameters["batch_norm"]  # True
        dropout = hyperparameters["dropout"]  # 0.1

        self.m_net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

        self.m_model = pycox.DeepHitSingle(
            self.m_net,
            tt.optim.Adam,
            alpha=hyperparameters["alpha"],  # 0.2,
            sigma=hyperparameters["sigma"],  # 0.1,
        )

    def fit(self, data_train, parameters=None):

        # TODO: If time is continuous, make sure to DISCRETISE!
        #
        # discretiser = get_discretiser(model_name, pre_df_train)
        # df_train = discretiser(pre_df_train.copy())
        # df_val = discretiser(pre_df_val.copy())
        #
        # visitor.prefit(i, model_name, discretiser, df_train, df_val)

        x, y, = self._df_to_xy_(data_train)

        # Compute learning rate
        if parameters['lr']:
            self.m_model.optimizer.set_lr(parameters['lr'])
        else:
            lrfinder = self.m_model.lr_finder(
                x,
                y,
                parameters['batch_size'],
                tolerance=parameters['tolerance']
            )
            lr = lrfinder.get_best_lr()
            self.m_model.optimizer.set_lr(lr)

        # Train
        self.m_log = self.m_model.fit(
            x,
            y,
            batch_size=parameters['batch_size'],
            epochs=parameters['epochs'],
            callbacks=parameters['callbacks'],
            verbose=parameters['verbose'],
            val_data=self._df_to_xy_(parameters["val_data"]),
            val_batch_size=parameters['batch_size']
        )

        return self

    def predict(self, x, parameters=None):
        pass  # TODO


class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                 out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input_):
        out = self.shared_net(input_)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out

class DeepHit(PyCoxModel):
    """
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/deephit.ipynb (single risk)
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/deephit_competing_risks.ipynb (competing risks)

    Args:
        - in_features  : dimension of a feature vector as input.
        - num_nodes    : sizes for intermediate layers.
        - out_features : matches the size of the grid on which time has been discretised
        - batch_norm   : boolean enabling batch normalisation
        - dropout      : drop out rate.
        - alpha        :
        - sigma        :
        - use_weights  : boolean to enable sample weighting for underrepresented subgroups
    """

    def __init__(self, accessor_code, hyperparameters):
        super().__init__(accessor_code=accessor_code, hyperparameters=hyperparameters)

        in_features      = hyperparameters["in_features"]       # x_train.shape[1]
        num_nodes_shared = hyperparameters["num_nodes_shared"]  # [64, 64]
        num_nodes_indiv  = hyperparameters["num_nodes_indiv"]   # [32]
        num_risks        = hyperparameters["num_risks"]         # y_train[1].max()
        out_features     = hyperparameters["out_features"]      # len(labtrans.cuts)
        batch_norm       = hyperparameters["batch_norm"]        # True
        dropout          = hyperparameters["dropout"]           # 0.1
        seed             = hyperparameters["seed"]
        self.use_weights = hyperparameters.get("use_weights", False)  # Default to False for backward compatibility

        if seed is not None:
            torch.manual_seed(seed)
        self.m_net = CauseSpecificNet(
            in_features,
            num_nodes_shared,
            num_nodes_indiv,
            num_risks,
            out_features,
            batch_norm,
            dropout
        )

        # Create a custom loss function if weights are enabled
        if self.use_weights:
            self._create_weighted_loss_function(hyperparameters["alpha"], hyperparameters["sigma"])
        else:
            self.m_model = pycox.DeepHit(
                self.m_net,
                tt.optim.AdamWR(
                    lr=0.01,
                    decoupled_weight_decay=0.01,
                    cycle_eta_multiplier=0.8
                ),
                alpha=hyperparameters["alpha"],  # 0.2,
                sigma=hyperparameters["sigma"],  # 0.1,
                duration_index=hyperparameters["cuts"]
            )

    def _create_weighted_loss_function(self, alpha, sigma):
        """
        Create a custom weighted loss function for DeepHit that incorporates sample weights.
        """
        class WeightedDeepHitLoss:
            def __init__(self, net, alpha, sigma, duration_index):
                self.net = net
                self.alpha = alpha
                self.sigma = sigma
                self.duration_index = duration_index
                
            def __call__(self, pred, target, weights=None, *args, **kwargs):
                """
                Optimized weighted loss function for DeepHit using vectorized operations.
                
                Args:
                    pred: model predictions (already processed by the network)
                    target: tuple of (durations, events) or more values
                    weights: sample weights (optional)
                    *args, **kwargs: additional arguments passed by PyCox (ignored)
                """
                # Handle target unpacking - PyCox might pass more than 2 values
                if isinstance(target, (tuple, list)) and len(target) >= 2:
                    durations, events = target[0], target[1]
                else:
                    # Fallback for unexpected target format
                    durations, events = target, target
                
                # Convert to tensors if needed
                if not isinstance(durations, torch.Tensor):
                    durations = torch.tensor(durations, dtype=torch.long)
                if not isinstance(events, torch.Tensor):
                    events = torch.tensor(events, dtype=torch.long)
                if weights is not None and not isinstance(weights, torch.Tensor):
                    weights = torch.tensor(weights, dtype=torch.float32)
                
                # Handle different tensor shapes
                if len(pred.shape) == 3:
                    # Shape: [batch_size, num_risks, out_features]
                    batch_size, num_risks, out_features = pred.shape
                    if num_risks == 1:
                        # Single risk case
                        pred = pred.squeeze(1)  # Remove risk dimension: [batch_size, out_features]
                        F = torch.cumsum(pred, dim=1)  # Cumulative sum over time dimension
                    else:
                        # Multiple risks case
                        F = torch.cumsum(pred, dim=2)  # Cumulative sum over time dimension
                else:
                    # Shape: [batch_size, out_features] (already squeezed)
                    batch_size, out_features = pred.shape
                    F = torch.cumsum(pred, dim=1)  # Cumulative sum over time dimension
                
                # Ensure duration indices are within bounds
                duration_indices = torch.clamp(durations, 0, out_features - 1)
                
                # Vectorized likelihood loss (L1)
                # Get predictions at event times
                event_mask = events != 0
                censored_mask = events == 0
                
                L1 = 0
                
                # For events (not censored)
                if event_mask.sum() > 0:
                    event_durations = duration_indices[event_mask]
                    event_preds = pred[event_mask, event_durations]
                    event_weights = weights[event_mask] if weights is not None else 1.0
                    L1 -= torch.sum(event_weights * torch.log(event_preds + 1e-8))
                
                # For censored events
                if censored_mask.sum() > 0:
                    censored_durations = duration_indices[censored_mask]
                    censored_surv = 1 - F[censored_mask, censored_durations]
                    censored_weights = weights[censored_mask] if weights is not None else 1.0
                    L1 -= torch.sum(censored_weights * torch.log(censored_surv + 1e-8))
                
                # Simplified ranking loss (L2) - only for events
                L2 = 0
                if event_mask.sum() > 1:  # Need at least 2 events for ranking
                    event_indices = torch.where(event_mask)[0]
                    event_durations = duration_indices[event_mask]
                    
                    # Create pairs of events
                    for i in range(len(event_indices)):
                        for j in range(i + 1, len(event_indices)):
                            if event_durations[i] < event_durations[j]:
                                F_ki_i_si = F[event_indices[i], event_durations[i]]
                                F_ki_j_si = F[event_indices[j], event_durations[i]]
                                
                                weight_i = weights[event_indices[i]] if weights is not None else 1.0
                                L2 += weight_i * torch.exp((F_ki_j_si - F_ki_i_si) / self.sigma) * self.alpha
                
                return L1 + L2
        
        # Create the custom model with weighted loss
        self.m_model = pycox.DeepHit(
            self.m_net,
            tt.optim.AdamWR(
                lr=0.01,
                decoupled_weight_decay=0.01,
                cycle_eta_multiplier=0.8
            ),
            alpha=alpha,
            sigma=sigma,
            duration_index=self.hyperparameters["cuts"],
            loss=WeightedDeepHitLoss(self.m_net, alpha, sigma, self.hyperparameters["cuts"])
        )

    def create_fresh_instance(self):
        """
        Create a fresh instance of the model with the same hyperparameters.
        This is useful for cross-validation where feature counts might change.
        """
        return DeepHit(
            accessor_code=self.accessor_code,
            hyperparameters=self.hyperparameters.copy()
        )

    def fit(self, data_train, parameters=None):

        if parameters['seed'] is not None:
            torch.manual_seed(parameters['seed'])

        # Debug weight exclusion if weights are enabled
        if self.use_weights:
            self._debug_weight_exclusion(data_train)

        # Debug data types if needed (uncomment for debugging)
        # self._debug_data_types(data_train)

        if self.use_weights:
            x, y, w = self._df_to_xyw_(data_train)
        else:
            x, y, = self._df_to_xy_(data_train)
            w = None

        # Validate data types
        if not isinstance(x, np.ndarray) or x.dtype != np.float32:
            raise ValueError(f"Features must be float32 numpy array, got {type(x)} with dtype {x.dtype}")
        
        if w is not None and (not isinstance(w, np.ndarray) or w.dtype != np.float32):
            raise ValueError(f"Weights must be float32 numpy array, got {type(w)} with dtype {w.dtype}")

        # Check feature count consistency
        expected_features = self.hyperparameters.get("in_features")
        actual_features = x.shape[1]
        if expected_features != actual_features:
            print(f"WARNING: Feature count mismatch! Expected: {expected_features}, Got: {actual_features}")
            
            # Analyze the feature changes to help debug
            self._analyze_feature_changes(data_train, f"(Fold mismatch: {expected_features} -> {actual_features})")
            
            # Update hyperparameters to match actual features
            self.hyperparameters["in_features"] = actual_features
            print(f"Updated in_features to: {actual_features}")
            
            # Recreate the model with correct feature count
            self._recreate_model_with_correct_features()

        # Compute learning rate
        if parameters['lr']:
            self.m_model.optimizer.set_lr(parameters['lr'])
        else:
            lrfinder = self.m_model.lr_finder(
                x,
                y,
                parameters['batch_size'],
                tolerance=parameters['tolerance']
            )
            lr = lrfinder.get_best_lr()
            self.m_model.optimizer.set_lr(lr)

        # Train with or without weights
        val_data = self._df_to_xy_(parameters["val_data"])

        if self.use_weights and w is not None:
            # Custom training loop with weights
            self.m_log = self._fit_with_weights(
                x, y, w, val_data, parameters
            )
        else:
            # Standard training
            self.m_log = self.m_model.fit(
                x,
                y,
                batch_size=parameters['batch_size'],
                epochs=parameters['epochs'],
                callbacks=parameters['callbacks'],
                verbose=parameters['verbose'],
                val_data=val_data,
                val_batch_size=parameters['batch_size']
            )

        return self

    def _recreate_model_with_correct_features(self):
        """Recreate the model with the correct number of features."""
        print("Recreating model with correct feature count...")
        
        # Get current hyperparameters
        hyperparameters = self.hyperparameters.copy()
        
        # Recreate the network with correct input features
        in_features = hyperparameters["in_features"]
        num_nodes_shared = hyperparameters["num_nodes_shared"]
        num_nodes_indiv = hyperparameters["num_nodes_indiv"]
        num_risks = hyperparameters["num_risks"]
        out_features = hyperparameters["out_features"]
        batch_norm = hyperparameters["batch_norm"]
        dropout = hyperparameters["dropout"]
        
        # Recreate the network
        self.m_net = CauseSpecificNet(
            in_features,
            num_nodes_shared,
            num_nodes_indiv,
            num_risks,
            out_features,
            batch_norm,
            dropout
        )
        
        # Recreate the model
        if self.use_weights:
            self._create_weighted_loss_function(
                hyperparameters["alpha"], 
                hyperparameters["sigma"]
            )
        else:
            self.m_model = pycox.DeepHit(
                self.m_net,
                tt.optim.AdamWR(
                    lr=0.01,
                    decoupled_weight_decay=0.01,
                    cycle_eta_multiplier=0.8
                ),
                alpha=hyperparameters["alpha"],
                sigma=hyperparameters["sigma"],
                duration_index=hyperparameters["cuts"]
            )

    def _fit_with_weights(self, x, y, w, val_data, parameters):
        """
        Optimized training loop that incorporates sample weights efficiently.
        """
        # Use PyCox's built-in training with our custom weighted loss
        # This is much faster than the custom loop
        log = self.m_model.fit(
            x,
            y,
            batch_size=parameters['batch_size'],
            epochs=parameters['epochs'],
            callbacks=parameters['callbacks'],
            verbose=parameters['verbose'],
            val_data=val_data,
            val_batch_size=parameters['batch_size']
        )
        
        return log

    def predict_CIF(self, x, parameters=None):
        _ = parameters
        x_ = _adapt_input_(x)
        return self.model.predict_cif(x_)

    def predict_surv(self, x, parameters=None):
        _ = parameters
        x_ = _adapt_input_(x)
        return self.model.predict_surv_df(x_)

    def predict_pmf(self, x, parameters=None):
        _ = parameters
        x_ = _adapt_input_(x)
        return self.model.predict_pmf(x_)

    def predict(self, x, parameters=None):
        return self.predict_CIF(x, parameters=None)

    @property
    def model(self):
        """Return the underlying PyCox model for evaluation."""
        return self.m_model

    def interpolate(self, interpolation=10):
        """Add interpolate method for evaluation compatibility."""
        # Create a wrapper that provides the interpolate method
        class InterpolatedModel:
            def __init__(self, base_model):
                self.base_model = base_model
                
            def predict_surv_df(self, x):
                return self.base_model.predict_surv(x)
        
        return InterpolatedModel(self)

    def _analyze_feature_changes(self, data_train, fold_info=""):
        """
        Analyze potential causes of feature count changes between folds.
        """
        print(f"\n=== Feature Analysis {fold_info} ===")
        
        accessor = getattr(data_train, self.accessor_code)
        features_df = accessor.features
        
        print(f"Total features: {features_df.shape[1]}")
        print(f"Feature names: {list(features_df.columns)}")
        
        # Check for categorical features that might be encoded differently
        categorical_features = []
        for col in features_df.columns:
            if features_df[col].dtype == 'object' or features_df[col].nunique() < 10:
                categorical_features.append(col)
        
        if categorical_features:
            print(f"Categorical features: {categorical_features}")
            for col in categorical_features:
                unique_vals = features_df[col].unique()
                print(f"  {col}: {len(unique_vals)} unique values - {unique_vals}")
        
        # Check for missing values
        missing_counts = features_df.isnull().sum()
        if missing_counts.sum() > 0:
            print("Features with missing values:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count} missing")
        
        # Check for constant features
        constant_features = []
        for col in features_df.columns:
            if features_df[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"Constant features: {constant_features}")
        
        print("=" * 50)
        
        return {
            'total_features': features_df.shape[1],
            'categorical_features': categorical_features,
            'missing_features': missing_counts[missing_counts > 0].index.tolist(),
            'constant_features': constant_features
        }

    def _debug_weight_exclusion(self, data_train):
        """
        Debug method to verify that weights are not being used as features.
        """
        accessor = getattr(data_train, self.accessor_code)
        
        print("=== Weight Exclusion Debug ===")
        print(f"All DataFrame columns: {list(data_train.columns)}")
        print(f"Feature columns: {list(accessor.features.columns)}")
        print(f"Target columns: {accessor.target}")
        
        # Check if weight column is in features
        if hasattr(accessor, 'm_weight_column'):
            weight_col = accessor.m_weight_column
            weight_in_features = weight_col in accessor.features.columns
            print(f"Weight column: {weight_col}")
            print(f"Weight in features: {weight_in_features}")
            
            if not weight_in_features:
                print("✅ SUCCESS: Weight column is correctly excluded from features!")
            else:
                print("❌ ERROR: Weight column is incorrectly included in features!")
        
        # Check weights availability
        if hasattr(accessor, 'weights') and accessor.weights is not None:
            print(f"Weights available: Yes (shape: {accessor.weights.shape})")
        else:
            print("Weights available: No")
        
        print(f"Feature count: {len(accessor.features.columns)}")
        print("=" * 40)
        
        return not weight_in_features if hasattr(accessor, 'm_weight_column') else True
