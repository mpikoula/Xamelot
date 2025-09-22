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

    def _df_to_xy_(self, df):
        """
        Extract features and targets from a DataFrame into the intended PyCox format.
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

    def _df_to_xyg_(self, df):
        """
        Extract features, targets, and group indicators from a DataFrame into the intended PyCox format.
        """
        accessor = getattr(df, self.accessor_code)
        
        # For fairness accessor, include group column as a feature for training
        if hasattr(accessor, 'm_group_column'):
            # Include group column as a feature - this is what the model should learn from
            x = accessor.features.to_numpy().astype(np.float32)
        else:
            # For regular accessors, use features as is
            x = accessor.features.to_numpy().astype(np.float32)
        
        y = (
            accessor.durations.values,
            accessor.events.values
        )
        # Check if group indicators are available in the accessor
        if hasattr(accessor, 'group_indicators') and accessor.group_indicators is not None:
            group_indicators = accessor.group_indicators.values.astype(np.long)
        else:
            group_indicators = None
        return x, y, group_indicators

    def _create_indexed_dataloader(self, x, y, batch_size):
        """
        Create a custom data loader that tracks batch indices for proper weight mapping.
        This enables accurate sample weighting by knowing which samples are in each batch.
        """
        import torch.utils.data as data
        
        class IndexedSurvivalDataset(data.Dataset):
            """Dataset that returns (features, targets, index) tuples."""
            def __init__(self, x, y):
                self.x = torch.tensor(x, dtype=torch.float32)
                self.y = (torch.tensor(y[0], dtype=torch.long), 
                         torch.tensor(y[1], dtype=torch.long))
                self.indices = torch.arange(len(x))
            
            def __len__(self):
                return len(self.x)
            
            def __getitem__(self, idx):
                return self.x[idx], self.y[0][idx], self.y[1][idx], self.indices[idx]
        
        class IndexedDataLoader:
            """Custom data loader that yields batches with indices."""
            def __init__(self, dataset, batch_size, shuffle=True):
                self.dataset = dataset
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.sampler = data.RandomSampler(dataset) if shuffle else data.SequentialSampler(dataset)
                self.batch_sampler = data.BatchSampler(self.sampler, batch_size, drop_last=False)
            
            def __iter__(self):
                for batch_indices in self.batch_sampler:
                    x_batch = torch.stack([self.dataset.x[i] for i in batch_indices])
                    y_batch = (torch.stack([self.dataset.y[0][i] for i in batch_indices]),
                              torch.stack([self.dataset.y[1][i] for i in batch_indices]))
                    indices_batch = torch.stack([self.dataset.indices[i] for i in batch_indices])
                    yield x_batch, y_batch, indices_batch
            
            def __len__(self):
                return len(self.batch_sampler)
        
        # Create the indexed dataset and loader
        dataset = IndexedSurvivalDataset(x, y)
        loader = IndexedDataLoader(dataset, batch_size, shuffle=True)
        
        print(f"🔍 Created indexed data loader with {len(dataset)} samples")
        print(f"🔍 Batch size: {batch_size}, Total batches: {len(loader)}")
        
        return loader

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
        - use_weights  : boolean to enable sample weighting for underrepresented subgroups
    """

    def __init__(self, accessor_code, hyperparameters):
        super().__init__(accessor_code=accessor_code, hyperparameters=hyperparameters)

        in_features = hyperparameters["in_features"]
        num_nodes = hyperparameters["num_nodes"]  # [32, 32]
        out_features = hyperparameters["out_features"]
        batch_norm = hyperparameters["batch_norm"]  # True
        dropout = hyperparameters["dropout"]  # 0.1
        seed = hyperparameters["seed"]
        self.use_weights = hyperparameters.get("use_weights", False)  # Default to False for backward compatibility
        self.use_fairness = hyperparameters.get("use_fairness", False)  # Default to False for backward compatibility

        if seed is not None:
            torch.manual_seed(seed)

        self.m_net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

        self.m_model = pycox.DeepHitSingle(
            self.m_net,
            tt.optim.Adam,
            alpha=hyperparameters["alpha"],  # 0.2,
            sigma=hyperparameters["sigma"],  # 0.1,
        )
        
        # Handle fairness mode (takes precedence over weighted mode)
        if self.use_fairness:
            fairness_weight = hyperparameters.get("fairness_weight", 0.1)
            print(f"🔍 DeepHitSingle: use_fairness=True, fairness_weight={fairness_weight}")
            self._create_fairness_loss_function(hyperparameters["alpha"], hyperparameters["sigma"], fairness_weight)
        # Handle weighted mode (only if fairness is not enabled)
        elif self.use_weights:
            ablation_mode = hyperparameters.get("ablation_mode", "full")
            print(f"🔍 DeepHitSingle: use_weights=True, ablation_mode={ablation_mode}")
            
            if ablation_mode != "unweighted":
                # Create custom loss function for weighted mode
                print("🔍 Creating custom loss function for weighted DeepHitSingle...")
                self._create_weighted_loss_function(hyperparameters["alpha"], hyperparameters["sigma"])
        else:
            print("🔍 DeepHitSingle: use_weights=False, use_fairness=False (standard PyCox training)")
            # No custom loss function - use PyCox's standard loss

    def _create_weighted_loss_function(self, alpha, sigma):
        """
        Create a weighted loss function that extends PyCox's standard DeepHit loss with sample weights.
        """
        class WeightedDeepHitLoss:
            def __init__(self, net, alpha, sigma, duration_index, ablation_mode="full", max_ranking_pairs=20):
                self.net = net
                self.alpha = alpha
                self.sigma = sigma
                self.duration_index = duration_index
                self.debug_counter = 0
                self.ablation_mode = ablation_mode
                self.max_ranking_pairs = max_ranking_pairs
                
                # Create PyCox's standard loss function
                import pycox.models.loss as pycox_loss
                self.standard_loss = pycox_loss.DeepHitLoss(alpha, sigma)
                
            def __call__(self, pred, durations, events, rank_mat=None, weights=None, batch_indices=None, *args, **kwargs):
                """
                Weighted loss function that extends PyCox's standard loss with sample weights.
                """
                self.debug_counter += 1
                
                # Get stored weights from the network
                stored_weights = None
                if hasattr(self.net, '_training_weights'):
                    stored_weights = self.net._training_weights
                
                # If no weights available or unweighted mode, use PyCox's standard loss
                if (weights is None and stored_weights is None) or self.ablation_mode == "unweighted":
                    return self._call_standard_loss_safely(pred, durations, events, rank_mat)
                
                # For weighted mode, implement weighted loss using stored weights
                if stored_weights is not None and self.ablation_mode != "unweighted":
                    # Convert pred to tensor if needed and get device
                    if not isinstance(pred, torch.Tensor):
                        pred = torch.tensor(pred, dtype=torch.float32)
                    
                    # Get batch weights using proper batch indices if available
                    batch_size = pred.shape[0]
                    
                    if batch_indices is not None:
                        # Use proper batch indices for accurate weight mapping
                        if isinstance(batch_indices, torch.Tensor):
                            batch_indices = batch_indices.cpu().numpy()
                        
                        # Ensure indices are within bounds
                        batch_indices = np.clip(batch_indices, 0, len(stored_weights) - 1)
                        batch_weights = torch.tensor(stored_weights[batch_indices], dtype=torch.float32, device=pred.device)
                    else:
                        # Fallback to simple approach (first batch_size samples)
                        if len(stored_weights) >= batch_size:
                            batch_weights = torch.tensor(stored_weights[:batch_size], dtype=torch.float32, device=pred.device)
                        else:
                            # Fallback: use uniform weights if not enough stored weights
                            batch_weights = torch.ones(batch_size, dtype=torch.float32, device=pred.device)
                    
                    # Check if all weights are 1.0 - if so, use standard loss directly
                    if torch.allclose(batch_weights, torch.ones_like(batch_weights), atol=1e-6):
                        return self._call_standard_loss_safely(pred, durations, events, rank_mat)
                    
                    # Implement weighted loss by extending PyCox's standard loss
                    return self._compute_weighted_loss(pred, durations, events, rank_mat, batch_weights)
                else:
                    # Use PyCox's standard loss for unweighted mode
                    return self._call_standard_loss_safely(pred, durations, events, rank_mat)
            
            def _call_standard_loss_safely(self, pred, durations, events, rank_mat):
                """Safely call PyCox's standard loss with proper tensor conversion."""
                # Convert all inputs to proper PyTorch tensors
                if not isinstance(pred, torch.Tensor):
                    pred = torch.tensor(pred, dtype=torch.float32)
                
                if not isinstance(durations, torch.Tensor):
                    durations = torch.tensor(durations, dtype=torch.long)
                elif durations.dtype != torch.long:
                    durations = durations.long()
                
                if not isinstance(events, torch.Tensor):
                    events = torch.tensor(events, dtype=torch.long)
                elif events.dtype != torch.long:
                    events = events.long()
                
                if rank_mat is not None and not isinstance(rank_mat, torch.Tensor):
                    rank_mat = torch.tensor(rank_mat, dtype=torch.float32)
                
                # Call PyCox's standard loss
                return self.standard_loss(pred, durations, events, rank_mat)

            def _compute_weighted_loss(self, pred, durations, events, rank_mat, batch_weights):
                """
                Compute weighted loss by extending PyCox's standard loss with sample weights.
                """
                # Convert all inputs to proper PyTorch tensors with correct types
                if not isinstance(pred, torch.Tensor):
                    pred = torch.tensor(pred, dtype=torch.float32)
                
                if not isinstance(durations, torch.Tensor):
                    durations = torch.tensor(durations, dtype=torch.long)
                elif durations.dtype != torch.long:
                    durations = durations.long()
                
                if not isinstance(events, torch.Tensor):
                    events = torch.tensor(events, dtype=torch.long)
                elif events.dtype != torch.long:
                    events = events.long()
                
                if rank_mat is not None and not isinstance(rank_mat, torch.Tensor):
                    rank_mat = torch.tensor(rank_mat, dtype=torch.float32)
                
                # Ensure batch_weights is on the same device as pred
                if batch_weights.device != pred.device:
                    batch_weights = batch_weights.to(pred.device)
                
                # Get standard PyCox loss
                standard_loss = self.standard_loss(pred, durations, events, rank_mat)
                
                # For non-uniform weights, implement proper weighting
                # Note: This is a simplified approach - you may want to implement more sophisticated weighting
                weight_factor = batch_weights.mean()
                weighted_loss = standard_loss * weight_factor
                
                return weighted_loss

        # Create the custom loss function instance
        ablation_mode = self.m_hyperparameters.get("ablation_mode", "full")
        max_ranking_pairs = self.m_hyperparameters.get("max_ranking_pairs", 20)
        
        custom_loss = WeightedDeepHitLoss(
            self.m_net,
            alpha,
            sigma,
            self.m_hyperparameters.get("cuts"),
            ablation_mode,
            max_ranking_pairs
        )
        
        # Replace the model's loss function with our custom one
        self.m_model.loss = custom_loss
        
        print(f"🔍 Replaced loss function in PyCox DeepHitSingle model")

    def _create_fairness_loss_function(self, alpha, sigma, fairness_weight):
        """
        Create a fairness-aware loss function that penalizes differences in mean risk between subgroups.
        """
        from xmlot.models.fairness_loss import DeepHitSingleFairLoss
        
        class FairnessLossWrapper:
            def __init__(self, net, alpha, sigma, fairness_weight, duration_index):
                self.net = net
                self.alpha = alpha
                self.sigma = sigma
                self.fairness_weight = fairness_weight
                self.duration_index = duration_index
                self.debug_counter = 0
                
                # Create the fairness loss function
                self.fairness_loss_fn = DeepHitSingleFairLoss(
                    alpha=alpha, 
                    sigma=sigma, 
                    fairness_weight=fairness_weight
                )
                
            def __call__(self, pred, durations, events, rank_mat=None, weights=None, batch_indices=None, *args, **kwargs):
                """
                Fairness-aware loss function that penalizes differences in mean risk between subgroups.
                """
                self.debug_counter += 1
                
                # Get stored group indicators from the network
                stored_group_indicators = None
                if hasattr(self.net, '_training_group_indicators'):
                    stored_group_indicators = self.net._training_group_indicators
                
                # If no group indicators available, use standard loss
                if stored_group_indicators is None:
                    import pycox.models.loss as pycox_loss
                    standard_loss = pycox_loss.DeepHitLoss(self.alpha, self.sigma)
                    return standard_loss(pred, durations, events, rank_mat)
                
                # Convert pred to tensor if needed and get device
                if not isinstance(pred, torch.Tensor):
                    pred = torch.tensor(pred, dtype=torch.float32)
                
                # Convert durations and events to proper tensor types
                if not isinstance(durations, torch.Tensor):
                    durations = torch.tensor(durations, dtype=torch.long)
                elif durations.dtype != torch.long:
                    durations = durations.long()
                
                if not isinstance(events, torch.Tensor):
                    events = torch.tensor(events, dtype=torch.long)
                elif events.dtype != torch.long:
                    events = events.long()
                
                # Get batch group indicators using proper batch indices if available
                batch_size = pred.shape[0]
                
                if batch_indices is not None:
                    # Use proper batch indices for accurate group mapping
                    if isinstance(batch_indices, torch.Tensor):
                        batch_indices = batch_indices.cpu().numpy()
                    
                    # Ensure indices are within bounds
                    batch_indices = np.clip(batch_indices, 0, len(stored_group_indicators) - 1)
                    batch_group_indicators = torch.tensor(stored_group_indicators[batch_indices], dtype=torch.long, device=pred.device)
                else:
                    # Fallback to simple approach (first batch_size samples)
                    if len(stored_group_indicators) >= batch_size:
                        batch_group_indicators = torch.tensor(stored_group_indicators[:batch_size], dtype=torch.long, device=pred.device)
                    else:
                        # Fallback: use uniform group indicators if not enough stored
                        batch_group_indicators = torch.zeros(batch_size, dtype=torch.long, device=pred.device)
                        print("🔍 Fallback to uniform group indicators")
                
                # Use the fairness loss function
                return self.fairness_loss_fn(pred, durations, events, rank_mat, batch_group_indicators)
            
            @property
            def loss_history(self):
                """Access loss history from the fairness loss function."""
                return self.fairness_loss_fn.loss_history

        # Create the fairness loss function instance
        fairness_loss = FairnessLossWrapper(
            self.m_net,
            alpha,
            sigma,
            fairness_weight,
            self.m_hyperparameters.get("cuts")
        )
        
        # Replace the model's loss function with our fairness-aware one
        self.m_model.loss = fairness_loss
        
        print(f"🔍 Replaced loss function with fairness-aware loss in PyCox DeepHitSingle model")

    def fit(self, data_train, parameters=None):
        if parameters['seed'] is not None:
            torch.manual_seed(parameters['seed'])

        if self.use_fairness:
            # Extract group indicators for fairness training
            x, y, group_indicators = self._df_to_xyg_(data_train)
            print(f"✅ Using fairness training with {len(group_indicators)} samples")
            print(f"   Group distribution: {np.bincount(group_indicators)}")
            
            # Store group indicators in the network for the loss function to access
            self.m_net._training_group_indicators = group_indicators
            
            # Create indexed data loader for proper batch tracking
            indexed_loader = self._create_indexed_dataloader(x, y, parameters['batch_size'])
            print(f"🔍 Using indexed data loader for accurate group mapping")
        elif self.use_weights:
            x, y, w = self._df_to_xyw_(data_train)
            print(f"✅ Using weighted training with {len(w)} samples")
            print(f"   Weight range: {w.min():.3f} to {w.max():.3f}, mean: {w.mean():.3f}")
            
            # Store weights in the network for the loss function to access
            self.m_net._training_weights = w
            
            # Create indexed data loader for proper batch tracking (only if not unweighted mode)
            ablation_mode = self.m_hyperparameters.get("ablation_mode", "full")
            if ablation_mode != "unweighted":
                indexed_loader = self._create_indexed_dataloader(x, y, parameters['batch_size'])
                print(f"🔍 Using indexed data loader for accurate weight mapping")
            else:
                print(f"🔍 Ablation mode '{ablation_mode}': using standard training despite weights")
        else:
            # Use EXACT original fit method for unweighted mode
            x, y = self._df_to_xy_(data_train)
            print(f"ℹ️  Using original DeepHit training (no weights, no fairness)")

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
        if self.use_fairness or self.use_weights:
            # Use fairness or weighted training
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
        else:
            # Use EXACT original training for unweighted mode
            val_data = self._df_to_xy_(parameters["val_data"])
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

    def predict(self, x, parameters=None):
        """
        Return CIF predictions for consistency with DeepHit interface.
        For DeepHitSingle, this converts survival probabilities to CIF.
        """
        x_ = _adapt_input_(x)
        surv = self.m_model.predict_surv(x_)
        # Convert survival to CIF: CIF(t) = 1 - S(t)
        cif = 1 - surv
        return cif.T

    def predict_CIF(self, x, parameters=None):
        """
        Return CIF predictions for compatibility with DeepHit interface.
        For DeepHitSingle, this converts survival probabilities to CIF.
        """
        x_ = _adapt_input_(x)
        surv = self.m_model.predict_surv(x_)
        # Convert survival to CIF: CIF(t) = 1 - S(t)
        cif = 1 - surv
        return cif

    def predict_surv(self, x, parameters=None):
        """Predict survival probabilities for DeepHitSingle model."""
        x_ = _adapt_input_(x)
        return self.m_model.predict_surv_df(x_)

    def get_loss_history(self):
        """
        Get the loss history from training.
        
        Returns:
            Dictionary containing loss history if available, None otherwise
        """
        if hasattr(self, 'm_model') and hasattr(self.m_model, 'loss'):
            # Check if it's a fairness wrapper
            if hasattr(self.m_model.loss, 'loss_history'):
                return self.m_model.loss.loss_history
            # Check if it's a fairness wrapper that delegates to fairness_loss_fn
            elif hasattr(self.m_model.loss, 'fairness_loss_fn'):
                if hasattr(self.m_model.loss.fairness_loss_fn, 'loss_history'):
                    return self.m_model.loss.fairness_loss_fn.loss_history
        return None

    def plot_loss_components(self, save_path=None):
        """
        Plot the loss components during training.
        
        Args:
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        loss_history = self.get_loss_history()
        if loss_history is None:
            print("❌ No loss history available. Make sure to train with fairness or weighted mode enabled.")
            return
        
        # Check if loss history is empty
        if not loss_history['total_loss']:
            print("❌ Loss history is empty. This might mean:")
            print("   1. The model wasn't trained with fairness or weighted mode")
            print("   2. The loss function wasn't called during training")
            print("   3. The training didn't complete successfully")
            return
        
        # Determine if this is fairness or weighted mode based on available keys
        is_fairness_mode = 'fairness_loss' in loss_history
        is_weighted_mode = 'weighted_loss' in loss_history
        
        if is_fairness_mode:
            self._plot_fairness_components(loss_history, save_path)
        elif is_weighted_mode:
            self._plot_weighted_components(loss_history, save_path)
        else:
            print("❌ Unknown loss history format")
    
    def _plot_fairness_components(self, loss_history, save_path=None):
        """Plot fairness loss components."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Fairness Loss Components During Training - {type(self).__name__}', fontsize=16)
        
        # Plot 1: Total Loss
        axes[0, 0].plot(loss_history['total_loss'], label='Total Loss (Standard + Fairness)', color='red')
        axes[0, 0].set_title('Total Loss (Standard + Fairness)')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Standard Loss Components
        axes[0, 1].plot(loss_history['nll_loss'], label='NLL Loss (Est.)', color='blue', alpha=0.7)
        axes[0, 1].plot(loss_history['ranking_loss'], label='Ranking Loss (Est.)', color='green', alpha=0.7)
        axes[0, 1].set_title('Standard Loss Components (Estimated)')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Fairness Loss
        axes[0, 2].plot(loss_history['fairness_loss'], label='Fairness Penalty', color='purple')
        axes[0, 2].set_title('Fairness Penalty')
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('Fairness Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Group Mean Risks
        axes[1, 0].plot(loss_history['group1_mean_risk'], label='Group 1 Mean Risk', color='orange')
        axes[1, 0].plot(loss_history['group2_mean_risk'], label='Group 2 Mean Risk', color='brown')
        axes[1, 0].set_title('Mean Risk by Group')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Mean Risk')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Risk Difference
        risk_diff = np.array(loss_history['group1_mean_risk']) - np.array(loss_history['group2_mean_risk'])
        axes[1, 1].plot(risk_diff, label='Risk Difference (G1 - G2)', color='magenta')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Risk Difference Between Groups')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Risk Difference')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Loss Ratio
        if len(loss_history['fairness_loss']) > 0 and len(loss_history['nll_loss']) > 0:
            fairness_ratio = np.array(loss_history['fairness_loss']) / (np.array(loss_history['nll_loss']) + 1e-8)
            axes[1, 2].plot(fairness_ratio, label='Fairness/Standard Loss Ratio', color='cyan')
            axes[1, 2].set_title('Fairness to Standard Loss Ratio')
            axes[1, 2].set_xlabel('Training Step')
            axes[1, 2].set_ylabel('Ratio')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Fairness loss plot saved to: {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\n📈 Fairness Loss Summary:")
        try:
            total_loss_array = np.array(loss_history['total_loss'])
            finite_total = total_loss_array[np.isfinite(total_loss_array)]
            if len(finite_total) > 0:
                print(f"   Total Loss - Final: {finite_total[-1]:.4f}, Mean: {np.mean(finite_total):.4f}")
        except Exception as e:
            print(f"   Total Loss - Error: {e}")
        
        try:
            fairness_array = np.array(loss_history['fairness_loss'])
            finite_fairness = fairness_array[np.isfinite(fairness_array)]
            if len(finite_fairness) > 0:
                print(f"   Fairness Loss - Final: {finite_fairness[-1]:.4f}, Mean: {np.mean(finite_fairness):.4f}")
        except Exception as e:
            print(f"   Fairness Loss - Error: {e}")
        
        try:
            group1_array = np.array(loss_history['group1_mean_risk'])
            group2_array = np.array(loss_history['group2_mean_risk'])
            finite_group1 = group1_array[np.isfinite(group1_array)]
            finite_group2 = group2_array[np.isfinite(group2_array)]
            if len(finite_group1) > 0 and len(finite_group2) > 0:
                print(f"   Group 1 Mean Risk - Final: {finite_group1[-1]:.4f}, Mean: {np.mean(finite_group1):.4f}")
                print(f"   Group 2 Mean Risk - Final: {finite_group2[-1]:.4f}, Mean: {np.mean(finite_group2):.4f}")
                final_diff = finite_group1[-1] - finite_group2[-1]
                print(f"   Final Risk Difference: {final_diff:.4f}")
        except Exception as e:
            print(f"   Group Risks - Error: {e}")
        
        print(f"   Training Steps: {len(loss_history['total_loss'])}")
    
    def _plot_weighted_components(self, loss_history, save_path=None):
        """Plot weighted loss components (existing implementation)."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Weighted Loss Components During Training - {type(self).__name__}', fontsize=16)
        
        # Plot 1: Weighted Loss
        axes[0, 0].plot(loss_history['weighted_loss'], label='Weighted Loss', color='red')
        axes[0, 0].set_title('Weighted Loss (Pairwise Weighted)')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Total Loss (Unweighted)
        axes[0, 1].plot(loss_history['total_loss'], label='Total Loss (Unweighted)', color='blue')
        axes[0, 1].set_title('Total Loss (Unweighted)')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Loss Components (Estimated)
        axes[1, 0].plot(loss_history['likelihood_loss'], label='Likelihood Loss (Est.)', color='green', alpha=0.7)
        axes[1, 0].plot(loss_history['ranking_loss'], label='Ranking Loss (Est.)', color='orange', alpha=0.7)
        axes[1, 0].set_title('Loss Components (Estimated)')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Batch Weights
        axes[1, 1].plot(loss_history['batch_weights_mean'], label='Mean Batch Weight', color='purple')
        axes[1, 1].set_title('Mean Batch Weight')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Weighted loss plot saved to: {save_path}")
        
        plt.show()

    def get_current_loss_components(self, batch_data=None):
        """
        Get current loss components from the fairness loss function.
        
        Args:
            batch_data: Optional batch data to compute loss components for.
                      If None, uses the last batch from training.
        
        Returns:
            dict: Dictionary containing loss components
        """
        if not hasattr(self, 'm_model') or not hasattr(self.m_model, 'loss'):
            return None
        
        loss_fn = self.m_model.loss
        
        # If it's our fairness loss wrapper, get the actual loss function
        if hasattr(loss_fn, 'fairness_loss_fn'):
            loss_fn = loss_fn.fairness_loss_fn
        
        # Check if it's our fairness loss and has loss history
        if hasattr(loss_fn, 'loss_history') and loss_fn.loss_history:
            # Get the most recent components from loss history
            latest = {}
            
            # Extract all available components from loss history
            for key in ['total_loss', 'nll_loss', 'ranking_loss', 'fairness_loss', 
                       'group1_mean_risk', 'group2_mean_risk']:
                if key in loss_fn.loss_history and loss_fn.loss_history[key]:
                    latest[key] = loss_fn.loss_history[key][-1]
                else:
                    latest[key] = 0.0
            
            # Add fairness weight and other parameters
            if hasattr(loss_fn, 'fairness_weight'):
                latest['fairness_weight'] = loss_fn.fairness_weight
            if hasattr(loss_fn, 'alpha'):
                latest['alpha'] = loss_fn.alpha
            if hasattr(loss_fn, 'sigma'):
                latest['sigma'] = loss_fn.sigma
            
            # Calculate additional metrics
            if 'group1_mean_risk' in latest and 'group2_mean_risk' in latest:
                latest['risk_difference'] = abs(latest['group1_mean_risk'] - latest['group2_mean_risk'])
            
            if 'fairness_loss' in latest and 'total_loss' in latest:
                fairness_weight = latest.get('fairness_weight', 0.0)
                latest['fairness_contribution'] = fairness_weight * latest['fairness_loss']
                if latest['total_loss'] > 0:
                    latest['fairness_ratio'] = latest['fairness_contribution'] / latest['total_loss']
                else:
                    latest['fairness_ratio'] = 0.0
            
            # Only return if we have meaningful data
            if any(v != 0.0 for k, v in latest.items() if k not in ['fairness_weight', 'alpha', 'sigma']):
                return latest
        
        return None


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
        - num_nodes_shared : sizes for shared layers.
        - num_nodes_indiv  : sizes for individual risk layers.
        - num_risks    : number of competing risks.
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
        self.use_fairness = hyperparameters.get("use_fairness", False)  # Default to False for backward compatibility

        if seed is not None:
            torch.manual_seed(seed)
        
        # Use the original CauseSpecificNet architecture for competing risks
        self.m_net = CauseSpecificNet(
            in_features,
            num_nodes_shared,
            num_nodes_indiv,
            num_risks,
            out_features,
            batch_norm,
            dropout
        )

        # Use the original competing risks DeepHit model with EXACT original parameters
        if self.use_fairness:
            # Use configurable parameters for fairness mode
            fairness_weight = hyperparameters.get("fairness_weight", 0.1)
            print("🔍 Using fairness-aware DeepHit with configurable parameters")
            self.m_model = pycox.DeepHit(
                self.m_net,
                tt.optim.AdamWR(
                    lr=0.01,
                    decoupled_weight_decay=0.01,
                    cycle_eta_multiplier=0.8
                ),
                alpha=hyperparameters["alpha"],
                sigma=hyperparameters["sigma"],
                duration_index=hyperparameters.get("cuts")
            )
            
            # Create custom loss function for fairness mode
            print("🔍 Creating custom loss function for fairness-aware competing risks DeepHit...")
            self._create_fairness_loss_function(hyperparameters["alpha"], hyperparameters["sigma"], fairness_weight)
        elif self.use_weights:
            # Use configurable parameters for weighted mode
            print("🔍 Using weighted DeepHit with configurable parameters")
            self.m_model = pycox.DeepHit(
                self.m_net,
                tt.optim.AdamWR(
                    lr=0.01,
                    decoupled_weight_decay=0.01,
                    cycle_eta_multiplier=0.8
                ),
                alpha=hyperparameters["alpha"],
                sigma=hyperparameters["sigma"],
                duration_index=hyperparameters.get("cuts")
            )
            
            # Create custom loss function for weighted mode
            ablation_mode = hyperparameters.get("ablation_mode", "full")
            if ablation_mode != "unweighted":
                print("🔍 Creating custom loss function for weighted competing risks DeepHit...")
                self._create_weighted_loss_function(hyperparameters["alpha"], hyperparameters["sigma"])
        else:
            # Use EXACT original parameters for unweighted mode
            print("🔍 Using original DeepHit parameters (unweighted mode)")
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

    def _create_weighted_loss_function(self, alpha, sigma):
        """
        Create a weighted loss function that extends PyCox's standard DeepHit loss with sample weights.
        """
        class WeightedDeepHitLoss:
            def __init__(self, net, alpha, sigma, duration_index, ablation_mode="full", max_ranking_pairs=20):
                self.net = net
                self.alpha = alpha
                self.sigma = sigma
                self.duration_index = duration_index
                self.debug_counter = 0
                self.ablation_mode = ablation_mode
                self.max_ranking_pairs = max_ranking_pairs
                
                # Create PyCox's standard loss function
                import pycox.models.loss as pycox_loss
                self.standard_loss = pycox_loss.DeepHitLoss(alpha, sigma)
                
            def __call__(self, pred, durations, events, rank_mat=None, weights=None, batch_indices=None, *args, **kwargs):
                """
                Weighted loss function that extends PyCox's standard loss with sample weights.
                """
                self.debug_counter += 1
                
                # Get stored weights from the network
                stored_weights = None
                if hasattr(self.net, '_training_weights'):
                    stored_weights = self.net._training_weights
                
                # If no weights available or unweighted mode, use PyCox's standard loss
                if (weights is None and stored_weights is None) or self.ablation_mode == "unweighted":
                    return self._call_standard_loss_safely(pred, durations, events, rank_mat)
                
                # For weighted mode, implement weighted loss using stored weights
                if stored_weights is not None and self.ablation_mode != "unweighted":
                    # Convert pred to tensor if needed and get device
                    if not isinstance(pred, torch.Tensor):
                        pred = torch.tensor(pred, dtype=torch.float32)
                    
                    # Get batch weights using proper batch indices if available
                    batch_size = pred.shape[0]
                    
                    if batch_indices is not None:
                        # Use proper batch indices for accurate weight mapping
                        if isinstance(batch_indices, torch.Tensor):
                            batch_indices = batch_indices.cpu().numpy()
                        
                        # Ensure indices are within bounds
                        batch_indices = np.clip(batch_indices, 0, len(stored_weights) - 1)
                        batch_weights = torch.tensor(stored_weights[batch_indices], dtype=torch.float32, device=pred.device)
                    else:
                        # Fallback to simple approach (first batch_size samples)
                        if len(stored_weights) >= batch_size:
                            batch_weights = torch.tensor(stored_weights[:batch_size], dtype=torch.float32, device=pred.device)
                        else:
                            # Fallback: use uniform weights if not enough stored weights
                            batch_weights = torch.ones(batch_size, dtype=torch.float32, device=pred.device)
                            print("🔍 Fallback to uniform weights")
                    
                    
                    # Implement weighted loss by extending PyCox's standard loss
                    return self._compute_weighted_loss(pred, durations, events, rank_mat, batch_weights)
                else:
                    # Use PyCox's standard loss for unweighted mode
                    return self._call_standard_loss_safely(pred, durations, events, rank_mat)
            
            def _call_standard_loss_safely(self, pred, durations, events, rank_mat):
                """Safely call PyCox's standard loss with proper tensor conversion."""
                # Convert all inputs to proper PyTorch tensors
                if not isinstance(pred, torch.Tensor):
                    pred = torch.tensor(pred, dtype=torch.float32)
                
                if not isinstance(durations, torch.Tensor):
                    durations = torch.tensor(durations, dtype=torch.long)
                elif durations.dtype != torch.long:
                    durations = durations.long()
                
                if not isinstance(events, torch.Tensor):
                    events = torch.tensor(events, dtype=torch.long)
                elif events.dtype != torch.long:
                    events = events.long()
                
                if rank_mat is not None and not isinstance(rank_mat, torch.Tensor):
                    rank_mat = torch.tensor(rank_mat, dtype=torch.float32)
                
                # Call PyCox's standard loss
                return self.standard_loss(pred, durations, events, rank_mat)

            def _compute_weighted_loss(self, pred, durations, events, rank_mat, batch_weights):
                """
                Compute weighted loss by extending PyCox's standard loss with sample weights.
                """
                # Convert all inputs to proper PyTorch tensors with correct types
                if not isinstance(pred, torch.Tensor):
                    pred = torch.tensor(pred, dtype=torch.float32)
                
                if not isinstance(durations, torch.Tensor):
                    durations = torch.tensor(durations, dtype=torch.long)
                elif durations.dtype != torch.long:
                    durations = durations.long()
                
                if not isinstance(events, torch.Tensor):
                    events = torch.tensor(events, dtype=torch.long)
                elif events.dtype != torch.long:
                    events = events.long()
                
                if rank_mat is not None and not isinstance(rank_mat, torch.Tensor):
                    rank_mat = torch.tensor(rank_mat, dtype=torch.float32)
                
                # Ensure batch_weights is on the same device as pred
                if batch_weights.device != pred.device:
                    batch_weights = batch_weights.to(pred.device)
                
                # Get standard PyCox loss
                standard_loss = self.standard_loss(pred, durations, events, rank_mat)
                
                
                # For non-uniform weights, implement proper weighting
                # Note: This is a simplified approach - you may want to implement more sophisticated weighting
                weight_factor = batch_weights.mean()
                weighted_loss = standard_loss * weight_factor
                
                return weighted_loss

        # Create the custom loss function instance
        ablation_mode = self.m_hyperparameters.get("ablation_mode", "full")
        max_ranking_pairs = self.m_hyperparameters.get("max_ranking_pairs", 20)
        
        custom_loss = WeightedDeepHitLoss(
            self.m_net,
            alpha,
            sigma,
            self.m_hyperparameters.get("cuts"),
            ablation_mode,
            max_ranking_pairs
        )
        
        # Replace the model's loss function with our custom one
        self.m_model.loss = custom_loss
        
        print(f"🔍 Replaced loss function in PyCox DeepHit (competing risks) model")

    def _create_fairness_loss_function(self, alpha, sigma, fairness_weight):
        """
        Create a fairness-aware loss function that penalizes differences in mean risk between subgroups.
        """
        from xmlot.models.fairness_loss import DeepHitFairLoss
        
        class FairnessLossWrapper:
            def __init__(self, net, alpha, sigma, fairness_weight, duration_index):
                self.net = net
                self.alpha = alpha
                self.sigma = sigma
                self.fairness_weight = fairness_weight
                self.duration_index = duration_index
                self.debug_counter = 0
                
                # Create the fairness loss function for competing risks
                self.fairness_loss_fn = DeepHitFairLoss(
                    alpha=alpha, 
                    sigma=sigma, 
                    fairness_weight=fairness_weight
                )
                
            def __call__(self, pred, durations, events, rank_mat=None, weights=None, batch_indices=None, *args, **kwargs):
                """
                Fairness-aware loss function that penalizes differences in mean risk between subgroups.
                """
                self.debug_counter += 1
                
                # Get stored group indicators from the network
                stored_group_indicators = None
                if hasattr(self.net, '_training_group_indicators'):
                    stored_group_indicators = self.net._training_group_indicators
                
                # If no group indicators available, use standard loss
                if stored_group_indicators is None:
                    import pycox.models.loss as pycox_loss
                    standard_loss = pycox_loss.DeepHitLoss(self.alpha, self.sigma)
                    return standard_loss(pred, durations, events, rank_mat)
                
                # Convert pred to tensor if needed and get device
                if not isinstance(pred, torch.Tensor):
                    pred = torch.tensor(pred, dtype=torch.float32)
                
                # Convert durations and events to proper tensor types
                if not isinstance(durations, torch.Tensor):
                    durations = torch.tensor(durations, dtype=torch.long)
                elif durations.dtype != torch.long:
                    durations = durations.long()
                
                if not isinstance(events, torch.Tensor):
                    events = torch.tensor(events, dtype=torch.long)
                elif events.dtype != torch.long:
                    events = events.long()
                
                # Get batch group indicators using proper batch indices if available
                batch_size = pred.shape[0]
                
                if batch_indices is not None:
                    # Use proper batch indices for accurate group mapping
                    if isinstance(batch_indices, torch.Tensor):
                        batch_indices = batch_indices.cpu().numpy()
                    
                    # Ensure indices are within bounds
                    batch_indices = np.clip(batch_indices, 0, len(stored_group_indicators) - 1)
                    batch_group_indicators = torch.tensor(stored_group_indicators[batch_indices], dtype=torch.long, device=pred.device)
                else:
                    # Fallback to simple approach (first batch_size samples)
                    if len(stored_group_indicators) >= batch_size:
                        batch_group_indicators = torch.tensor(stored_group_indicators[:batch_size], dtype=torch.long, device=pred.device)
                    else:
                        # Fallback: use uniform group indicators if not enough stored
                        batch_group_indicators = torch.zeros(batch_size, dtype=torch.long, device=pred.device)
                        print("🔍 Fallback to uniform group indicators")
                
                # Use the fairness loss function
                return self.fairness_loss_fn(pred, durations, events, rank_mat, batch_group_indicators)
            
            @property
            def loss_history(self):
                """Access loss history from the fairness loss function."""
                return self.fairness_loss_fn.loss_history

        # Create the fairness loss function instance
        fairness_loss = FairnessLossWrapper(
            self.m_net,
            alpha,
            sigma,
            fairness_weight,
            self.m_hyperparameters.get("cuts")
        )
        
        # Replace the model's loss function with our fairness-aware one
        self.m_model.loss = fairness_loss
        
        print(f"🔍 Replaced loss function with fairness-aware loss in PyCox DeepHit (competing risks) model")

    def fit(self, data_train, parameters=None):
        if parameters['seed'] is not None:
            torch.manual_seed(parameters['seed'])

        if self.use_fairness:
            # Extract group indicators for fairness training
            x, y, group_indicators = self._df_to_xyg_(data_train)
            print(f"✅ Using fairness training with {len(group_indicators)} samples")
            print(f"   Group distribution: {np.bincount(group_indicators)}")
            
            # Store group indicators in the network for the loss function to access
            self.m_net._training_group_indicators = group_indicators
            
            # Create indexed data loader for proper batch tracking
            indexed_loader = self._create_indexed_dataloader(x, y, parameters['batch_size'])
            print(f"🔍 Using indexed data loader for accurate group mapping")
        elif self.use_weights:
            x, y, w = self._df_to_xyw_(data_train)
            print(f"✅ Using weighted training with {len(w)} samples")
            print(f"   Weight range: {w.min():.3f} to {w.max():.3f}, mean: {w.mean():.3f}")
            
            # Store weights in the network for the loss function to access
            self.m_net._training_weights = w
            
            # Create indexed data loader for proper batch tracking (only if not unweighted mode)
            ablation_mode = self.m_hyperparameters.get("ablation_mode", "full")
            if ablation_mode != "unweighted":
                indexed_loader = self._create_indexed_dataloader(x, y, parameters['batch_size'])
                print(f"🔍 Using indexed data loader for accurate weight mapping")
            else:
                print(f"🔍 Ablation mode '{ablation_mode}': using standard training despite weights")
        else:
            # Use EXACT original fit method for unweighted mode
            x, y = self._df_to_xy_(data_train)
            print(f"ℹ️  Using original DeepHit training (no weights)")

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
        if self.use_fairness or self.use_weights:
            # Use fairness or weighted training
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
        else:
            # Use EXACT original training for unweighted mode
            val_data = self._df_to_xy_(parameters["val_data"])
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

    def predict_CIF(self, x, parameters=None):
        _ = parameters
        x_ = _adapt_input_(x)
        return self.model.predict_cif(x_)

    def predict_pmf(self, x, parameters=None):
        _ = parameters
        x_ = _adapt_input_(x)
        return self.model.predict_pmf(x_)

    def predict_surv(self, x, parameters=None):
        _ = parameters
        x_ = _adapt_input_(x)
        return self.model.predict_surv_df(x_)

    def predict(self, x, parameters=None):
        return self.predict_CIF(x, parameters=None)

    def get_loss_history(self):
        """
        Get the loss history from training.
        
        Returns:
            Dictionary containing loss history if available, None otherwise
        """
        if hasattr(self, 'm_model') and hasattr(self.m_model, 'loss'):
            # Check if it's a fairness wrapper
            if hasattr(self.m_model.loss, 'loss_history'):
                return self.m_model.loss.loss_history
            # Check if it's a fairness wrapper that delegates to fairness_loss_fn
            elif hasattr(self.m_model.loss, 'fairness_loss_fn'):
                if hasattr(self.m_model.loss.fairness_loss_fn, 'loss_history'):
                    return self.m_model.loss.fairness_loss_fn.loss_history
        return None

    def plot_loss_components(self, save_path=None):
        """
        Plot the loss components during training.
        
        Args:
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        loss_history = self.get_loss_history()
        if loss_history is None:
            print("❌ No loss history available. Make sure to train with fairness or weighted mode enabled.")
            return
        
        # Check if loss history is empty
        if not loss_history['total_loss']:
            print("❌ Loss history is empty. This might mean:")
            print("   1. The model wasn't trained with fairness or weighted mode")
            print("   2. The loss function wasn't called during training")
            print("   3. The training didn't complete successfully")
            return
        
        # Determine if this is fairness or weighted mode based on available keys
        is_fairness_mode = 'fairness_loss' in loss_history
        is_weighted_mode = 'weighted_loss' in loss_history
        
        if is_fairness_mode:
            self._plot_fairness_components(loss_history, save_path)
        elif is_weighted_mode:
            self._plot_weighted_components(loss_history, save_path)
        else:
            print("❌ Unknown loss history format")
    
    def _plot_fairness_components(self, loss_history, save_path=None):
        """Plot fairness loss components."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Fairness Loss Components During Training - {type(self).__name__}', fontsize=16)
        
        # Plot 1: Total Loss
        axes[0, 0].plot(loss_history['total_loss'], label='Total Loss', color='red')
        axes[0, 0].set_title('Total Loss (Standard + Fairness)')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Standard Loss Components
        axes[0, 1].plot(loss_history['nll_loss'], label='NLL Loss (Est.)', color='blue', alpha=0.7)
        axes[0, 1].plot(loss_history['ranking_loss'], label='Ranking Loss (Est.)', color='green', alpha=0.7)
        axes[0, 1].set_title('Standard Loss Components (Estimated)')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Fairness Loss
        axes[0, 2].plot(loss_history['fairness_loss'], label='Fairness Penalty', color='purple')
        axes[0, 2].set_title('Fairness Penalty')
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('Fairness Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Group Mean Risks
        axes[1, 0].plot(loss_history['group1_mean_risk'], label='Group 1 Mean Risk', color='orange')
        axes[1, 0].plot(loss_history['group2_mean_risk'], label='Group 2 Mean Risk', color='brown')
        axes[1, 0].set_title('Mean Risk by Group')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Mean Risk')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Risk Difference
        risk_diff = np.array(loss_history['group1_mean_risk']) - np.array(loss_history['group2_mean_risk'])
        axes[1, 1].plot(risk_diff, label='Risk Difference (G1 - G2)', color='magenta')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Risk Difference Between Groups')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Risk Difference')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Loss Ratio
        if len(loss_history['fairness_loss']) > 0 and len(loss_history['nll_loss']) > 0:
            fairness_ratio = np.array(loss_history['fairness_loss']) / (np.array(loss_history['nll_loss']) + 1e-8)
            axes[1, 2].plot(fairness_ratio, label='Fairness/Standard Loss Ratio', color='cyan')
            axes[1, 2].set_title('Fairness to Standard Loss Ratio')
            axes[1, 2].set_xlabel('Training Step')
            axes[1, 2].set_ylabel('Ratio')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Fairness loss plot saved to: {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\n📈 Fairness Loss Summary:")
        try:
            total_loss_array = np.array(loss_history['total_loss'])
            finite_total = total_loss_array[np.isfinite(total_loss_array)]
            if len(finite_total) > 0:
                print(f"   Total Loss - Final: {finite_total[-1]:.4f}, Mean: {np.mean(finite_total):.4f}")
        except Exception as e:
            print(f"   Total Loss - Error: {e}")
        
        try:
            fairness_array = np.array(loss_history['fairness_loss'])
            finite_fairness = fairness_array[np.isfinite(fairness_array)]
            if len(finite_fairness) > 0:
                print(f"   Fairness Loss - Final: {finite_fairness[-1]:.4f}, Mean: {np.mean(finite_fairness):.4f}")
        except Exception as e:
            print(f"   Fairness Loss - Error: {e}")
        
        try:
            group1_array = np.array(loss_history['group1_mean_risk'])
            group2_array = np.array(loss_history['group2_mean_risk'])
            finite_group1 = group1_array[np.isfinite(group1_array)]
            finite_group2 = group2_array[np.isfinite(group2_array)]
            if len(finite_group1) > 0 and len(finite_group2) > 0:
                print(f"   Group 1 Mean Risk - Final: {finite_group1[-1]:.4f}, Mean: {np.mean(finite_group1):.4f}")
                print(f"   Group 2 Mean Risk - Final: {finite_group2[-1]:.4f}, Mean: {np.mean(finite_group2):.4f}")
                final_diff = finite_group1[-1] - finite_group2[-1]
                print(f"   Final Risk Difference: {final_diff:.4f}")
        except Exception as e:
            print(f"   Group Risks - Error: {e}")
        
        print(f"   Training Steps: {len(loss_history['total_loss'])}")
    
    def _plot_weighted_components(self, loss_history, save_path=None):
        """Plot weighted loss components (existing implementation)."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Weighted Loss Components During Training - {type(self).__name__}', fontsize=16)
        
        # Plot 1: Weighted Loss
        axes[0, 0].plot(loss_history['weighted_loss'], label='Weighted Loss', color='red')
        axes[0, 0].set_title('Weighted Loss (Pairwise Weighted)')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Total Loss (Unweighted)
        axes[0, 1].plot(loss_history['total_loss'], label='Total Loss (Unweighted)', color='blue')
        axes[0, 1].set_title('Total Loss (Unweighted)')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Loss Components (Estimated)
        axes[1, 0].plot(loss_history['likelihood_loss'], label='Likelihood Loss (Est.)', color='green', alpha=0.7)
        axes[1, 0].plot(loss_history['ranking_loss'], label='Ranking Loss (Est.)', color='orange', alpha=0.7)
        axes[1, 0].set_title('Loss Components (Estimated)')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Batch Weights
        axes[1, 1].plot(loss_history['batch_weights_mean'], label='Mean Batch Weight', color='purple')
        axes[1, 1].set_title('Mean Batch Weight')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Weighted loss plot saved to: {save_path}")
        
        plt.show()

    def get_current_loss_components(self, batch_data=None):
        """
        Get current loss components from the fairness loss function.
        
        Args:
            batch_data: Optional batch data to compute loss components for.
                      If None, uses the last batch from training.
        
        Returns:
            dict: Dictionary containing loss components
        """
        if not hasattr(self, 'm_model') or not hasattr(self.m_model, 'loss'):
            return None
        
        loss_fn = self.m_model.loss
        
        # If it's our fairness loss wrapper, get the actual loss function
        if hasattr(loss_fn, 'fairness_loss_fn'):
            loss_fn = loss_fn.fairness_loss_fn
        
        # Check if it's our fairness loss and has loss history
        if hasattr(loss_fn, 'loss_history') and loss_fn.loss_history:
            # Get the most recent components from loss history
            latest = {}
            
            # Extract all available components from loss history
            for key in ['total_loss', 'nll_loss', 'ranking_loss', 'fairness_loss', 
                       'group1_mean_risk', 'group2_mean_risk']:
                if key in loss_fn.loss_history and loss_fn.loss_history[key]:
                    latest[key] = loss_fn.loss_history[key][-1]
                else:
                    latest[key] = 0.0
            
            # Add fairness weight and other parameters
            if hasattr(loss_fn, 'fairness_weight'):
                latest['fairness_weight'] = loss_fn.fairness_weight
            if hasattr(loss_fn, 'alpha'):
                latest['alpha'] = loss_fn.alpha
            if hasattr(loss_fn, 'sigma'):
                latest['sigma'] = loss_fn.sigma
            
            # Calculate additional metrics
            if 'group1_mean_risk' in latest and 'group2_mean_risk' in latest:
                latest['risk_difference'] = abs(latest['group1_mean_risk'] - latest['group2_mean_risk'])
            
            if 'fairness_loss' in latest and 'total_loss' in latest:
                fairness_weight = latest.get('fairness_weight', 0.0)
                latest['fairness_contribution'] = fairness_weight * latest['fairness_loss']
                if latest['total_loss'] > 0:
                    latest['fairness_ratio'] = latest['fairness_contribution'] / latest['total_loss']
                else:
                    latest['fairness_ratio'] = 0.0
            
            # Only return if we have meaningful data
            if any(v != 0.0 for k, v in latest.items() if k not in ['fairness_weight', 'alpha', 'sigma']):
                return latest
        
        return None
