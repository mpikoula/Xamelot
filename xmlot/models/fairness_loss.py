"""
Fairness Loss Functions for DeepHit

This module implements fairness-aware loss functions that penalize differences in mean risk
between demographic subgroups, specifically designed for survival analysis with DeepHit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pycox.models.loss as pycox_loss
import pycox.models.utils as pycox_utils


def pad_col(input):
    """Pad with one column (row) of zeros after the last column (row)."""
    if len(input.shape) == 1:
        return F.pad(input, (0, 1), mode='constant', value=0)
    else:
        return F.pad(input, (0, 1), mode='constant', value=0)


def nll_pmf(phi, idx_durations, events, reduction='mean'):
    """Negative log-likelihood for PMF parameterization."""
    # phi: [B, T] or [B, R, T]
    # idx_durations: [B]
    # events: [B]
    
    if phi.dim() == 3:  # Competing risks
        # [B, R, T] -> [B, T]
        phi = phi.sum(dim=1)
    
    # Get PMF: pad and softmax
    pmf = pad_col(phi).softmax(dim=1)  # [B, T+1]
    
    # Clamp duration indices to valid range
    max_duration = pmf.shape[1] - 1  # Max valid index
    idx_durations = torch.clamp(idx_durations, 0, max_duration)
    
    # Get probability at event time
    event_pmf = pmf[torch.arange(len(events)), idx_durations]  # [B]
    
    # For censored events, use survival probability
    censored = (events == 0)
    if censored.any():
        # Survival = 1 - cumulative PMF
        surv = 1 - pmf[:, :idx_durations.max()+1].cumsum(dim=1)  # [B, T+1]
        surv_at_time = surv[torch.arange(len(events)), idx_durations]  # [B]
        event_pmf[censored] = surv_at_time[censored]
    
    # Negative log-likelihood
    nll = -torch.log(event_pmf + 1e-8)
    
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'none':
        return nll
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def rank_loss_deephit_single(phi, idx_durations, events, rank_mat, sigma, reduction='mean'):
    """Ranking loss for DeepHit single risk - matching PyCox's implementation exactly."""
    # phi: [B, T]
    # idx_durations: [B]
    # events: [B]
    # rank_mat: [B, B]
    
    # Match PyCox's implementation exactly
    idx_durations = idx_durations.view(-1, 1)
    pmf = pad_col(phi).softmax(1)
    y = torch.zeros_like(pmf).scatter(1, idx_durations, 1.)  # one-hot
    rank_loss = _rank_loss_deephit(pmf, y, rank_mat, sigma, reduction)
    return rank_loss


def rank_loss_deephit_cr(phi, idx_durations, events, rank_mat, sigma, reduction='mean'):
    """Ranking loss for DeepHit competing risks - matching PyCox's implementation exactly."""
    # phi: [B, R, T]
    # idx_durations: [B]
    # events: [B]
    # rank_mat: [B, B]
    
    # Match PyCox's implementation exactly
    idx_durations = idx_durations.view(-1)
    events = events.view(-1) - 1  # Convert to 0-based indexing
    event_01 = (events == -1).float()

    batch_size, n_risks = phi.shape[:2]
    pmf = pad_col(phi.view(batch_size, -1)).softmax(1)
    pmf = pmf[:, :-1].view(phi.shape)
    y = torch.zeros_like(pmf)
    y[torch.arange(batch_size), :, idx_durations] = 1.

    loss = []
    for i in range(n_risks):
        rank_loss_i = _rank_loss_deephit(pmf[:, i, :], y[:, i, :], rank_mat, sigma, 'none')
        loss.append(rank_loss_i.view(-1) * (events == i).float())

    if reduction == 'none':
        return sum(loss)
    elif reduction == 'mean':
        return sum([lo.mean() for lo in loss])
    elif reduction == 'sum':
        return sum([lo.sum() for lo in loss])
    else:
        return _reduction(sum(loss), reduction)


def _rank_loss_deephit(pmf, y, rank_mat, sigma, reduction='mean'):
    """Ranking loss from DeepHit - matching PyCox's implementation exactly."""
    r = _diff_cdf_at_time_i(pmf, y)
    loss = rank_mat * torch.exp(-r/sigma)
    loss = loss.mean(1, keepdim=True)
    return _reduction(loss, reduction)


def _diff_cdf_at_time_i(pmf, y):
    """R is the matrix from the DeepHit code giving the difference in CDF between individual
    i and j, at the event time of j. 
    I.e: R_ij = F_i(T_i) - F_j(T_i)
    
    Arguments:
        pmf {torch.tensor} -- Matrix with probability mass function pmf_ij = f_i(t_j)
        y {torch.tensor} -- Matrix with indicator of duration/censor time.
    
    Returns:
        torch.tensor -- R_ij = F_i(T_i) - F_j(T_i)
    """
    n = pmf.shape[0]
    ones = torch.ones((n, 1), device=pmf.device)
    r = pmf.cumsum(1).matmul(y.transpose(0, 1))
    diag_r = r.diag().view(1, -1)
    r = ones.matmul(diag_r) - r
    return r.transpose(0, 1)


def _reduction(loss, reduction='mean'):
    """Apply reduction to loss tensor."""
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")


class DeepHitFairLoss(nn.Module):
    """
    Fairness-aware DeepHit loss that penalizes differences in mean risk between subgroups.
    
    Loss = α·NLL + (1-α)·ranking_loss + fairness_weight·[(μ_group1 - μ_group2)²]
    
    where μ_group1 and μ_group2 are the mean risks for different demographic subgroups.
    """
    
    def __init__(self, alpha=0.2, sigma=0.1, fairness_weight=0.1, reduction='mean'):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if fairness_weight < 0:
            raise ValueError("fairness_weight must be non-negative")
            
        self.alpha = alpha
        self.sigma = sigma
        self.fairness_weight = fairness_weight
        self.reduction = reduction
        
        # Track loss components for debugging
        self.loss_history = {
            'total_loss': [],
            'nll_loss': [],
            'ranking_loss': [],
            'fairness_loss': [],
            'group1_mean_risk': [],
            'group2_mean_risk': []
        }
        self.step_counter = 0

    def forward(self, pred, durations, events, rank_mat=None, group_indicator=None):
        """
        Compute fairness-aware DeepHit loss.
        
        Args:
            pred: [batch_size, num_risks, num_time_bins] or [num_risks, batch_size, num_time_bins]
                  Raw network outputs
            durations: [batch_size] Duration indices
            events: [batch_size] Event indicators (0=censored, 1+=event type)
            rank_mat: [batch_size, batch_size] Optional ranking matrix
            group_indicator: [batch_size] Group membership (0=group1, 1=group2)
        
        Returns:
            Total loss including fairness penalty
        """
        self.step_counter += 1
        
        # Handle tensor shape: pred can be [batch_size, num_risks, num_time_bins] or [num_risks, batch_size, num_time_bins]
        if pred.shape[0] < pred.shape[1]:  # [num_risks, batch_size, num_time_bins]
            pred = pred.transpose(0, 1)  # [batch_size, num_risks, num_time_bins]
        
        batch_size = pred.shape[0]
        
        # Ensure all inputs are tensors
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not isinstance(durations, torch.Tensor):
            durations = torch.tensor(durations, dtype=torch.long)
        if not isinstance(events, torch.Tensor):
            events = torch.tensor(events, dtype=torch.long)
        
        # Handle group indicator
        if group_indicator is None:
            # If no group indicator provided, use standard loss
            return self._compute_standard_loss(pred, durations, events, rank_mat)
        
        if not isinstance(group_indicator, torch.Tensor):
            group_indicator = torch.tensor(group_indicator, dtype=torch.long)
        
        # Ensure group_indicator is on the same device as pred
        if group_indicator.device != pred.device:
            group_indicator = group_indicator.to(pred.device)
        
        # 1) Compute standard DeepHit loss components (exactly like PyCox)
        nll_loss, ranking_loss = self._compute_standard_loss_components(pred, durations, events, rank_mat)
        standard_loss = self.alpha * nll_loss + (1 - self.alpha) * ranking_loss
        
        # 2) Compute fairness penalty
        fairness_loss = self._compute_fairness_penalty(pred, group_indicator)
        
        # 3) Combine losses
        total_loss = standard_loss + self.fairness_weight * fairness_loss
        
        # Track loss components
        self.loss_history['total_loss'].append(total_loss.item())
        self.loss_history['nll_loss'].append(nll_loss.item())
        self.loss_history['ranking_loss'].append(ranking_loss.item())
        self.loss_history['fairness_loss'].append(fairness_loss.item())
        
        return total_loss
    
    def _compute_standard_loss(self, pred, durations, events, rank_mat):
        """Compute standard DeepHit loss without fairness penalty."""
        nll_loss, ranking_loss = self._compute_standard_loss_components(pred, durations, events, rank_mat)
        return self.alpha * nll_loss + (1 - self.alpha) * ranking_loss
    
    def _compute_standard_loss_components(self, pred, durations, events, rank_mat):
        """Compute NLL and ranking loss components exactly like PyCox."""
        # Compute NLL loss
        nll_loss = nll_pmf(pred, durations, events, self.reduction)
        
        # Compute ranking loss for competing risks
        if rank_mat is None:
            # Create ranking matrix based on duration comparisons
            # Only consider uncensored events for ranking
            uncensored = (events > 0)
            if not uncensored.any():
                # If no uncensored events, ranking loss should be 0
                ranking_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
            else:
                # Create ranking matrix for uncensored samples
                uncensored_indices = torch.where(uncensored)[0]
                n_uncensored = len(uncensored_indices)
                
                rank_mat = torch.zeros(n_uncensored, n_uncensored, device=pred.device)
                for i in range(n_uncensored):
                    for j in range(n_uncensored):
                        if i != j:
                            idx_i = uncensored_indices[i]
                            idx_j = uncensored_indices[j]
                            # If i has shorter duration than j, i should be ranked higher
                            if durations[idx_i] < durations[idx_j]:
                                rank_mat[i, j] = 1.0
                
                # Use only the uncensored samples for ranking loss
                pred_uncensored = pred[uncensored]
                durations_uncensored = durations[uncensored]
                events_uncensored = events[uncensored]
                
                ranking_loss = rank_loss_deephit_cr(
                    pred_uncensored, durations_uncensored, events_uncensored, 
                    rank_mat, self.sigma, self.reduction
                )
        else:
            # Use provided ranking matrix
            ranking_loss = rank_loss_deephit_cr(pred, durations, events, rank_mat, self.sigma, self.reduction)
        
        return nll_loss, ranking_loss
    
    def _compute_fairness_penalty(self, pred, group_indicator):
        """
        Compute fairness penalty based on difference in mean risk between groups.
        
        Args:
            pred: [batch_size, num_risks, num_time_bins] Raw network outputs
            group_indicator: [batch_size] Group membership (0=group1, 1=group2)
        
        Returns:
            Fairness penalty: (μ_group1 - μ_group2)²
        """
        # For competing risks, sum across risks to get total event probability
        if pred.dim() == 3:  # [batch_size, num_risks, num_time_bins]
            pred_total = pred.sum(dim=1)  # [batch_size, num_time_bins]
        else:  # [batch_size, num_time_bins]
            pred_total = pred
        
        # Get PMF using PyCox's exact method
        pmf = pad_col(pred_total).softmax(dim=1)  # [batch_size, num_time_bins+1]
        
        # Risk = P(event sometime) = sum of PMF (excluding censoring at time 0)
        # For DeepHit, the first column is typically censoring, so we sum from index 1
        if pmf.shape[1] > 1:
            risk = pmf[:, 1:].sum(dim=1, keepdim=True)  # [batch_size, 1]
        else:
            risk = pmf.sum(dim=1, keepdim=True)  # [batch_size, 1]
        
        # Compute mean risk for each group
        eps = 1e-8
        
        # Group 1 (group_indicator == 0)
        group1_mask = (group_indicator == 0).float().unsqueeze(1)  # [batch_size, 1]
        group1_sum = (risk * group1_mask).sum()
        group1_count = group1_mask.sum() + eps
        mu_group1 = group1_sum / group1_count
        
        # Group 2 (group_indicator == 1)
        group2_mask = (group_indicator == 1).float().unsqueeze(1)  # [batch_size, 1]
        group2_sum = (risk * group2_mask).sum()
        group2_count = group2_mask.sum() + eps
        mu_group2 = group2_sum / group2_count
        
        # Track mean risks for debugging
        self.loss_history['group1_mean_risk'].append(mu_group1.item())
        self.loss_history['group2_mean_risk'].append(mu_group2.item())
        
        # Compute squared difference penalty
        fairness_penalty = (mu_group1 - mu_group2).pow(2)
        
        return fairness_penalty

    def get_loss_components(self, pred, durations, events, rank_mat=None, group_indicator=None):
        """
        Get individual loss components for analysis.
        
        Args:
            pred: [batch_size, num_risks, num_time_bins] or [num_risks, batch_size, num_time_bins]
                  Raw network outputs
            durations: [batch_size] Duration indices
            events: [batch_size] Event indicators (0=censored, 1+=event type)
            rank_mat: [batch_size, batch_size] Optional ranking matrix
            group_indicator: [batch_size] Group membership (0=group1, 1=group2)
        
        Returns:
            dict: Dictionary containing individual loss components
        """
        # Handle tensor shape: pred can be [batch_size, num_risks, num_time_bins] or [num_risks, batch_size, num_time_bins]
        if pred.shape[0] < pred.shape[1]:  # [num_risks, batch_size, num_time_bins]
            pred = pred.transpose(0, 1)  # [batch_size, num_risks, num_time_bins]
        
        # Ensure all inputs are tensors
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not isinstance(durations, torch.Tensor):
            durations = torch.tensor(durations, dtype=torch.long)
        if not isinstance(events, torch.Tensor):
            events = torch.tensor(events, dtype=torch.long)
        
        # Compute standard loss components
        nll_loss, ranking_loss = self._compute_standard_loss_components(pred, durations, events, rank_mat)
        standard_loss = self.alpha * nll_loss + (1 - self.alpha) * ranking_loss
        
        # Initialize result
        result = {
            'nll_loss': nll_loss.item(),
            'ranking_loss': ranking_loss.item(),
            'standard_loss': standard_loss.item(),
            'fairness_loss': 0.0,
            'total_loss': standard_loss.item(),
            'alpha': self.alpha,
            'sigma': self.sigma,
            'fairness_weight': self.fairness_weight
        }
        
        # Compute fairness penalty if group indicator is provided
        if group_indicator is not None:
            if not isinstance(group_indicator, torch.Tensor):
                group_indicator = torch.tensor(group_indicator, dtype=torch.long)
            
            # Ensure group_indicator is on the same device as pred
            if group_indicator.device != pred.device:
                group_indicator = group_indicator.to(pred.device)
            
            fairness_loss = self._compute_fairness_penalty(pred, group_indicator)
            total_loss = standard_loss + self.fairness_weight * fairness_loss
            
            result.update({
                'fairness_loss': fairness_loss.item(),
                'total_loss': total_loss.item(),
                'group1_mean_risk': self.loss_history['group1_mean_risk'][-1] if self.loss_history['group1_mean_risk'] else 0.0,
                'group2_mean_risk': self.loss_history['group2_mean_risk'][-1] if self.loss_history['group2_mean_risk'] else 0.0,
                'risk_difference': abs(result.get('group1_mean_risk', 0.0) - result.get('group2_mean_risk', 0.0))
            })
        
        return result


class DeepHitSingleFairLoss(nn.Module):
    """
    Fairness-aware DeepHit loss for single-risk scenarios.
    
    Loss = α·NLL + (1-α)·ranking_loss + fairness_weight·[(μ_group1 - μ_group2)²]
    """
    
    def __init__(self, alpha=0.2, sigma=0.1, fairness_weight=0.1, reduction='mean'):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if fairness_weight < 0:
            raise ValueError("fairness_weight must be non-negative")
            
        self.alpha = alpha
        self.sigma = sigma
        self.fairness_weight = fairness_weight
        self.reduction = reduction
        
        # Track loss components for debugging
        self.loss_history = {
            'total_loss': [],
            'nll_loss': [],
            'ranking_loss': [],
            'fairness_loss': [],
            'group1_mean_risk': [],
            'group2_mean_risk': []
        }
        self.step_counter = 0

    def forward(self, pred, durations, events, rank_mat=None, group_indicator=None):
        """
        Compute fairness-aware DeepHit loss for single-risk scenarios.
        
        Args:
            pred: [batch_size, num_time_bins] Raw network outputs
            durations: [batch_size] Duration indices
            events: [batch_size] Event indicators (0=censored, 1=event)
            rank_mat: [batch_size, batch_size] Optional ranking matrix
            group_indicator: [batch_size] Group membership (0=group1, 1=group2)
        
        Returns:
            Total loss including fairness penalty
        """
        self.step_counter += 1
        
        # Ensure all inputs are tensors
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not isinstance(durations, torch.Tensor):
            durations = torch.tensor(durations, dtype=torch.long)
        if not isinstance(events, torch.Tensor):
            events = torch.tensor(events, dtype=torch.long)
        
        # Handle group indicator
        if group_indicator is None:
            # If no group indicator provided, use standard loss
            return self._compute_standard_loss(pred, durations, events, rank_mat)
        
        if not isinstance(group_indicator, torch.Tensor):
            group_indicator = torch.tensor(group_indicator, dtype=torch.long)
        
        # Ensure group_indicator is on the same device as pred
        if group_indicator.device != pred.device:
            group_indicator = group_indicator.to(pred.device)
        
        # 1) Compute standard DeepHit loss components (exactly like PyCox)
        nll_loss, ranking_loss = self._compute_standard_loss_components(pred, durations, events, rank_mat)
        standard_loss = self.alpha * nll_loss + (1 - self.alpha) * ranking_loss
        
        # 2) Compute fairness penalty
        fairness_loss = self._compute_fairness_penalty(pred, group_indicator)
        
        # 3) Combine losses
        total_loss = standard_loss + self.fairness_weight * fairness_loss
        
        # Track loss components
        self.loss_history['total_loss'].append(total_loss.item())
        self.loss_history['nll_loss'].append(nll_loss.item())
        self.loss_history['ranking_loss'].append(ranking_loss.item())
        self.loss_history['fairness_loss'].append(fairness_loss.item())
        
        return total_loss
    
    def _compute_standard_loss(self, pred, durations, events, rank_mat):
        """Compute standard DeepHit loss without fairness penalty."""
        nll_loss, ranking_loss = self._compute_standard_loss_components(pred, durations, events, rank_mat)
        return self.alpha * nll_loss + (1 - self.alpha) * ranking_loss
    
    def _compute_standard_loss_components(self, pred, durations, events, rank_mat):
        """Compute NLL and ranking loss components exactly like PyCox."""
        # Compute NLL loss
        nll_loss = nll_pmf(pred, durations, events, self.reduction)
        
        # Compute ranking loss
        if rank_mat is None:
            # Create ranking matrix based on duration comparisons
            # Only consider uncensored events for ranking
            uncensored = (events > 0)
            if not uncensored.any():
                # If no uncensored events, ranking loss should be 0
                ranking_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
            else:
                # Create ranking matrix for uncensored samples
                uncensored_indices = torch.where(uncensored)[0]
                n_uncensored = len(uncensored_indices)
                
                rank_mat = torch.zeros(n_uncensored, n_uncensored, device=pred.device)
                for i in range(n_uncensored):
                    for j in range(n_uncensored):
                        if i != j:
                            idx_i = uncensored_indices[i]
                            idx_j = uncensored_indices[j]
                            # If i has shorter duration than j, i should be ranked higher
                            if durations[idx_i] < durations[idx_j]:
                                rank_mat[i, j] = 1.0
                
                # Use only the uncensored samples for ranking loss
                pred_uncensored = pred[uncensored]
                durations_uncensored = durations[uncensored]
                events_uncensored = events[uncensored]
                
                ranking_loss = rank_loss_deephit_single(
                    pred_uncensored, durations_uncensored, events_uncensored, 
                    rank_mat, self.sigma, self.reduction
                )
        else:
            # Use provided ranking matrix
            ranking_loss = rank_loss_deephit_single(pred, durations, events, rank_mat, self.sigma, self.reduction)
        
        return nll_loss, ranking_loss
    
    def _compute_fairness_penalty(self, pred, group_indicator):
        """
        Compute fairness penalty based on difference in mean risk between groups.
        
        Args:
            pred: [batch_size, num_time_bins] Raw network outputs
            group_indicator: [batch_size] Group membership (0=group1, 1=group2)
        
        Returns:
            Fairness penalty: (μ_group1 - μ_group2)²
        """
        # Get PMF using PyCox's exact method
        pmf = pad_col(pred).softmax(dim=1)  # [batch_size, num_time_bins+1]
        
        # Risk = P(event sometime) = sum of PMF (excluding censoring at time 0)
        if pmf.shape[1] > 1:
            risk = pmf[:, 1:].sum(dim=1, keepdim=True)  # [batch_size, 1]
        else:
            risk = pmf.sum(dim=1, keepdim=True)  # [batch_size, 1]
        
        # Compute mean risk for each group
        eps = 1e-8
        
        # Group 1 (group_indicator == 0)
        group1_mask = (group_indicator == 0).float().unsqueeze(1)  # [batch_size, 1]
        group1_sum = (risk * group1_mask).sum()
        group1_count = group1_mask.sum() + eps
        mu_group1 = group1_sum / group1_count
        
        # Group 2 (group_indicator == 1)
        group2_mask = (group_indicator == 1).float().unsqueeze(1)  # [batch_size, 1]
        group2_sum = (risk * group2_mask).sum()
        group2_count = group2_mask.sum() + eps
        mu_group2 = group2_sum / group2_count
        
        # Track mean risks for debugging
        self.loss_history['group1_mean_risk'].append(mu_group1.item())
        self.loss_history['group2_mean_risk'].append(mu_group2.item())
        
        # Compute squared difference penalty
        fairness_penalty = (mu_group1 - mu_group2).pow(2)
        
        return fairness_penalty 

    def get_loss_components(self, pred, durations, events, rank_mat=None, group_indicator=None):
        """
        Get individual loss components for analysis.
        
        Args:
            pred: [batch_size, num_time_bins] Raw network outputs
            durations: [batch_size] Duration indices
            events: [batch_size] Event indicators (0=censored, 1=event)
            rank_mat: [batch_size, batch_size] Optional ranking matrix
            group_indicator: [batch_size] Group membership (0=group1, 1=group2)
        
        Returns:
            dict: Dictionary containing individual loss components
        """
        # Ensure all inputs are tensors
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not isinstance(durations, torch.Tensor):
            durations = torch.tensor(durations, dtype=torch.long)
        if not isinstance(events, torch.Tensor):
            events = torch.tensor(events, dtype=torch.long)
        
        # Compute standard loss components
        nll_loss, ranking_loss = self._compute_standard_loss_components(pred, durations, events, rank_mat)
        standard_loss = self.alpha * nll_loss + (1 - self.alpha) * ranking_loss
        
        # Initialize result
        result = {
            'nll_loss': nll_loss.item(),
            'ranking_loss': ranking_loss.item(),
            'standard_loss': standard_loss.item(),
            'fairness_loss': 0.0,
            'total_loss': standard_loss.item(),
            'alpha': self.alpha,
            'sigma': self.sigma,
            'fairness_weight': self.fairness_weight
        }
        
        # Compute fairness penalty if group indicator is provided
        if group_indicator is not None:
            if not isinstance(group_indicator, torch.Tensor):
                group_indicator = torch.tensor(group_indicator, dtype=torch.long)
            
            # Ensure group_indicator is on the same device as pred
            if group_indicator.device != pred.device:
                group_indicator = group_indicator.to(pred.device)
            
            fairness_loss = self._compute_fairness_penalty(pred, group_indicator)
            total_loss = standard_loss + self.fairness_weight * fairness_loss
            
            result.update({
                'fairness_loss': fairness_loss.item(),
                'total_loss': total_loss.item(),
                'group1_mean_risk': self.loss_history['group1_mean_risk'][-1] if self.loss_history['group1_mean_risk'] else 0.0,
                'group2_mean_risk': self.loss_history['group2_mean_risk'][-1] if self.loss_history['group2_mean_risk'] else 0.0,
                'risk_difference': abs(result.get('group1_mean_risk', 0.0) - result.get('group2_mean_risk', 0.0))
            })
        
        return result 