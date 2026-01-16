#!/usr/bin/env python3
"""
Metrics utility functions for bioassay evaluation.
"""

import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import pearsonr, kendalltau

def parse_entry_id(entry_id):
    """
    Parse entry_id to extract protein_id (target).
    Args:
        entry_id (str): Entry ID in format {prot_id}_{ligand_id}
    Returns:
        str: Protein ID (target)
    """
    return entry_id.split('_')[0]


def calculate_enrichment_factor(y_true, y_scores, percentages=[0.5, 1, 2, 5]):
    """
    Calculate Enrichment Factor at different percentages.
    Args:
        y_true (array): True binary labels
        y_scores (array): Predicted scores
        percentages (list): List of percentages to calculate EF
    Returns:
        dict: Dictionary with EF values for each percentage
    """
    # Handle both numeric and string labels
    y_true_processed = []
    for label in y_true:
        if isinstance(label, str):
            y_true_processed.append(1 if label == 'True' else 0)
        else:
            y_true_processed.append(1 if label == 1.0 else 0)
    y_true = np.array(y_true_processed)
    y_scores = np.array(y_scores)
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    n_total = len(y_true)
    n_actives = np.sum(y_true)
    ef_results = {}
    for percentage in percentages:
        n_top = int(n_total * percentage / 100)
        if n_top == 0:
            n_top = 1
        n_actives_top = np.sum(y_true_sorted[:n_top])
        if n_actives == 0:
            ef = np.nan
        else:
            random_rate = n_actives / n_total
            observed_rate = n_actives_top / n_top
            ef = observed_rate / random_rate
        ef_results[f'EF_{str(percentage).replace(".", "_")}'] = ef
    return ef_results


def calculate_metrics_per_target(scores_data):
    """
    Calculate metrics per target and average them.
    Args:
        scores_data (dict): Dictionary containing extracted scores
    Returns:
        dict: Dictionary containing calculated metrics
    """
    target_data = defaultdict(list)
    for entry_id, entry_scores in scores_data.items():
        target_id = parse_entry_id(entry_id)
        target_data[target_id].append({
            'label': entry_scores['label'],
            'score': entry_scores['score']
        })
    target_metrics = {}
    all_ap = []
    all_auroc = []
    all_auprc = []
    all_ef = defaultdict(list)
    ef_keys = ['EF_0_5', 'EF_1', 'EF_2', 'EF_5']
    for ef_key in ef_keys:
        all_ef[ef_key] = []
    for target_id, data_points in target_data.items():
        if len(data_points) < 2:
            continue
        y_true = [dp['label'] for dp in data_points]
        y_scores = [dp['score'] for dp in data_points]
        # Handle both numeric and string labels
        y_true_numeric = []
        for label in y_true:
            if isinstance(label, str):
                y_true_numeric.append(1 if label == 'True' else 0)
            else:
                y_true_numeric.append(1 if label == 1.0 else 0)
        if len(set(y_true_numeric)) < 2:
            continue
        ap = average_precision_score(y_true_numeric, y_scores)
        all_ap.append(ap)
        auroc = roc_auc_score(y_true_numeric, y_scores)
        all_auroc.append(auroc)
        auprc = average_precision_score(y_true_numeric, y_scores)
        all_auprc.append(auprc)
        ef_results = calculate_enrichment_factor(y_true, y_scores)
        for ef_key, ef_value in ef_results.items():
            all_ef[ef_key].append(ef_value)
        target_metrics[target_id] = {
            'AP': ap,
            'AUROC': auroc,
            **ef_results
        }
    avg_metrics = {
        'AP': np.nanmean(all_ap) if all_ap else np.nan,
        'AUROC': np.nanmean(all_auroc) if all_auroc else np.nan,
        'AUPRC': np.nanmean(all_auprc) if all_auprc else np.nan,
    }
    for ef_key in ef_keys:
        avg_metrics[ef_key] = np.nanmean(all_ef[ef_key]) if all_ef[ef_key] else np.nan
    return avg_metrics, target_metrics


def calculate_global_auroc(scores_data):
    """
    Calculate global AUROC across all compounds and targets.
    Args:
        scores_data (dict): Dictionary containing extracted scores
    Returns:
        float: Global AUROC
    """
    all_labels = []
    all_scores = []
    for entry_id, entry_scores in scores_data.items():
        all_labels.append(entry_scores['label'])
        all_scores.append(entry_scores['score'])
    # Handle both numeric and string labels
    y_true_numeric = []
    for label in all_labels:
        if isinstance(label, str):
            y_true_numeric.append(1 if label == 'True' else 0)
        else:
            y_true_numeric.append(1 if label == 1.0 else 0)
    if len(set(y_true_numeric)) < 2:
        return np.nan
    return roc_auc_score(y_true_numeric, all_scores)


def calculate_global_auprc(scores_data):
    """
    Calculate global AUPRC (Area Under Precision-Recall Curve) across all compounds and targets.
    Args:
        scores_data (dict): Dictionary containing extracted scores
    Returns:
        float: Global AUPRC (Average Precision)
    """
    all_labels = []
    all_scores = []
    for entry_id, entry_scores in scores_data.items():
        all_labels.append(entry_scores['label'])
        all_scores.append(entry_scores['score'])
    # Handle both numeric and string labels
    y_true_numeric = []
    for label in all_labels:
        if isinstance(label, str):
            y_true_numeric.append(1 if label == 'True' else 0)
        else:
            y_true_numeric.append(1 if label == 1.0 else 0)
    if len(set(y_true_numeric)) < 2:
        return np.nan
    return average_precision_score(y_true_numeric, all_scores)

def convert_log10_uM_to_kcal_mol(values, temperature=298.15):
    """
    Convert log10(uM) values to kcal/mol binding free energy.
    
    Args:
        values: array-like of log10(uM) values
        temperature: temperature in Kelvin (default 298.15K)
    
    Returns:
        numpy array of kcal/mol values
    """
    R = 1.987e-3  # kcal/(mol·K)
    values_array = np.array(values)
    
    # Convert log10(uM) to M, then to kcal/mol
    # log10(uM) -> uM -> M: divide by 1e6
    # ΔG = RT × ln(Kd_M)
    Kd_M = 10**(values_array - 6)  # Convert to M
    delta_G = R * temperature * np.log(Kd_M)
    
    return delta_G

def calculate_pearson_correlation(y_true, y_pred):
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        y_true: array of true values
        y_pred: array of predicted values
    
    Returns:
        float: Pearson correlation coefficient (or NaN if calculation fails)
    """
    try:
        if len(y_true) < 2 or len(set(y_true)) < 2:
            return np.nan
        correlation, _ = pearsonr(y_true, y_pred)
        return correlation
    except Exception:
        return np.nan


def calculate_kendall_tau(y_true, y_pred):
    """
    Calculate Kendall's Tau rank correlation coefficient.
    
    Args:
        y_true: array of true values
        y_pred: array of predicted values
    
    Returns:
        float: Kendall's Tau coefficient (or NaN if calculation fails)
    """
    try:
        if len(y_true) < 2:
            return np.nan
        tau, _ = kendalltau(y_true, y_pred)
        return tau
    except Exception:
        return np.nan


def calculate_pmae(y_true, y_pred):
    """
    Calculate Pairwise Mean Absolute Error (PMAE).
    PMAE is the MAE over the pair-wise difference of affinity between any pair of compounds.
    
    Args:
        y_true: array of true values
        y_pred: array of predicted values
    
    Returns:
        float: PMAE value (or NaN if calculation fails)
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) < 2:
            return np.nan
        
        # Calculate all pairwise differences
        n = len(y_true)
        pairwise_errors = []
        
        for i in range(n):
            for j in range(i + 1, n):
                true_diff = y_true[i] - y_true[j]
                pred_diff = y_pred[i] - y_pred[j]
                pairwise_errors.append(abs(true_diff - pred_diff))
        
        return np.mean(pairwise_errors) if pairwise_errors else np.nan
    except Exception:
        return np.nan


def calculate_mae_and_percentages(y_true, y_pred, with_centering=False):
    """
    Calculate MAE and percentages within 1 and 2 kcal/mol.
    
    Args:
        y_true: array of true values in kcal/mol
        y_pred: array of predicted values in kcal/mol
        with_centering: if True, center predictions to match mean of true values
    
    Returns:
        dict: Dictionary containing MAE, Perc_1kcal, and Perc_2kcal
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) == 0:
            return {"MAE": np.nan, "Perc_1kcal": np.nan, "Perc_2kcal": np.nan}
        
        if with_centering:
            # Center predictions to match mean of true values
            centering_offset = np.mean(y_true) - np.mean(y_pred)
            y_pred = y_pred + centering_offset
        
        # Calculate absolute errors
        abs_errors = np.abs(y_true - y_pred)
        
        # Calculate MAE
        mae = np.mean(abs_errors)
        
        # Calculate percentages within 1 and 2 kcal/mol
        perc_1kcal = np.mean(abs_errors <= 1.0) * 100
        perc_2kcal = np.mean(abs_errors <= 2.0) * 100
        
        return {
            "MAE": mae,
            "Perc_1kcal": perc_1kcal,
            "Perc_2kcal": perc_2kcal
        }
    except Exception:
        return {"MAE": np.nan, "Perc_1kcal": np.nan, "Perc_2kcal": np.nan}


def calculate_value_metrics_per_target(scores_data):
    """
    Calculate value regression metrics per target and average them using weighted average.
    All calculations are performed in kcal/mol units after converting from log10(uM).
    The final average is weighted by the number of data points in each target.
    
    Args:
        scores_data (dict): Dictionary containing extracted scores with structure:
                           {entry_id: {'label': log10_uM_value, 'binary': log10_uM_pred}}
    
    Returns:
        tuple: (avg_metrics, target_metrics)
            - avg_metrics: dict with weighted averaged metrics across all targets
            - target_metrics: dict with metrics for each individual target
    """
    target_data = defaultdict(list)
    
    # Group data by target
    for entry_id, entry_scores in scores_data.items():
        target_id = parse_entry_id(entry_id)
        target_data[target_id].append({
            'label': entry_scores['label'],
            'score': entry_scores['score']
        })
    
    target_metrics = {}
    all_pearson = []
    all_kendall = []
    all_pmae = []
    all_mae = []
    all_mae_cent = []
    all_perc_1kcal = []
    all_perc_2kcal = []
    all_perc_1kcal_cent = []
    all_perc_2kcal_cent = []
    all_weights = []  # Store weights (number of data points) for each target
    
    for target_id, data_points in target_data.items():
        if len(data_points) < 2:
            continue
        
        # Extract log10(uM) values
        y_true_log10 = [dp['label'] for dp in data_points]
        y_pred_log10 = [dp['score'] for dp in data_points]
        
        # Convert to kcal/mol
        y_true_kcal = convert_log10_uM_to_kcal_mol(y_true_log10)
        y_pred_kcal = convert_log10_uM_to_kcal_mol(y_pred_log10)
        
        # Calculate individual metrics
        pearson_r = calculate_pearson_correlation(y_true_kcal, y_pred_kcal)
        kendall_tau = calculate_kendall_tau(y_true_kcal, y_pred_kcal)
        pmae = calculate_pmae(y_true_kcal, y_pred_kcal)
        
        # Calculate MAE and percentages (regular version)
        mae_results = calculate_mae_and_percentages(y_true_kcal, y_pred_kcal, with_centering=False)
        
        # Calculate MAE and percentages (centered version)
        mae_cent_results = calculate_mae_and_percentages(y_true_kcal, y_pred_kcal, with_centering=True)
        
        # Store target-specific metrics
        target_metrics[target_id] = {
            'Pearson_R': pearson_r,
            'Kendall_Tau': kendall_tau,
            'PMAE': pmae,
            'MAE': mae_results['MAE'],
            'MAE_cent': mae_cent_results['MAE'],
            'Perc_1kcal': mae_results['Perc_1kcal'],
            'Perc_2kcal': mae_results['Perc_2kcal'],
            'Perc_1kcal_cent': mae_cent_results['Perc_1kcal'],
            'Perc_2kcal_cent': mae_cent_results['Perc_2kcal']
        }
        
        # Collect for weighted averaging
        all_pearson.append(pearson_r)
        all_kendall.append(kendall_tau)
        all_pmae.append(pmae)
        all_mae.append(mae_results['MAE'])
        all_mae_cent.append(mae_cent_results['MAE'])
        all_perc_1kcal.append(mae_results['Perc_1kcal'])
        all_perc_2kcal.append(mae_results['Perc_2kcal'])
        all_perc_1kcal_cent.append(mae_cent_results['Perc_1kcal'])
        all_perc_2kcal_cent.append(mae_cent_results['Perc_2kcal'])
        all_weights.append(len(data_points))  # Weight by number of data points
    
    def weighted_nanmean(values, weights):
        """Calculate weighted average ignoring NaN values."""
        if not values:
            return np.nan
        
        values = np.array(values)
        weights = np.array(weights)
        
        # Create mask for non-NaN values
        valid_mask = ~np.isnan(values)
        
        if not np.any(valid_mask):
            return np.nan
        
        # Filter out NaN values and their corresponding weights
        valid_values = values[valid_mask]
        valid_weights = weights[valid_mask]
        
        # Calculate weighted average
        return np.average(valid_values, weights=valid_weights)
    
    # Calculate weighted averaged metrics
    avg_metrics = {
        'Pearson_R': weighted_nanmean(all_pearson, all_weights),
        'Kendall_Tau': weighted_nanmean(all_kendall, all_weights),
        'PMAE': weighted_nanmean(all_pmae, all_weights),
        'MAE': weighted_nanmean(all_mae, all_weights),
        'MAE_cent': weighted_nanmean(all_mae_cent, all_weights),
        'Perc_1kcal': weighted_nanmean(all_perc_1kcal, all_weights),
        'Perc_2kcal': weighted_nanmean(all_perc_2kcal, all_weights),
        'Perc_1kcal_cent': weighted_nanmean(all_perc_1kcal_cent, all_weights),
        'Perc_2kcal_cent': weighted_nanmean(all_perc_2kcal_cent, all_weights),
    }
    
    return avg_metrics, target_metrics


def save_json_with_nan_handling(data, filename):
    """
    Save JSON data with proper NaN handling.
    Args:
        data: Data to save
        filename: Output filename
    """
    class NaNEncoder(json.JSONEncoder):
        def encode(self, obj):
            if isinstance(obj, float) and np.isnan(obj):
                return 'null'
            return super(NaNEncoder, self).encode(obj)
        def iterencode(self, obj, _one_shot=False):
            if isinstance(obj, float) and np.isnan(obj):
                yield 'null'
            else:
                for chunk in super(NaNEncoder, self).iterencode(obj, _one_shot):
                    yield chunk
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, cls=NaNEncoder) 