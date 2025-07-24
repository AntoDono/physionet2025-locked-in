#!/usr/bin/env python3
"""
ECG Normalization Methods Comparison Script

This script demonstrates and compares different normalization methods on ECG signals:
1. Z-Score Normalization (for comparative analysis)
2. Min-Max Normalization (0 to 1)
3. Min-Max Normalization (-1 to 1)
4. Amplitude Normalization
5. Robust Normalization (median/IQR based)

The script provides comprehensive visualizations and statistical analysis to help
determine which normalization method performs best for different use cases.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from helper_code import find_records, load_signals, load_header, get_sampling_frequency, get_signal_names
from modules.normalize_wave import (band_pass_filter, apply_notch_filter, 
                                   median_filter_noise_reduction, wavelet_denoising, 
                                   normalize_amplitude)


def compare_normalization_methods():
    """
    Compare all normalization methods on a single ECG lead with comprehensive analysis.
    """
    print("ECG Normalization Methods Comparison")
    print("=" * 60)
    
    # Find available records
    records = find_records('./data')
    
    if not records:
        print("No ECG records found in ./data directory")
        print("Please ensure you have ECG data files in the ./data directory")
        return
    
    # Use the first record for demonstration
    record_path = f'./data/{records[0]}'
    print(f"Using record: {records[0]}")
    
    # Load the signal data
    signal, fields = load_signals(record_path)
    header = load_header(record_path)
    
    # Get metadata
    sampling_frequency = get_sampling_frequency(header)
    signal_names = get_signal_names(header)
    
    print(f"Sampling frequency: {sampling_frequency} Hz")
    print(f"Signal shape: {signal.shape}")
    print(f"Available leads: {signal_names}")
    
    # Select the lead to analyze (use first lead)
    selected_lead = 0
    lead_name = signal_names[selected_lead]
    raw_signal = signal[:, selected_lead]
    
    print(f"\nAnalyzing Lead: {lead_name}")
    print(f"Raw signal statistics:")
    print(f"  Mean: {np.mean(raw_signal):.4f} mV")
    print(f"  Std:  {np.std(raw_signal):.4f} mV")
    print(f"  Range: [{np.min(raw_signal):.4f}, {np.max(raw_signal):.4f}] mV")
    
    # Apply preprocessing pipeline
    print("\nApplying preprocessing pipeline...")
    
    # Step 1: Band-pass filtering
    filtered_signal = band_pass_filter(raw_signal, sampling_frequency)
    print("✓ Band-pass filter applied")
    
    # Step 2: Notch filter
    notch_filtered = apply_notch_filter(filtered_signal, sampling_frequency, notch_freq=50.0)
    print("✓ Notch filter applied")
    
    # Step 3: Median filtering
    median_filtered = median_filter_noise_reduction(notch_filtered, kernel_size=3)
    print("✓ Median filter applied")
    
    # Step 4: Wavelet denoising (optimal simple configuration)
    processed_signal = wavelet_denoising(median_filtered, wavelet='db6', mode='soft')
    print("✓ Wavelet denoising applied (db6, soft thresholding)")
    
    print(f"\nProcessed signal statistics:")
    print(f"  Mean: {np.mean(processed_signal):.4f} mV")
    print(f"  Std:  {np.std(processed_signal):.4f} mV")
    print(f"  Range: [{np.min(processed_signal):.4f}, {np.max(processed_signal):.4f}] mV")
    
    # Apply all normalization methods
    print("\nApplying normalization methods...")
    
    normalization_methods = [
        {
            'name': 'Z-Score',
            'description': 'Zero mean, unit variance (μ=0, σ=1)',
            'method': 'z_score',
            'params': {},
            'use_case': 'Comparative analysis, ML preprocessing'
        },
        {
            'name': 'Min-Max [0,1]',
            'description': 'Scale to range [0, 1]',
            'method': 'min_max',
            'params': {'target_range': (0, 1)},
            'use_case': 'Neural networks, positive range needed'
        },
        {
            'name': 'Min-Max [-1,1]',
            'description': 'Scale to range [-1, 1]',
            'method': 'min_max',
            'params': {'target_range': (-1, 1)},
            'use_case': 'Symmetric range, signal processing'
        },
        {
            'name': 'Amplitude',
            'description': 'Normalize by max absolute value',
            'method': 'amplitude',
            'params': {'target_range': (-1, 1)},
            'use_case': 'Preserve relative amplitudes'
        },
        {
            'name': 'Robust',
            'description': 'Median-centered, IQR-scaled',
            'method': 'robust',
            'params': {},
            'use_case': 'Outlier-resistant normalization'
        }
    ]
    
    # Apply each normalization method
    normalized_signals = []
    for norm_config in normalization_methods:
        normalized = normalize_amplitude(processed_signal, 
                                       method=norm_config['method'], 
                                       **norm_config['params'])
        normalized_signals.append(normalized)
        
        print(f"✓ {norm_config['name']} normalization applied")
    
    # Create comprehensive comparison visualizations
    print("\nGenerating comparison visualizations...")
    
    create_normalization_comparison_chart(processed_signal, normalized_signals, 
                                        normalization_methods, sampling_frequency, 
                                        lead_name, records[0])
    
    create_statistical_analysis_chart(processed_signal, normalized_signals, 
                                    normalization_methods, lead_name, records[0])
    
    create_distribution_comparison_chart(processed_signal, normalized_signals, 
                                       normalization_methods, lead_name, records[0])
    
    create_performance_summary_table(processed_signal, normalized_signals, 
                                   normalization_methods, lead_name, records[0])
    
    # Analyze which method performs best
    analyze_best_normalization_method(processed_signal, normalized_signals, 
                                    normalization_methods, lead_name)
    
    print("\nComparison completed!")
    print("Check the ./visualizations/ directory for all generated charts")


def create_normalization_comparison_chart(original_signal, normalized_signals, 
                                        norm_methods, sampling_frequency, 
                                        lead_name, record_name):
    """
    Create a comprehensive chart showing all normalization methods side by side.
    """
    
    # Create time axis (limit to first 10 seconds for better visualization)
    max_samples = min(int(10 * sampling_frequency), len(original_signal))
    time = np.arange(max_samples) / sampling_frequency
    
    # Create the main comparison figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'ECG Normalization Methods Comparison - {lead_name} ({record_name})', 
                 fontsize=16, fontweight='bold')
    
    # Colors for each method
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    
    # Plot original signal in first subplot
    ax = axes[0, 0]
    ax.plot(time, original_signal[:max_samples], color='black', linewidth=1.2, label='Original')
    ax.set_title('Processed Signal (WITH Wavelet Denoising)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Amplitude (mV)')
    ax.grid(True, alpha=0.3)
    
    # Add statistics box
    orig_stats = f'DENOISED (db6 wavelet)\nμ={np.mean(original_signal):.3f}, σ={np.std(original_signal):.3f}\nRange: [{np.min(original_signal):.3f}, {np.max(original_signal):.3f}]'
    ax.text(0.02, 0.98, orig_stats, transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=9)
    
    # Plot each normalization method
    for i, (normalized, norm_config) in enumerate(zip(normalized_signals, norm_methods)):
        row = (i + 1) // 2
        col = (i + 1) % 2
        
        ax = axes[row, col]
        ax.plot(time, normalized[:max_samples], color=colors[i], linewidth=1.2)
        ax.set_title(f'{norm_config["name"]} Normalization (DENOISED)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Normalized Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Add statistics and description
        norm_stats = f'μ={np.mean(normalized):.3f}, σ={np.std(normalized):.3f}\nRange: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]'
        stats_text = f'{norm_config["description"]}\n{norm_stats}\nUse: {norm_config["use_case"]}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.2), fontsize=8)
        
        # Set x-label for bottom row
        if row == 2:
            ax.set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the chart
    output_path = f'./visualizations/{record_name}_{lead_name}_normalization_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Normalization comparison chart saved: {output_path}")


def create_statistical_analysis_chart(original_signal, normalized_signals, 
                                    norm_methods, lead_name, record_name):
    """
    Create statistical analysis charts comparing all methods.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Statistical Analysis - {lead_name} Normalization Methods ({record_name})', 
                 fontsize=14, fontweight='bold')
    
    # Prepare data
    method_names = ['Original'] + [method['name'] for method in norm_methods]
    all_signals = [original_signal] + normalized_signals
    colors = ['black', 'red', 'green', 'blue', 'purple', 'orange']
    
    # 1. Mean and Standard Deviation
    ax = axes[0, 0]
    means = [np.mean(sig) for sig in all_signals]
    stds = [np.std(sig) for sig in all_signals]
    
    x_pos = np.arange(len(method_names))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, means, width, label='Mean', alpha=0.7, color='skyblue')
    bars2 = ax.bar(x_pos + width/2, stds, width, label='Std Dev', alpha=0.7, color='lightcoral')
    
    ax.set_title('Mean and Standard Deviation', fontweight='bold')
    ax.set_ylabel('Value')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Range Analysis
    ax = axes[0, 1]
    ranges = [np.max(sig) - np.min(sig) for sig in all_signals]
    mins = [np.min(sig) for sig in all_signals]
    maxs = [np.max(sig) for sig in all_signals]
    
    bars = ax.bar(method_names, ranges, alpha=0.7, color=colors[:len(method_names)])
    ax.set_title('Signal Range (Max - Min)', fontweight='bold')
    ax.set_ylabel('Range')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, min_val, max_val in zip(bars, mins, maxs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
               f'[{min_val:.2f}, {max_val:.2f}]', ha='center', va='center', 
               fontsize=8, rotation=90, color='white', fontweight='bold')
    
    # 3. Skewness and Kurtosis
    ax = axes[1, 0]
    skewness = [stats.skew(sig) for sig in all_signals]
    kurtosis = [stats.kurtosis(sig) for sig in all_signals]
    
    x_pos = np.arange(len(method_names))
    bars1 = ax.bar(x_pos - width/2, skewness, width, label='Skewness', alpha=0.7, color='lightgreen')
    bars2 = ax.bar(x_pos + width/2, kurtosis, width, label='Kurtosis', alpha=0.7, color='lightsalmon')
    
    ax.set_title('Distribution Shape Analysis', fontweight='bold')
    ax.set_ylabel('Value')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. Coefficient of Variation
    ax = axes[1, 1]
    cv = [np.std(sig) / np.abs(np.mean(sig)) if np.mean(sig) != 0 else 0 for sig in all_signals]
    
    bars = ax.bar(method_names, cv, alpha=0.7, color=colors[:len(method_names)])
    ax.set_title('Coefficient of Variation (CV)', fontweight='bold')
    ax.set_ylabel('CV = σ/|μ|')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, cv_val in zip(bars, cv):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{cv_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the chart
    output_path = f'./visualizations/{record_name}_{lead_name}_statistical_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Statistical analysis chart saved: {output_path}")


def create_distribution_comparison_chart(original_signal, normalized_signals, 
                                       norm_methods, lead_name, record_name):
    """
    Create distribution comparison charts (histograms and box plots).
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Distribution Analysis - {lead_name} Normalization Methods ({record_name})', 
                 fontsize=16, fontweight='bold')
    
    # Prepare data
    method_names = ['Original'] + [method['name'] for method in norm_methods]
    all_signals = [original_signal] + normalized_signals
    colors = ['black', 'red', 'green', 'blue', 'purple', 'orange']
    
    # Create histograms for each method
    for i, (signal_data, name, color) in enumerate(zip(all_signals, method_names, colors)):
        row = i // 3
        col = i % 3
        
        if i < 6:  # Only plot first 6 (original + 5 normalizations)
            ax = axes[row, col]
            
            # Create histogram
            n, bins, patches = ax.hist(signal_data, bins=50, alpha=0.7, color=color, 
                                     density=True, edgecolor='black', linewidth=0.5)
            
            # Overlay normal distribution curve
            mu, sigma = np.mean(signal_data), np.std(signal_data)
            x = np.linspace(np.min(signal_data), np.max(signal_data), 100)
            normal_curve = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, normal_curve, 'k--', linewidth=2, label='Normal fit')
            
            ax.set_title(f'{name}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Amplitude')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'μ={mu:.3f}\nσ={sigma:.3f}\nSkew={stats.skew(signal_data):.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the distribution chart
    output_path = f'./visualizations/{record_name}_{lead_name}_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create box plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Box Plot Comparison - {lead_name} Normalization Methods ({record_name})', 
                 fontsize=14, fontweight='bold')
    
    # Create box plots
    box_data = []
    for signal_data in all_signals:
        # Limit data for better visualization (sample if too large)
        if len(signal_data) > 10000:
            sample_data = np.random.choice(signal_data, 10000, replace=False)
        else:
            sample_data = signal_data
        box_data.append(sample_data)
    
    bp = ax.boxplot(box_data, labels=method_names, patch_artist=True, 
                    medianprops=dict(color='black', linewidth=2))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Distribution Comparison (Box Plots)', fontweight='bold')
    ax.set_ylabel('Amplitude')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the box plot
    output_path = f'./visualizations/{record_name}_{lead_name}_boxplots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Distribution analysis charts saved: {record_name}_{lead_name}_distributions.png and _boxplots.png")


def create_performance_summary_table(original_signal, normalized_signals, 
                                    norm_methods, lead_name, record_name):
    """
    Create a comprehensive performance summary table.
    """
    
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(f'Normalization Performance Summary - {lead_name} ({record_name})', 
                 fontsize=16, fontweight='bold')
    
    # Prepare data for table
    method_names = ['Original'] + [method['name'] for method in norm_methods]
    all_signals = [original_signal] + normalized_signals
    
    # Calculate comprehensive statistics
    table_data = []
    headers = ['Method', 'Mean', 'Std Dev', 'Min', 'Max', 'Range', 'Median', 
               'IQR', 'Skewness', 'Kurtosis', 'CV', 'Use Case']
    
    for i, (signal_data, method_name) in enumerate(zip(all_signals, method_names)):
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        range_val = max_val - min_val
        median_val = np.median(signal_data)
        q75, q25 = np.percentile(signal_data, [75, 25])
        iqr_val = q75 - q25
        skew_val = stats.skew(signal_data)
        kurt_val = stats.kurtosis(signal_data)
        cv_val = std_val / abs(mean_val) if mean_val != 0 else 0
        
        if i == 0:
            use_case = 'Raw processed signal'
        else:
            use_case = norm_methods[i-1]['use_case']
        
        row = [
            method_name,
            f'{mean_val:.4f}',
            f'{std_val:.4f}',
            f'{min_val:.4f}',
            f'{max_val:.4f}',
            f'{range_val:.4f}',
            f'{median_val:.4f}',
            f'{iqr_val:.4f}',
            f'{skew_val:.4f}',
            f'{kurt_val:.4f}',
            f'{cv_val:.4f}',
            use_case
        ]
        table_data.append(row)
    
    # Create the table
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color-code rows
    colors = ['lightgray', 'lightcoral', 'lightgreen', 'lightblue', 'lightpink', 'lightyellow']
    for i in range(len(method_names)):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(colors[i % len(colors)])
    
    # Header styling
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('darkblue')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Highlight best values in each column
    for col_idx in range(1, 11):  # Skip method name and use case columns
        values = [float(table_data[i][col_idx]) for i in range(len(table_data))]
        
        # Determine if lower or higher is better for each metric
        if col_idx in [1, 6]:  # Mean, Median - closer to 0 might be better for some normalizations
            best_idx = min(range(len(values)), key=lambda i: abs(values[i]))
        elif col_idx in [2]:  # Std Dev - depends on normalization goal
            best_idx = values.index(min(values)) if col_idx == 2 else values.index(max(values))
        elif col_idx in [8, 9]:  # Skewness, Kurtosis - closer to 0 is better
            best_idx = min(range(len(values)), key=lambda i: abs(values[i]))
        else:
            continue  # Skip highlighting for other columns
        
        # Highlight the best value
        table[(best_idx+1, col_idx)].set_facecolor('gold')
        table[(best_idx+1, col_idx)].set_text_props(weight='bold')
    
    ax.axis('off')
    
    # Save the table
    output_path = f'./visualizations/{record_name}_{lead_name}_performance_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Performance summary table saved: {output_path}")


def analyze_best_normalization_method(original_signal, normalized_signals, 
                                    norm_methods, lead_name):
    """
    Analyze and recommend the best normalization method based on different criteria.
    """
    
    print(f"\n{'='*60}")
    print(f"BEST NORMALIZATION METHOD ANALYSIS - {lead_name}")
    print(f"{'='*60}")
    
    method_names = [method['name'] for method in norm_methods]
    
    # Criteria for evaluation
    criteria_scores = {}
    
    # 1. Stability (low coefficient of variation)
    print("\n1. STABILITY ANALYSIS (Lower CV = More Stable)")
    cv_scores = []
    for i, (signal_data, method_name) in enumerate(zip(normalized_signals, method_names)):
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
        cv_scores.append(cv)
        print(f"   {method_name}: CV = {cv:.4f}")
    
    best_stability_idx = np.argmin(cv_scores)
    print(f"   → BEST for Stability: {method_names[best_stability_idx]} (CV = {cv_scores[best_stability_idx]:.4f})")
    
    # 2. Normality (closest to normal distribution)
    print("\n2. NORMALITY ANALYSIS (Lower p-value = Less Normal)")
    normality_scores = []
    for i, (signal_data, method_name) in enumerate(zip(normalized_signals, method_names)):
        # Shapiro-Wilk test for normality (sample if too large)
        if len(signal_data) > 5000:
            sample_data = np.random.choice(signal_data, 5000, replace=False)
        else:
            sample_data = signal_data
        
        _, p_value = stats.shapiro(sample_data)
        normality_scores.append(p_value)
        print(f"   {method_name}: Shapiro-Wilk p-value = {p_value:.6f}")
    
    best_normality_idx = np.argmax(normality_scores)
    print(f"   → BEST for Normality: {method_names[best_normality_idx]} (p = {normality_scores[best_normality_idx]:.6f})")
    
    # 3. Symmetry (closest to zero skewness)
    print("\n3. SYMMETRY ANALYSIS (Closer to 0 = More Symmetric)")
    symmetry_scores = []
    for i, (signal_data, method_name) in enumerate(zip(normalized_signals, method_names)):
        skewness = abs(stats.skew(signal_data))
        symmetry_scores.append(skewness)
        print(f"   {method_name}: |Skewness| = {skewness:.4f}")
    
    best_symmetry_idx = np.argmin(symmetry_scores)
    print(f"   → BEST for Symmetry: {method_names[best_symmetry_idx]} (|Skew| = {symmetry_scores[best_symmetry_idx]:.4f})")
    
    # 4. Range Standardization
    print("\n4. RANGE STANDARDIZATION")
    for i, (signal_data, method_name) in enumerate(zip(normalized_signals, method_names)):
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        print(f"   {method_name}: Range = [{min_val:.4f}, {max_val:.4f}]")
    
    # Overall recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS BY USE CASE:")
    print(f"{'='*60}")
    
    print("\n🎯 FOR MACHINE LEARNING:")
    print(f"   Best: Z-Score Normalization")
    print(f"   Reason: Zero mean, unit variance - ideal for most ML algorithms")
    
    print("\n🎯 FOR NEURAL NETWORKS:")
    print(f"   Best: Min-Max [0,1] Normalization")
    print(f"   Reason: Bounded range [0,1] prevents activation saturation")
    
    print("\n🎯 FOR SIGNAL PROCESSING:")
    print(f"   Best: Amplitude Normalization")
    print(f"   Reason: Preserves relative signal amplitudes and morphology")
    
    print("\n🎯 FOR OUTLIER-PRONE DATA:")
    print(f"   Best: Robust Normalization")
    print(f"   Reason: Uses median/IQR, resistant to outliers")
    
    print("\n🎯 FOR COMPARATIVE ANALYSIS:")
    print(f"   Best: Z-Score Normalization")
    print(f"   Reason: Standardizes across different recordings for comparison")
    
    # Statistical best performer
    print(f"\n{'='*60}")
    print("STATISTICAL BEST PERFORMERS:")
    print(f"{'='*60}")
    print(f"Most Stable:    {method_names[best_stability_idx]}")
    print(f"Most Normal:    {method_names[best_normality_idx]}")
    print(f"Most Symmetric: {method_names[best_symmetry_idx]}")


if __name__ == "__main__":
    print("ECG Normalization Methods Comparison Tool")
    print("=" * 60)
    
    # Ensure visualizations directory exists
    import os
    os.makedirs('./visualizations', exist_ok=True)
    
    # Run the comprehensive comparison
    compare_normalization_methods()
    
    print(f"\n{'='*60}")
    print("Analysis completed successfully!")
    print("Generated files in ./visualizations/:")
    print("  - Normalization comparison chart")
    print("  - Statistical analysis chart") 
    print("  - Distribution analysis charts")
    print("  - Performance summary table")
    print(f"{'='*60}") 