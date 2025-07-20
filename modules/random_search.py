import random
import itertools
from typing import Dict, Any, Callable, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

def random_search(
    train_function: Callable,
    param_space: Dict[str, List[Any]], 
    n_trials: int = 20,
    random_seed: int = 42,
    verbose: bool = True,
    **train_kwargs
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Perform random search for hyperparameter optimization.
    
    Args:
        train_function: Function that takes parameters and returns (model, stats)
                       where stats contains 'accuracy', 'precision', 'recall', etc.
        param_space: Dictionary of parameter names to lists of possible values
                    Example: {'lr': [0.001, 0.01, 0.1], 'batch_size': [16, 32, 64]}
        n_trials: Number of random trials to run
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress and results
        **train_kwargs: Additional keyword arguments to pass to train_function
        
    Returns:
        best_params: Best hyperparameters found
        results: List of all results sorted by performance (best first)
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    if verbose:
        print(f"Starting Random Search with {n_trials} trials")
        print(f"Parameter space: {param_space}")
        print("=" * 60)
    
    results = []
    
    for trial in range(n_trials):
        if verbose:
            print(f"\nTrial {trial + 1}/{n_trials}")
            print("-" * 30)
        
        # Sample random parameters
        trial_params = {}
        for param_name, param_values in param_space.items():
            trial_params[param_name] = random.choice(param_values)
        
        if verbose:
            print(f"Parameters: {trial_params}")
        
        try:
            # Run training with sampled parameters
            start_time = datetime.now()
            model, stats = train_function(**trial_params, **train_kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Ensure stats has required keys with default values
            required_keys = ['accuracy', 'precision', 'recall', 'auc', 'loss']
            for key in required_keys:
                if key not in stats:
                    stats[key] = 0.0
            
            # Store results
            result = {
                'trial': trial + 1,
                'params': trial_params.copy(),
                'stats': stats.copy(),
                'duration_seconds': duration
            }
            results.append(result)
            
            if verbose:
                print(f"Results: Accuracy={stats['accuracy']:.4f}, Precision={stats['precision']:.4f}, "
                      f"Recall={stats['recall']:.4f}, AUC={stats['auc']:.4f}, Loss={stats['loss']:.4f}")
                print(f"Duration: {duration:.1f}s")
        
        except Exception as e:
            if verbose:
                print(f"Trial failed with error: {str(e)}")
            # Store failed trial
            result = {
                'trial': trial + 1,
                'params': trial_params.copy(),
                'stats': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.0, 'loss': float('inf')},
                'duration_seconds': 0.0,
                'error': str(e)
            }
            results.append(result)
    
    # Sort results by performance criteria
    # Primary: accuracy (descending), Secondary: precision (descending), Tertiary: recall (descending)
    results.sort(key=lambda x: (
        -x['stats']['accuracy'],  # Negative for descending
        -x['stats']['precision'], # Negative for descending  
        -x['stats']['recall']     # Negative for descending
    ))
    
    if verbose:
        print_ranked_results(results)
        print_ranked_parameters(results)
    
    # Get best parameters
    best_params = results[0]['params'] if results else {}
    
    return best_params, results


def print_ranked_results(results: List[Dict[str, Any]], top_n: int = None) -> None:
    """
    Print ranked results in a formatted table.
    
    Args:
        results: List of result dictionaries (should be pre-sorted)
        top_n: Number of top results to show (None = show all)
    """
    if not results:
        print("No results to display")
        return
    
    print("\n" + "=" * 120)
    print("RANKED RESULTS (Best to Worst)")
    print("=" * 120)
    
    # Show top_n results or all if top_n is None
    display_results = results[:top_n] if top_n else results
    
    for i, result in enumerate(display_results):
        stats = result['stats']
        params = result['params']
        
        print(f"\nRank {i+1}:")
        print(f"  Accuracy: {stats['accuracy']:.4f} | Precision: {stats['precision']:.4f} | Recall: {stats['recall']:.4f} | AUC: {stats['auc']:.4f} | Loss: {stats['loss']:.4f}")
        print(f"  Duration: {result['duration_seconds']:.1f}s")
        print(f"  Parameters: {params}")
        
        # Show error if present
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
    
    print("\n" + "=" * 120)
    
    # Summary statistics
    if results:
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best = valid_results[0]['stats']
            print(f"\nBEST PERFORMANCE:")
            print(f"  Accuracy: {best['accuracy']:.4f}")
            print(f"  Precision: {best['precision']:.4f}")
            print(f"  Recall: {best['recall']:.4f}")
            print(f"  AUC: {best['auc']:.4f}")
            print(f"  Loss: {best['loss']:.4f}")
            
            print(f"\nSUMMARY:")
            print(f"  Total trials: {len(results)}")
            print(f"  Successful trials: {len(valid_results)}")
            print(f"  Failed trials: {len(results) - len(valid_results)}")


def print_ranked_parameters(results: List[Dict[str, Any]], top_n: int = None) -> None:
    """
    Print just the ranked parameters list in a clean format.
    
    Args:
        results: List of result dictionaries (should be pre-sorted)
        top_n: Number of top results to show (None = show all)
    """
    if not results:
        print("No results to display")
        return
    
    print("\n" + "=" * 80)
    print("RANKED PARAMETERS (Best to Worst Performance)")
    print("=" * 80)
    
    # Show top_n results or all if top_n is None
    display_results = results[:top_n] if top_n else results
    valid_results = [r for r in display_results if 'error' not in r]
    
    for i, result in enumerate(valid_results):
        stats = result['stats']
        params = result['params']
        
        print(f"\n{i+1}. Accuracy: {stats['accuracy']:.4f}")
        for param_name, param_value in params.items():
            print(f"   {param_name}: {param_value}")
    
    print("\n" + "=" * 80)


def grid_search(
    train_function: Callable,
    param_space: Dict[str, List[Any]],
    random_seed: int = 42,
    verbose: bool = True,
    **train_kwargs
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Perform exhaustive grid search for hyperparameter optimization.
    
    Args:
        train_function: Function that takes parameters and returns (model, stats)
        param_space: Dictionary of parameter names to lists of possible values
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress and results
        **train_kwargs: Additional keyword arguments to pass to train_function
        
    Returns:
        best_params: Best hyperparameters found
        results: List of all results sorted by performance (best first)
    """
    
    # Generate all combinations
    param_names = list(param_space.keys())
    param_values = list(param_space.values())
    all_combinations = list(itertools.product(*param_values))
    
    if verbose:
        print(f"Starting Grid Search with {len(all_combinations)} combinations")
        print(f"Parameter space: {param_space}")
        print("=" * 60)
    
    results = []
    
    for trial, combination in enumerate(all_combinations):
        if verbose:
            print(f"\nTrial {trial + 1}/{len(all_combinations)}")
            print("-" * 30)
        
        # Create parameter dictionary
        trial_params = dict(zip(param_names, combination))
        
        if verbose:
            print(f"Parameters: {trial_params}")
        
        try:
            # Run training with current parameters
            start_time = datetime.now()
            model, stats = train_function(**trial_params, **train_kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Ensure stats has required keys with default values
            required_keys = ['accuracy', 'precision', 'recall', 'auc', 'loss']
            for key in required_keys:
                if key not in stats:
                    stats[key] = 0.0
            
            # Store results
            result = {
                'trial': trial + 1,
                'params': trial_params.copy(),
                'stats': stats.copy(),
                'duration_seconds': duration
            }
            results.append(result)
            
            if verbose:
                print(f"Results: Accuracy={stats['accuracy']:.4f}, Precision={stats['precision']:.4f}, "
                      f"Recall={stats['recall']:.4f}, AUC={stats['auc']:.4f}, Loss={stats['loss']:.4f}")
                print(f"Duration: {duration:.1f}s")
        
        except Exception as e:
            if verbose:
                print(f"Trial failed with error: {str(e)}")
            # Store failed trial
            result = {
                'trial': trial + 1,
                'params': trial_params.copy(),
                'stats': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.0, 'loss': float('inf')},
                'duration_seconds': 0.0,
                'error': str(e)
            }
            results.append(result)
    
    # Sort results by performance criteria
    results.sort(key=lambda x: (
        -x['stats']['accuracy'],  # Negative for descending
        -x['stats']['precision'], # Negative for descending  
        -x['stats']['recall']     # Negative for descending
    ))
    
    if verbose:
        print_ranked_results(results)
        print_ranked_parameters(results)
    
    # Get best parameters
    best_params = results[0]['params'] if results else {}
    
    return best_params, results


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def example_train_function(lr=0.001, batch_size=32, dropout=0.1, **kwargs):
    """
    Example training function for testing the random search.
    Returns fake stats that simulate a real training function.
    """
    import time
    import random
    
    # Simulate training time
    time.sleep(0.1)
    
    # Simulate performance based on parameters (fake but realistic)
    base_accuracy = 0.7
    lr_factor = 1.0 if lr == 0.001 else (0.95 if lr > 0.01 else 0.98)
    batch_factor = 1.0 if batch_size == 32 else (0.97 if batch_size > 64 else 0.99)
    dropout_factor = 1.0 if 0.1 <= dropout <= 0.3 else 0.95
    
    accuracy = base_accuracy * lr_factor * batch_factor * dropout_factor + random.uniform(-0.05, 0.05)
    accuracy = max(0.0, min(1.0, accuracy))  # Clamp to valid range
    
    precision = accuracy + random.uniform(-0.1, 0.05)
    recall = accuracy + random.uniform(-0.05, 0.1)
    auc = accuracy + random.uniform(-0.05, 0.05)
    loss = (1 - accuracy) + random.uniform(-0.1, 0.1)
    
    # Clamp all metrics to valid ranges
    precision = max(0.0, min(1.0, precision))
    recall = max(0.0, min(1.0, recall))
    auc = max(0.0, min(1.0, auc))
    loss = max(0.0, loss)
    
    stats = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'loss': loss
    }
    
    return None, stats  # Return None for model, stats dict


if __name__ == "__main__":
    # Handle imports when running as script vs when imported as module
    try:
        from ..implementations.gnn import train_model
    except ImportError:
        # When running as script, add parent directory to path
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from implementations.gnn import train_model
    # Test with example function
    param_space = {
        'lr': [0.0002, 0.002, 0.02, 0.2],
        'batch_size': [16, 32, 64, 128, 256],
        'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
        'epochs': [5, 10, 15, 20, 25, 30]
    }
    
    print("Testing Random Search...")
    best_params, results = random_search(
        train_function=train_model,
        param_space=param_space,
        n_trials=10,
        verbose=True,
        train_verbose=False,
        training_data_folder="data",
        model_folder="models",
        device="cuda:0"
    )
    
    print(f"\nBest parameters found: {best_params}")
 