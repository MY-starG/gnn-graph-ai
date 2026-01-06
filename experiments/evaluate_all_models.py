import torch
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from models.gcn import GCN
from models.gat import GAT
from models.graphsage import GraphSAGE
from utils.data_loader import CoraDataLoader
from evaluation.metrics import (
    evaluate_model, 
    print_metrics, 
    plot_confusion_matrix,
    plot_per_class_metrics,
    compare_models_metrics,
    save_metrics_to_file,
    CORA_CLASS_NAMES
)


def load_and_evaluate_model(model_class, model_path, data, loader, model_name):
    """Load a trained model and evaluate it"""
    
    # Initialize model
    if model_name == "GAT":
        model = model_class(
            num_features=loader.get_num_features(),
            hidden_channels=8,
            num_classes=loader.get_num_classes(),
            heads=8,
            dropout=0.6
        )
    else:
        model = model_class(
            num_features=loader.get_num_features(),
            hidden_channels=64,
            num_classes=loader.get_num_classes(),
            dropout=0.6
        )
    
    # Load trained weights if available
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"✓ Loaded {model_name} from {model_path}")
    else:
        print(f"⚠ No saved model found at {model_path}, using untrained model")
    
    # Evaluate
    metrics, y_true, y_pred = evaluate_model(model, data, mask_type='test')
    
    return metrics, model


if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    # Load data
    loader = CoraDataLoader()
    data = loader.get_data()
    
    # Model configurations
    models_config = [
        (GCN, '../results/models/best_gcn.pt', 'GCN'),
        (GAT, '../results/models/best_gat.pt', 'GAT'),
        (GraphSAGE, '../results/models/best_graphsage.pt', 'GraphSAGE')
    ]
    
    all_metrics = []
    model_names = []
    
    # Evaluate each model
    for model_class, model_path, model_name in models_config:
        print(f"\nEvaluating {model_name}...")
        metrics, model = load_and_evaluate_model(model_class, model_path, data, loader, model_name)
        
        # Print metrics
        print_metrics(metrics, model_name)
        
        # Save metrics to file
        save_metrics_to_file(
            metrics, 
            model_name, 
            CORA_CLASS_NAMES,
            f'../results/logs/{model_name.lower()}_metrics.txt'
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            CORA_CLASS_NAMES,
            model_name,
            f'../results/figures/{model_name.lower()}_confusion_matrix.png'
        )
        
        # Plot per-class metrics
        plot_per_class_metrics(
            metrics,
            CORA_CLASS_NAMES,
            model_name,
            f'../results/figures/{model_name.lower()}_per_class.png'
        )
        
        all_metrics.append(metrics)
        model_names.append(model_name)
    
    # Create comparison plot
    print("\nCreating model comparison plot...")
    compare_models_metrics(
        all_metrics,
        model_names,
        '../results/figures/models_comparison.png'
    )
    
    # Print summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Model':<15} {'Accuracy':<12} {'F1-Macro':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*80)
    for metrics, name in zip(all_metrics, model_names):
        print(f"{name:<15} {metrics['accuracy']:<12.4f} {metrics['f1_macro']:<12.4f} "
              f"{metrics['precision_macro']:<12.4f} {metrics['recall_macro']:<12.4f}")
    print("="*80)
    
    print("\n✓ All evaluations complete!")
    print("✓ Results saved to results/logs/")
    print("✓ Figures saved to results/figures/")