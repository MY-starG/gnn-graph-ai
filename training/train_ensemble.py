import torch
import torch.nn.functional as F
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from models.gcn import GCN
from models.gat import GAT
from utils.data_loader import CoraDataLoader


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum() / data.train_mask.sum()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    
    return train_acc.item(), val_acc.item(), test_acc.item()


def ensemble_test(models, data):
    """Test ensemble by averaging predictions"""
    for model in models:
        model.eval()
    
    # Get predictions from all models
    with torch.no_grad():
        outputs = [model(data.x, data.edge_index) for model in models]
        # Average the logits
        avg_out = torch.stack(outputs).mean(dim=0)
        pred = avg_out.argmax(dim=1)
    
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum() / data.train_mask.sum()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    
    return train_acc.item(), val_acc.item(), test_acc.item()


if __name__ == "__main__":
    print("Starting Ensemble Training (GCN + GAT)...")
    
    # ADDED: Show current directory info
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
    
    loader = CoraDataLoader()
    data = loader.get_data()
    print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges\n")
    
    # Initialize GCN
    print("Initializing GCN...")
    gcn_model = GCN(
        num_features=loader.get_num_features(),
        hidden_channels=64,
        num_classes=loader.get_num_classes(),
        dropout=0.6
    )
    gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.005, weight_decay=5e-4)
    
    # Initialize GAT
    print("Initializing GAT...")
    gat_model = GAT(
        num_features=loader.get_num_features(),
        hidden_channels=8,
        num_classes=loader.get_num_classes(),
        heads=8,
        dropout=0.6
    )
    gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.005, weight_decay=5e-4)
    
    print("\nTraining both models...")
    print("=" * 80)
    
    best_gcn_val = 0
    best_gcn_test = 0
    best_gat_val = 0
    best_gat_test = 0
    best_ensemble_val = 0
    best_ensemble_test = 0
    
    for epoch in range(1, 1001):
        # Train GCN
        gcn_loss = train(gcn_model, data, gcn_optimizer)
        gcn_train, gcn_val, gcn_test = test(gcn_model, data)
        
        if gcn_val > best_gcn_val:
            best_gcn_val = gcn_val
            best_gcn_test = gcn_test
        
        # Train GAT
        gat_loss = train(gat_model, data, gat_optimizer)
        gat_train, gat_val, gat_test = test(gat_model, data)
        
        if gat_val > best_gat_val:
            best_gat_val = gat_val
            best_gat_test = gat_test
        
        # Test ensemble
        ens_train, ens_val, ens_test = ensemble_test([gcn_model, gat_model], data)
        
        if ens_val > best_ensemble_val:
            best_ensemble_val = ens_val
            best_ensemble_test = ens_test
        
        if epoch % 50 == 0:
            print(f'\nEpoch: {epoch:03d}')
            print(f'  GCN  -> Loss: {gcn_loss:.4f}, Val: {gcn_val:.4f}, Test: {gcn_test:.4f}')
            print(f'  GAT  -> Loss: {gat_loss:.4f}, Val: {gat_val:.4f}, Test: {gat_test:.4f}')
            print(f'  ENS  -> Val: {ens_val:.4f}, Test: {ens_test:.4f}')
    
    print("\n" + "=" * 80)
    print("\nFINAL RESULTS:")
    print("-" * 80)
    print(f'GCN Best:      Val: {best_gcn_val:.4f} | Test: {best_gcn_test:.4f}')
    print(f'GAT Best:      Val: {best_gat_val:.4f} | Test: {best_gat_test:.4f}')
    print(f'ENSEMBLE Best: Val: {best_ensemble_val:.4f} | Test: {best_ensemble_test:.4f}')
    print("-" * 80)
    
    # ADDED: Show what paths we're trying to save to
    print("\n" + "=" * 80)
    print("SAVING FILES TO:")
    print("=" * 80)
    
    # Calculate absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    model_path_gcn = os.path.join(project_root, 'results', 'models', 'best_gcn.pt')
    model_path_gat = os.path.join(project_root, 'results', 'models', 'best_gat.pt')
    log_path = os.path.join(project_root, 'results', 'logs', 'ensemble_results.txt')
    
    print(f"1. GCN model: {model_path_gcn}")
    print(f"2. GAT model: {model_path_gat}")
    print(f"3. Results log: {log_path}")
    
    # Original save code (unchanged)
    os.makedirs('../results/models', exist_ok=True)
    torch.save(gcn_model.state_dict(), '../results/models/best_gcn.pt')
    torch.save(gat_model.state_dict(), '../results/models/best_gat.pt')
    
    # Save results to file
    os.makedirs('../results/logs', exist_ok=True)
    with open('../results/logs/ensemble_results.txt', 'w') as f:
        f.write("ENSEMBLE TRAINING RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f'GCN Best:      Val: {best_gcn_val:.4f} | Test: {best_gcn_test:.4f}\n')
        f.write(f'GAT Best:      Val: {best_gat_val:.4f} | Test: {best_gat_test:.4f}\n')
        f.write(f'ENSEMBLE Best: Val: {best_ensemble_val:.4f} | Test: {best_ensemble_test:.4f}\n')
    
    print("\n✓ Models saved to results/models/")
    print("✓ Results saved to results/logs/ensemble_results.txt")
    improvement = (best_ensemble_test - max(best_gcn_test, best_gat_test)) * 100
    print(f'\n✓ Ensemble improvement: +{improvement:.2f}% over best single model')
    
    # ADDED: Verify files were actually saved
    print("\n" + "=" * 80)
    print("VERIFYING SAVED FILES:")
    print("=" * 80)
    
    # Check each file
    files_to_check = [
        ('best_gcn.pt', os.path.join(project_root, 'results', 'models', 'best_gcn.pt')),
        ('best_gat.pt', os.path.join(project_root, 'results', 'models', 'best_gat.pt')),
        ('ensemble_results.txt', os.path.join(project_root, 'results', 'logs', 'ensemble_results.txt'))
    ]
    
    all_saved = True
    for file_name, file_path in files_to_check:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"{status} {file_name}: {file_path}")
        if not exists:
            all_saved = False
            # Show where it might be
            print(f"   File not found! Searching nearby...")
            # Try to find it
            for root, dirs, files in os.walk(project_root):
                if file_name in files:
                    print(f"   Found at: {os.path.join(root, file_name)}")
    
    if all_saved:
        print("\n✅ ALL FILES SAVED SUCCESSFULLY!")
    else:
        print("\n⚠ SOME FILES MAY NOT HAVE BEEN SAVED!")
        print(f"Check if 'results' folder exists at: {os.path.join(project_root, 'results')}")
        print(f"Current folder contents at project root:")
        try:
            for item in os.listdir(project_root):
                print(f"  - {item}")
        except:
            pass
    
    print("\nTraining complete!")