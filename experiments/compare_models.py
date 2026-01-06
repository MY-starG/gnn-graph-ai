import torch
import torch.nn.functional as F
import os
import sys
import time

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from models.gcn import GCN
from models.gat import GAT
from models.graphsage import GraphSAGE
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


def train_model(model_name, model, data, epochs=1000):
    print(f"\nTraining {model_name}...")
    print("-" * 60)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
        
        if epoch % 100 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    training_time = time.time() - start_time
    
    print(f"\n{model_name} Results:")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Best Test Accuracy: {best_test_acc:.4f}")
    print(f"  Training Time: {training_time:.2f}s")
    
    return {
        'model_name': model_name,
        'best_val': best_val_acc,
        'best_test': best_test_acc,
        'best_epoch': best_epoch,
        'training_time': training_time
    }


if __name__ == "__main__":
    print("="*80)
    print("COMPARING GCN, GAT, AND GRAPHSAGE ON CORA DATASET")
    print("="*80)
    
    # Load data
    loader = CoraDataLoader()
    data = loader.get_data()
    
    results = []
    
    # Train GCN
    gcn = GCN(
        num_features=loader.get_num_features(),
        hidden_channels=64,
        num_classes=loader.get_num_classes(),
        dropout=0.6
    )
    results.append(train_model("GCN", gcn, data))
    
    # Train GAT
    gat = GAT(
        num_features=loader.get_num_features(),
        hidden_channels=8,
        num_classes=loader.get_num_classes(),
        heads=8,
        dropout=0.6
    )
    results.append(train_model("GAT", gat, data))
    
    # Train GraphSAGE
    sage = GraphSAGE(
        num_features=loader.get_num_features(),
        hidden_channels=64,
        num_classes=loader.get_num_classes(),
        dropout=0.6
    )
    results.append(train_model("GraphSAGE", sage, data))
    
    # Print comparison table
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"{'Model':<15} {'Val Acc':<12} {'Test Acc':<12} {'Time (s)':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['model_name']:<15} {r['best_val']:<12.4f} {r['best_test']:<12.4f} {r['training_time']:<12.2f}")
    print("="*80)
    
    # Save results
    os.makedirs('../results/logs', exist_ok=True)
    with open('../results/logs/model_comparison.txt', 'w') as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"{'Model':<15} {'Val Acc':<12} {'Test Acc':<12} {'Time (s)':<12}\n")
        f.write("-"*80 + "\n")
        for r in results:
            f.write(f"{r['model_name']:<15} {r['best_val']:<12.4f} {r['best_test']:<12.4f} {r['training_time']:<12.2f}\n")
    
    print("\n✓ Results saved to results/logs/model_comparison.txt")