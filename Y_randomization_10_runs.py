from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,)
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from training_advanced import EdgeAwareGCNNet
from Smiles_to_graph import smiles_to_graph

TRAIN_FILE = Path('train.csv')
VALIDATION_FILE = Path('validation.csv')
TEST_FILE = Path('test.csv')
OUTPUT_DIR = Path('GNN_Y_Randomization_Results')
SMILES_COL = 'SMILES'
LABEL_COL = 'Actual'

N_RUNS = 10
SEED = 42
EPOCHS = 600           
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-5
HIDDEN_DIM = 48
DROPOUT = 0.3
PREDICTION_THRESHOLD = 0.5  
PRINT_EVERY = 50
SAVE_MODELS = False      

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_labeled_csv(path, source):
    data = pd.read_csv(path)
    missing = {SMILES_COL, LABEL_COL}.difference(data.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    data = data[[SMILES_COL, LABEL_COL]].copy()
    if data.isna().any().any() or (data[SMILES_COL].astype(str).str.strip() == '').any():
        raise ValueError(f'{path}: missing/blank SMILES or Actual values')
    data[LABEL_COL] = data[LABEL_COL].replace({
        'Non-Inhibitor': 0, 'Inhibitor': 1, '0': 0, '1': 1,
    })
    data[LABEL_COL] = pd.to_numeric(data[LABEL_COL], errors='raise').astype(int)
    if not data[LABEL_COL].isin([0, 1]).all():
        raise ValueError(f'{path}: Actual must be binary 0/1')
    data['Source'] = source
    return data.reset_index(drop=True)

def graph_from_smiles(smiles, label):
    g = smiles_to_graph(str(smiles), float(label))
    if g is None or g.x is None or g.edge_attr is None:
        raise ValueError(f'Cannot convert SMILES to graph: {smiles}')
    if g.x.ndim != 2 or g.edge_attr.ndim != 2:
        raise ValueError(f'Invalid graph node/edge feature shape for: {smiles}')
    return g

def prepare_graphs(raw_train_graphs, raw_test_graphs, randomized_labels):
    # Normalization statistics are learned exclusively on TRAIN+VALIDATION.
    x = torch.cat([g.x.float() for g in raw_train_graphs], dim=0)
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    std = torch.where((std == 0) | (~torch.isfinite(std)), torch.ones_like(std), std)

    train_graphs = []
    for g, y in zip(raw_train_graphs, randomized_labels):
        h = g.clone()                     # Never modify shared raw graphs
        h.x = (g.x.float() - mean) / std
        h.y = torch.tensor([float(y)], dtype=torch.float32)
        train_graphs.append(h)

    test_graphs = []
    for g in raw_test_graphs:
        h = g.clone()
        h.x = (g.x.float() - mean) / std
        # Preserve actual test label from input graph
        test_graphs.append(h)
    return train_graphs, test_graphs, mean, std

def train_randomized_model(model, graphs, shuffled_labels, run, device):
    neg = int((shuffled_labels == 0).sum())
    pos = int((shuffled_labels == 1).sum())
    if pos == 0 or neg == 0:
        raise ValueError('Combined training data must contain both classes')

    # Same TWO imbalance treatments as original training_advanced.py
    positive_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    sampling_weights = np.where(shuffled_labels == 1, neg / pos, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sampling_weights, dtype=torch.double),
        num_samples=len(graphs), replacement=True,
        generator=torch.Generator().manual_seed(SEED + run),
    )
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, sampler=sampler)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits.view(-1), batch.y.view(-1).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_loss = float(np.mean(epoch_losses))
        losses.append({'Run': run, 'Epoch': epoch, 'Train_Loss': mean_loss})
        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == EPOCHS:
            print(f'Run {run:02d}/{N_RUNS} | Epoch {epoch:03d}/{EPOCHS} | '
                  f'loss={mean_loss:.5f}', flush=True)
    return losses


@torch.no_grad()
def predict_test(model, test_graphs, device):
    model.eval()
    probability = []
    for batch in DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False):
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probability.extend(torch.sigmoid(logits).view(-1).cpu().tolist())
    prob = np.asarray(probability, dtype=float)
    predicted = (prob > PREDICTION_THRESHOLD).astype(int)
    return predicted, prob


def summarize_run(run, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # Class 1 (inhibitors) only. No weighted averaging requested here.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    return {
        'Run': run,
        'N_Train': None,  # populated in main
        'N_Test': len(y_true),
        'Test_Accuracy': float(accuracy_score(y_true, y_pred)),
        'Class1_Precision': float(precision[0]),
        'Class1_Recall': float(recall[0]),
        'Class1_F1': float(f1[0]),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    set_seed(SEED)

    train = read_labeled_csv(TRAIN_FILE, 'Train')
    validation = read_labeled_csv(VALIDATION_FILE, 'Validation')
    test = read_labeled_csv(TEST_FILE, 'Test')
    train_val = pd.concat([train, validation], ignore_index=True)
    original_y = train_val[LABEL_COL].to_numpy(dtype=int)
    test_y = test[LABEL_COL].to_numpy(dtype=int)
    if len(np.unique(original_y)) != 2:
        raise ValueError('Train + validation must contain both classes')
    train_val.to_csv(OUTPUT_DIR / 'original_train_validation.csv', index=False)

    print('Converting SMILES into molecular graphs (original graph converter)...')
    raw_training = [graph_from_smiles(s, y) for s, y in
                    zip(train_val[SMILES_COL], original_y)]
    raw_test = [graph_from_smiles(s, y) for s, y in
                zip(test[SMILES_COL], test_y)]
    in_channels = raw_training[0].x.size(1)
    edge_dim = raw_training[0].edge_attr.size(1)
    for g in raw_training + raw_test:
        if g.x.size(1) != in_channels or g.edge_attr.size(1) != edge_dim:
            raise ValueError('Graph input dimensions differ between molecules')
    all_results, all_predictions, all_histories = [], [], []

    permutations = []
    for run in range(1, N_RUNS + 1):
        rng = np.random.default_rng(SEED + run)
        randomized_y = rng.permutation(original_y)
        permutations.append(randomized_y)
        
        set_seed(SEED + run)
        run_dir = OUTPUT_DIR / f'run_{run:02d}'
        run_dir.mkdir(exist_ok=True)

        labels_table = train_val[[SMILES_COL, LABEL_COL, 'Source']].copy()
        labels_table.insert(0, 'Original_Row', np.arange(len(train_val)))
        labels_table['Randomized_Label'] = randomized_y
        labels_table.to_csv(run_dir / 'randomized_training_labels.csv', index=False)

        train_graphs, test_graphs, mean, std = prepare_graphs(
            raw_training, raw_test, randomized_y
        )
        model = EdgeAwareGCNNet(
            in_channels=in_channels, edge_dim=edge_dim,
            hidden_dim=HIDDEN_DIM, dropout=DROPOUT
        ).to(device)
        history = train_randomized_model(model, train_graphs, randomized_y, run, device)
        all_histories.extend(history)

        predicted, probabilities = predict_test(model, test_graphs, device)
        if len(predicted) != len(test_y):
            raise RuntimeError('Prediction count does not match')
        row = summarize_run(run, test_y, predicted)
        row['N_Train'] = len(train_val)
        all_results.append(row)
        pd.DataFrame([row]).to_csv(run_dir / 'test_metrics.csv', index=False)

        pred_table = test[[SMILES_COL, LABEL_COL]].copy()
        pred_table.insert(0, 'Run', run)
        pred_table['Predicted'] = predicted
        pred_table['Prob_Class_1'] = probabilities
        pred_table.to_csv(run_dir / 'test_predictions.csv', index=False)
        all_predictions.append(pred_table)
        print('Test metrics:', {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in row.items()})
        if SAVE_MODELS:
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_mean': mean.cpu(), 'feature_std': std.cpu(),
                'in_channels': in_channels, 'edge_dim': edge_dim,
                'hidden_dim': HIDDEN_DIM, 'dropout': DROPOUT,
                'run': run, 'seed': SEED + run,
            }, run_dir / 'randomized_model.pt')

    results = pd.DataFrame(all_results)
    results.to_csv(OUTPUT_DIR / 'Y_randomization_10_runs_metrics.csv',
                   index=False, float_format='%.6f')
    pd.concat(all_predictions, ignore_index=True).to_csv(
        OUTPUT_DIR / 'Y_randomization_test_predictions.csv',
        index=False, float_format='%.6f'
    )
    pd.DataFrame(all_histories).to_csv(
        OUTPUT_DIR / 'Y_randomization_training_history.csv',
        index=False, float_format='%.6f'
    )
    perm_df = pd.DataFrame({'Original_Row': np.arange(len(original_y)),
                            'Original_Label': original_y})
    for run, shuffled in enumerate(permutations, start=1):
        perm_df[f'Run{run}_Randomized_Label'] = shuffled
    perm_df.to_csv(OUTPUT_DIR / 'Y_randomization_label_permutations.csv', index=False)

    metric_cols = ['Test_Accuracy', 'Class1_Precision', 'Class1_Recall', 'Class1_F1']
    summary = pd.DataFrame([
        {'Metric': metric,
         'Mean': results[metric].mean(),
         'SD': results[metric].std(ddof=1),
         'Mean ± SD': f'{results[metric].mean():.4f} ± {results[metric].std(ddof=1):.4f}'}
        for metric in metric_cols
    ])
    summary.to_csv(OUTPUT_DIR / 'Y_randomization_mean_sd.csv', index=False)
if __name__ == '__main__':
    main()
