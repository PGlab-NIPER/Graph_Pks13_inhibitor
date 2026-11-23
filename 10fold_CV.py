#10-fold stratified CV for the EdgeAwareGCNNet model.

from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,roc_auc_score, average_precision_score,)
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from training_advanced import EdgeAwareGCNNet
from Smiles_to_graph import smiles_to_graph

DATA_CSV = Path('train_validation.csv')
OUTPUT_DIR = Path('GNN_10Fold_CV_Results')
SMILES_COL = 'SMILES'
LABEL_COL = 'Actual'
N_SPLITS = 10
SEED = 42
EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-5
HIDDEN_DIM = 48
DROPOUT = 0.3
SAVE_FOLD_MODELS = False       
PRINT_EVERY = 50              

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data():
    df = pd.read_csv(DATA_CSV)
    missing = {SMILES_COL, LABEL_COL} - set(df.columns)
    df = df[[SMILES_COL, LABEL_COL]].copy()
    df["Source"] = "TData"
    df[LABEL_COL] = df[LABEL_COL].replace({
        "Non-Inhibitor": 0,
        "Inhibitor": 1,
        "0": 0,
        "1": 1
    })

    df[LABEL_COL] = pd.to_numeric(
        df[LABEL_COL],
        errors="raise"
    ).astype(int)

    if not df[LABEL_COL].isin([0, 1]).all():
        raise ValueError(
            "Actual must contain only Class 0 / Class 1."
        )

    class_counts = df[LABEL_COL].value_counts().reindex(
        [0, 1],
        fill_value=0
    )

    if class_counts.min() < N_SPLITS:
        raise ValueError("Error")
    duplicated = df[SMILES_COL].duplicated(
        keep=False
    ).sum()

    if duplicated:
        print("WARNING")
    return df


def as_graph(smiles, label):
    graph = smiles_to_graph(str(smiles), float(label))
    if graph is None or graph.x is None or graph.edge_attr is None:
        raise ValueError(f'Unable to construct graph for SMILES: {smiles}')
    if graph.x.dim() != 2 or graph.edge_attr.dim() != 2:
        raise ValueError(f'Malformed graph features for SMILES: {smiles}')
    return graph


def prepare_fold_graphs(raw_graphs, train_idx, val_idx):
    raw_train_x = torch.cat([raw_graphs[i].x.float() for i in train_idx], dim=0)
    mean = raw_train_x.mean(dim=0)
    std = raw_train_x.std(dim=0)
    std = torch.where((std == 0) | (~torch.isfinite(std)), torch.ones_like(std), std)

    def transformed(indices):
        graphs = []
        for i in indices:
            g = raw_graphs[int(i)].clone()
            g.x = (g.x.float() - mean) / std
            graphs.append(g)
        return graphs

    return transformed(train_idx), transformed(val_idx), mean, std


def train_one_fold(model, train_graphs, labels_train, device, fold):
    neg = int((labels_train == 0).sum())
    pos = int((labels_train == 1).sum())
    if pos == 0 or neg == 0:
        raise ValueError(f'Fold {fold}: one class absent in training partition.')
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    weights = [neg / pos if int(y) == 1 else 1.0 for y in labels_train]
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True,
        generator=torch.Generator().manual_seed(SEED + fold))
    loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits.view(-1), batch.y.view(-1).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
        epoch_loss = float(np.mean(batch_losses))
        history.append({'Fold': fold, 'Epoch': epoch, 'Train_Loss': epoch_loss})
        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == EPOCHS:
            print(f'Fold {fold:02d} | Epoch {epoch:03d}/{EPOCHS} | '
                  f'train loss {epoch_loss:.5f}', flush=True)
    return history


@torch.no_grad()
def evaluate(model, val_graphs, device):
    model.eval()
    true, prob = [], []
    loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        p = torch.sigmoid(logits).view(-1)
        prob.extend(p.cpu().numpy().tolist())
        true.extend(batch.y.view(-1).cpu().numpy().astype(int).tolist())
    y_true = np.asarray(true, dtype=int)
    y_prob = np.asarray(prob, dtype=float)
    y_pred = (y_prob > 0.5).astype(int)  # matches original code exactly
    return y_true, y_pred, y_prob


def score_fold(fold, y_true, y_pred, y_prob, n_train):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average='weighted', zero_division=0)
    if len(np.unique(y_true)) == 2:
        roc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
    else:
        roc, ap = np.nan, np.nan
    return {
        'Fold': fold, 'N_Train': n_train, 'N_Heldout': len(y_true),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Weighted_Precision': p_w,
        'Weighted_Recall': r_w,
        'Weighted_F1': f_w,
        'AU_ROC': roc, 'PR_AUC_AP': ap,
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),}


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    df = load_data()
    df.to_csv(OUTPUT_DIR / 'combined_train_validation.csv', index=False)
    labels = df[LABEL_COL].to_numpy(dtype=int)
    raw_graphs = [as_graph(s, y) for s, y in
                  zip(df[SMILES_COL].tolist(), labels.tolist())]
    in_channels = int(raw_graphs[0].x.size(1))
    edge_dim = int(raw_graphs[0].edge_attr.size(1))
    for i, g in enumerate(raw_graphs):
        if g.x.size(1) != in_channels or g.edge_attr.size(1) != edge_dim:
            raise ValueError(f'Inconsistent graph feature dimension at row {i}')
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    all_scores, all_classwise, all_predictions, all_histories = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(labels)), labels), 1):
        print(f'\n========== FOLD {fold} / {N_SPLITS} ==========', flush=True)
        set_seed(SEED + fold)  
        fold_dir = OUTPUT_DIR / f'fold_{fold:02d}'
        fold_dir.mkdir(exist_ok=True)
        train_graphs, val_graphs, mean, std = prepare_fold_graphs(raw_graphs, train_idx, val_idx)
        model = EdgeAwareGCNNet(
            in_channels=in_channels, edge_dim=edge_dim,
            hidden_dim=HIDDEN_DIM, dropout=DROPOUT
        ).to(device)

        all_histories.extend(train_one_fold(
            model, train_graphs, labels[train_idx], device, fold))
        y_true, y_pred, y_prob = evaluate(model, val_graphs, device)
        if not np.array_equal(y_true, labels[val_idx]):
            raise RuntimeError('Prediction order and held-out labels do not match.')
        result = score_fold(fold, y_true, y_pred, y_prob, len(train_idx))
        all_scores.append(result)
        pd.DataFrame([result]).to_csv(fold_dir / 'heldout_metrics.csv', index=False)
        print('Held-out metrics:', result)

        p, r, f, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
        for j, cls in enumerate([0, 1]):
            row = {'Fold': fold, 'Class': cls,
                   'Class_Name': 'Non-Inhibitor' if cls == 0 else 'Inhibitor',
                   'Support': int(support[j]), 'Precision': p[j],
                   'Recall': r[j], 'F1_Score': f[j]}
            all_classwise.append(row)
        pd.DataFrame(all_classwise[-2:]).to_csv(fold_dir / 'classwise_metrics.csv', index=False)

        pred_table = df.iloc[val_idx][[SMILES_COL, LABEL_COL, 'Source']].copy()
        pred_table.insert(0, 'Original_Row', val_idx)
        pred_table.insert(1, 'Fold', fold)
        pred_table['Predicted'] = y_pred
        pred_table['Prob_Class_1'] = y_prob
        pred_table.to_csv(fold_dir / 'heldout_predictions.csv', index=False)
        all_predictions.append(pred_table)

        if SAVE_FOLD_MODELS:
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_mean': mean.cpu(), 'feature_std': std.cpu(),
                'in_channels': in_channels, 'edge_dim': edge_dim,
                'hidden_dim': HIDDEN_DIM, 'dropout': DROPOUT,
                'fold': fold, 'train_indices': train_idx,
                'heldout_indices': val_idx
            }, fold_dir / 'model_checkpoint.pt')
        del model, train_graphs, val_graphs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores = pd.DataFrame(all_scores)
    classwise = pd.DataFrame(all_classwise)
    oof = pd.concat(all_predictions, ignore_index=True).sort_values('Original_Row')
    if len(oof) != len(df) or oof['Original_Row'].duplicated().any():
        raise RuntimeError('Out-of-fold predictions must cover every input row once.')

    score_cols = ['Accuracy', 'Weighted_Precision', 'Weighted_Recall',
                  'Weighted_F1', 'AU_ROC', 'PR_AUC_AP']
    summary = pd.DataFrame({
        'Metric': score_cols,
        'Mean': [scores[c].mean() for c in score_cols],
        'SD': [scores[c].std(ddof=1) for c in score_cols],
    })
    summary['Mean ± SD'] = summary.apply(
        lambda r: f"{r['Mean']:.4f} ± {r['SD']:.4f}", axis=1)

    classwise_summary = classwise.groupby('Class', as_index=False).agg(
        Precision_Mean=('Precision', 'mean'), Precision_SD=('Precision', 'std'),
        Recall_Mean=('Recall', 'mean'), Recall_SD=('Recall', 'std'),
        F1_Mean=('F1_Score', 'mean'), F1_SD=('F1_Score', 'std')
    )
    for base in ('Precision', 'Recall', 'F1'):
        classwise_summary[f'{base} Mean ± SD'] = classwise_summary.apply(
            lambda r: f"{r[f'{base}_Mean']:.4f} ± {r[f'{base}_SD']:.4f}", axis=1)

    # Publication-style one-row table with the same weighted metrics as before.
    pub = pd.DataFrame([{'Dataset': 'Train+Validation (10-fold CV)', **{
        name: summary.set_index('Metric').loc[col, 'Mean ± SD'] for name, col in (
            ('Acc', 'Accuracy'), ('Weighted-Pr', 'Weighted_Precision'),
            ('Weighted-Re', 'Weighted_Recall'), ('Weighted-F1', 'Weighted_F1'),
            ('AU-ROC', 'AU_ROC'), ('PR-AUC (AP)', 'PR_AUC_AP'))
    }}])

    scores.to_csv(OUTPUT_DIR / '10fold_heldout_metrics.csv', index=False, float_format='%.6f')
    summary.to_csv(OUTPUT_DIR / '10fold_mean_sd.csv', index=False)
    classwise.to_csv(OUTPUT_DIR / '10fold_classwise_metrics.csv', index=False, float_format='%.6f')
    classwise_summary.to_csv(OUTPUT_DIR / '10fold_classwise_mean_sd.csv', index=False)
    oof.to_csv(OUTPUT_DIR / '10fold_out_of_fold_predictions.csv', index=False, float_format='%.6f')
    pd.DataFrame(all_histories).to_csv(OUTPUT_DIR / '10fold_training_loss.csv', index=False)
    pub.to_csv(OUTPUT_DIR / 'publication_ready_table.csv', index=False)

    print('\n=== Individual held-out fold results ===')
    print(scores.to_string(index=False, float_format=lambda n: f'{n:.4f}'))
    print('\n=== 10-fold mean ± SD ===')
    print(summary[['Metric', 'Mean ± SD']].to_string(index=False))
    print('\nSaved output files to:', OUTPUT_DIR.resolve())
if __name__ == '__main__':
    main()
