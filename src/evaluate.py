"""
Funções de avaliação e geração de relatório de desempenho.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # sem GUI, salva em arquivo


def compute_metrics(y_true, y_pred, y_proba=None):
    """Retorna dict com as principais métricas de classificação binária."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TP rate / sensibilidade
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall_tp_rate": recall,
        "specificity": specificity,
        "f1_score": f1,
    }


def print_report(metrics: dict, title: str = "Relatório de Avaliação"):
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Acurácia Geral        : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precisão              : {metrics['precision']:.4f}")
    print(f"  Recall (TP Rate)      : {metrics['recall_tp_rate']:.4f}  ← foco principal")
    print(f"  Especificidade        : {metrics['specificity']:.4f}")
    print(f"  F1-Score              : {metrics['f1_score']:.4f}")
    print(f"\n  Matriz de Confusão:")
    print(f"             Previsto +   Previsto -")
    print(f"  Real +  :    {metrics['TP']:6d}       {metrics['FN']:6d}")
    print(f"  Real -  :    {metrics['FP']:6d}       {metrics['TN']:6d}")
    print(sep)


def plot_training_history(history: dict, output_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Treino")
    if history["val_loss"]:
        axes[0].plot(history["val_loss"], label="Validação")
    axes[0].set_title("Loss (Binary Cross-Entropy)")
    axes[0].set_xlabel("Época")
    axes[0].legend()
    axes[0].grid(True)

    if history["val_acc"]:
        axes[1].plot(history["val_acc"], color="green")
        axes[1].set_title("Acurácia na Validação")
        axes[1].set_xlabel("Época")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True)

    if history["val_tp_rate"]:
        axes[2].plot(history["val_tp_rate"], color="red")
        axes[2].set_title("Taxa de Verdadeiros Positivos (Recall)")
        axes[2].set_xlabel("Época")
        axes[2].set_ylim(0, 1)
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nGráfico salvo em: {output_path}")


def find_best_threshold(y_true, y_proba, metric: str = "f1"):
    """
    Varre thresholds de 0.1 a 0.9 e retorna o que maximiza a métrica escolhida.
    metric: 'f1' | 'recall' | 'precision'
    """
    best_thresh = 0.5
    best_score = 0.0

    for thresh in np.arange(0.1, 0.91, 0.01):
        y_pred = (y_proba >= thresh).astype(int)
        m = compute_metrics(y_true, y_pred)
        score = m[{"f1": "f1_score", "recall": "recall_tp_rate",
                    "precision": "precision"}[metric]]
        if score > best_score:
            best_score = score
            best_thresh = thresh

    return round(float(best_thresh), 2), best_score
