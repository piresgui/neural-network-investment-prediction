"""
Pipeline principal: pré-processamento → treinamento → avaliação.

Treino  : data/dataset_bank/bank.csv       (10% dos dados)
Validação: data/dataset_bank/bank-full.csv  (dataset completo)
"""

import os
import sys
import numpy as np

# Permite importar src/ mesmo rodando do diretório raiz
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import load_data, preprocess, preprocess_validation
from model import NeuralNetwork
from evaluate import (
    compute_metrics, print_report,
    plot_training_history, find_best_threshold,
)

# ─── Configurações ───────────────────────────────────────────────────────────
TRAIN_PATH = "data/dataset_bank/bank.csv"
VAL_PATH   = "data/dataset_bank/bank-full.csv"
OUTPUT_DIR = "outputs"

# Hiperparâmetros
HIDDEN_LAYERS = [64, 32]   # neurônios por camada oculta
LEARNING_RATE = 0.005
MOMENTUM      = 0.9
EPOCHS        = 200
BATCH_SIZE    = 32
SEED          = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # ── 1. Carregar dados ────────────────────────────────────────────────────
    print("Carregando dados...")
    df_train = load_data(TRAIN_PATH)
    df_val   = load_data(VAL_PATH)
    print(f"  Treino  : {len(df_train)} amostras")
    print(f"  Validação: {len(df_val)} amostras")

    # ── 2. Pré-processar ─────────────────────────────────────────────────────
    print("\nPré-processando...")
    X_train, y_train, fit_params, train_cols = preprocess(df_train)
    X_val, y_val = preprocess_validation(df_val, fit_params, train_cols)

    n_features = X_train.shape[1]
    print(f"  Features após encoding: {n_features}")
    print(f"  Classe positiva (treino): {y_train.sum():.0f}/{len(y_train)} "
          f"({y_train.mean()*100:.1f}%)")
    print(f"  Classe positiva (validação): {y_val.sum():.0f}/{len(y_val)} "
          f"({y_val.mean()*100:.1f}%)")

    # ── 3. Montar rede ───────────────────────────────────────────────────────
    layer_sizes = [n_features] + HIDDEN_LAYERS + [1]
    print(f"\nArquitetura da rede: {layer_sizes}")

    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        seed=SEED,
    )

    # ── 4. Treinar ───────────────────────────────────────────────────────────
    print(f"\nTreinando por {EPOCHS} épocas (batch={BATCH_SIZE})...\n")
    history = nn.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        X_val=X_val,
        y_val=y_val,
        verbose=True,
    )

    # ── 5. Avaliar com threshold padrão (0.5) ───────────────────────────────
    print("\n--- Avaliação com threshold=0.5 ---")
    y_proba = nn.predict_proba(X_val)
    y_pred  = (y_proba >= 0.5).astype(int)
    metrics_50 = compute_metrics(y_val, y_pred, y_proba)
    print_report(metrics_50, "Validação (threshold=0.50)")

    # ── 6. Buscar threshold ótimo para TP Rate (recall) ─────────────────────
    best_thresh, best_recall = find_best_threshold(y_val, y_proba, metric="recall")
    print(f"\nMelhor threshold para maximizar TP Rate: {best_thresh:.2f} → recall={best_recall:.4f}")

    y_pred_opt = (y_proba >= best_thresh).astype(int)
    metrics_opt = compute_metrics(y_val, y_pred_opt)
    print_report(metrics_opt, f"Validação (threshold={best_thresh:.2f} — máx TP Rate)")

    # Threshold ótimo por F1
    best_thresh_f1, best_f1 = find_best_threshold(y_val, y_proba, metric="f1")
    print(f"\nMelhor threshold para maximizar F1: {best_thresh_f1:.2f} → F1={best_f1:.4f}")
    y_pred_f1 = (y_proba >= best_thresh_f1).astype(int)
    metrics_f1 = compute_metrics(y_val, y_pred_f1)
    print_report(metrics_f1, f"Validação (threshold={best_thresh_f1:.2f} — máx F1)")

    # ── 7. Salvar gráfico de treinamento ─────────────────────────────────────
    plot_path = os.path.join(OUTPUT_DIR, "training_history.png")
    plot_training_history(history, plot_path)

    print("\nConcluído!")


if __name__ == "__main__":
    main()
