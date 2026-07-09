"""
Pré-processamento do dataset Bank Marketing.
- Normalização de variáveis numéricas (min-max para [0,1])
- Codificação binária para variáveis binárias
- One-Hot Encoding para variáveis categóricas multi-classe
"""

import numpy as np
import pandas as pd


# Colunas categóricas binárias: "yes"/"no" → 1/0
BINARY_COLS = ["default", "housing", "loan"]

# Colunas categóricas multi-classe → One-Hot Encoding
CATEGORICAL_COLS = ["job", "marital", "education", "contact", "month", "poutcome"]

# Colunas numéricas → normalização min-max
NUMERIC_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

TARGET_COL = "y"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def preprocess(df: pd.DataFrame, fit_params: dict = None):
    """
    Transforma o DataFrame em features numéricas prontas para a rede neural.

    Parâmetros
    ----------
    df         : DataFrame bruto
    fit_params : dict com min/max de cada coluna numérica (calculado no treino).
                 Se None, calcula a partir de df (modo treino).

    Retorna
    -------
    X          : np.ndarray (n_samples, n_features)
    y          : np.ndarray (n_samples,) com 0 ou 1
    fit_params : dict a ser passado ao processar dados de validação/teste
    """
    df = df.copy()

    # --- Target ---
    y = (df[TARGET_COL].str.strip().str.lower() == "yes").astype(np.float64).values

    # --- Binárias ---
    for col in BINARY_COLS:
        df[col] = (df[col].str.strip().str.lower() == "yes").astype(np.float64)

    # --- One-Hot ---
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # --- Numéricas: min-max ---
    if fit_params is None:
        fit_params = {}
        for col in NUMERIC_COLS:
            col_min = df[col].min()
            col_max = df[col].max()
            fit_params[col] = (col_min, col_max)

    for col in NUMERIC_COLS:
        col_min, col_max = fit_params[col]
        rng = col_max - col_min if col_max != col_min else 1.0
        df[col] = (df[col] - col_min) / rng

    # Remove coluna alvo antes de montar X
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values.astype(np.float64)

    return X, y, fit_params, feature_cols


def align_columns(X_df: pd.DataFrame, train_cols: list) -> np.ndarray:
    """
    Garante que o conjunto de validação tenha as mesmas colunas do treino,
    preenchendo com 0 colunas ausentes e descartando as excedentes.
    """
    for col in train_cols:
        if col not in X_df.columns:
            X_df[col] = 0.0
    return X_df[train_cols].values.astype(np.float64)


def preprocess_validation(df: pd.DataFrame, fit_params: dict, train_cols: list):
    """Aplica o mesmo pré-processamento do treino ao conjunto de validação."""
    df = df.copy()
    y = (df[TARGET_COL].str.strip().str.lower() == "yes").astype(np.float64).values

    for col in BINARY_COLS:
        df[col] = (df[col].str.strip().str.lower() == "yes").astype(np.float64)

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    for col in NUMERIC_COLS:
        col_min, col_max = fit_params[col]
        rng = col_max - col_min if col_max != col_min else 1.0
        df[col] = (df[col] - col_min) / rng

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X_df = df[feature_cols]
    X = align_columns(X_df, train_cols)

    return X, y
