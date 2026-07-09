"""
Rede Neural Multicamadas implementada em NumPy puro.

Arquitetura:
  Entrada → [Camadas Ocultas com ReLU] → Saída com Sigmoid (1 neurônio binário)

Gradiente via backpropagation com mini-batch SGD + momentum.
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_deriv(a):
    return a * (1.0 - a)


def relu(z):
    return np.maximum(0.0, z)


def relu_deriv(z):
    return (z > 0).astype(np.float64)


class NeuralNetwork:
    """
    Rede neural densa com camadas ocultas ReLU e saída sigmoid.

    Parâmetros
    ----------
    layer_sizes : list[int]  – tamanho de cada camada, incluindo entrada e saída.
                              Ex.: [48, 64, 32, 1]
    lr          : float      – taxa de aprendizado
    momentum    : float      – fator de momentum (0 = sem momentum)
    seed        : int        – semente para reprodutibilidade
    """

    def __init__(self, layer_sizes: list, lr: float = 0.01,
                 momentum: float = 0.9, seed: int = 42):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.momentum = momentum
        self.n_layers = len(layer_sizes) - 1  # número de camadas de pesos

        rng = np.random.default_rng(seed)

        # Inicialização He (boa para ReLU)
        self.W = []
        self.b = []
        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            std = np.sqrt(2.0 / fan_in)
            self.W.append(rng.normal(0, std, (layer_sizes[i], layer_sizes[i + 1])))
            self.b.append(np.zeros((1, layer_sizes[i + 1])))

        # Velocidades para momentum
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    def _forward(self, X):
        """Retorna lista de ativações por camada (incluindo entrada)."""
        activations = [X]
        z_values = []
        for i in range(self.n_layers):
            z = activations[-1] @ self.W[i] + self.b[i]
            z_values.append(z)
            if i < self.n_layers - 1:
                a = relu(z)
            else:
                a = sigmoid(z)  # saída binária
            activations.append(a)
        return activations, z_values

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidade de classe positiva."""
        activations, _ = self._forward(X)
        return activations[-1].ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    def _binary_cross_entropy(self, y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def train_step(self, X_batch, y_batch):
        """Executa forward + backward em um mini-batch. Retorna loss."""
        n = X_batch.shape[0]
        activations, z_values = self._forward(X_batch)
        y_pred = activations[-1]

        loss = self._binary_cross_entropy(y_batch.reshape(-1, 1), y_pred)

        # Backpropagation
        # Gradiente da saída (BCE + sigmoid combinados = y_pred - y_true)
        delta = (y_pred - y_batch.reshape(-1, 1)) / n

        dW_list = []
        db_list = []

        for i in reversed(range(self.n_layers)):
            dW = activations[i].T @ delta
            db = delta.sum(axis=0, keepdims=True)
            dW_list.insert(0, dW)
            db_list.insert(0, db)

            if i > 0:
                delta = (delta @ self.W[i].T) * relu_deriv(z_values[i - 1])

        # Atualização com momentum
        for i in range(self.n_layers):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * dW_list[i]
            self.vb[i] = self.momentum * self.vb[i] - self.lr * db_list[i]
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]

        return float(loss)

    def fit(self, X_train, y_train, epochs: int = 100, batch_size: int = 32,
            X_val=None, y_val=None, verbose: bool = True):
        """
        Treina a rede e retorna histórico de métricas por época.
        """
        n = X_train.shape[0]
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_tp_rate": []}
        rng = np.random.default_rng(0)

        for epoch in range(1, epochs + 1):
            idx = rng.permutation(n)
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            epoch_loss = 0.0
            steps = 0
            for start in range(0, n, batch_size):
                end = start + batch_size
                loss = self.train_step(X_shuf[start:end], y_shuf[start:end])
                epoch_loss += loss
                steps += 1

            history["train_loss"].append(epoch_loss / steps)

            if X_val is not None:
                val_proba = self.predict_proba(X_val)
                val_loss = self._binary_cross_entropy(y_val, val_proba)
                val_pred = (val_proba >= 0.5).astype(int)
                val_acc = np.mean(val_pred == y_val)

                # Taxa de verdadeiros positivos (recall da classe positiva)
                positives = y_val == 1
                tp_rate = np.mean(val_pred[positives] == 1) if positives.any() else 0.0

                history["val_loss"].append(float(val_loss))
                history["val_acc"].append(float(val_acc))
                history["val_tp_rate"].append(float(tp_rate))

                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:4d}/{epochs} | "
                          f"train_loss={epoch_loss/steps:.4f} | "
                          f"val_loss={val_loss:.4f} | "
                          f"val_acc={val_acc:.4f} | "
                          f"TP_rate={tp_rate:.4f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:4d}/{epochs} | train_loss={epoch_loss/steps:.4f}")

        return history
