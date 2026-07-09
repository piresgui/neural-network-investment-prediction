# Neural Network — Bank Term Deposit Prediction

Rede neural implementada **do zero em NumPy** para prever se um cliente de banco irá realizar um depósito a prazo (term deposit), com base em seu perfil e histórico de contatos.

## Objetivo
Maximizar a **taxa de verdadeiros positivos (TP Rate / Recall)** — identificar o maior número possível de clientes que realmente irão investir.

## Dataset
- **Treino**: `data/dataset_bank/bank.csv` — 4.521 registros (10% do total)
- **Validação/Teste**: `data/dataset_bank/bank-full.csv` — 45.211 registros

Fonte: [UCI Bank Marketing Dataset](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## Arquitetura da Rede
```
Entrada (n features) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
```
- Saída: 1 neurônio com sigmoid → probabilidade de depósito
- Loss: Binary Cross-Entropy
- Otimizador: Mini-batch SGD + Momentum

## Pré-processamento
| Tipo de variável | Tratamento |
|---|---|
| Numéricas | Normalização min-max → [0, 1] |
| Binárias (yes/no) | Codificação 0/1 |
| Categóricas multi-classe | One-Hot Encoding |

## Como executar

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar e avaliar
python main.py
```

## Estrutura
```
├── data/dataset_bank/     # CSVs do dataset
├── src/
│   ├── preprocessing.py   # Pré-processamento
│   ├── model.py           # Rede neural (NumPy puro)
│   └── evaluate.py        # Métricas e gráficos
├── outputs/               # Gráficos gerados
├── references/            # Material de referência
└── main.py                # Pipeline completo
```
