
# Lab 02 - Avaliação de modelos de ML

# Sentimentos reais (ground truth)
y_true = [
    "POSITIVE",
    "NEGATIVE",
    "POSITIVE",
    "POSITIVE",
    "NEGATIVE",
    "NEGATIVE",
    "POSITIVE",
    "NEGATIVE",
    "POSITIVE",
    "NEGATIVE",
]

# Previsões do modelo
y_pred = [
    "POSITIVE",
    "NEGATIVE",
    "NEGATIVE",
    "POSITIVE",
    "POSITIVE",
    "NEGATIVE",
    "POSITIVE",
    "NEGATIVE",
    "POSITIVE",
    "POSITIVE",
]

print("ANÁLISE DAS PREVISÕES")
print("-" * 45)

for i, (real, previsto) in enumerate(zip(y_true, y_pred), start=1):
    resultado = "ACERTO" if real == previsto else "ERRO"
    print(f"{i:02d} | Real: {real:8} | Previsto: {previsto:8} | {resultado}")


# cria matriz de confusão

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Define a ordem das classes
classes = ["POSITIVE", "NEGATIVE"]

# Constrói a matriz de confusão
matriz = confusion_matrix(
    y_true,
    y_pred,
    labels=classes,
)

print("\nMATRIZ DE CONFUSÃO")
print(matriz)

# Calcula as métricas para a classe POSITIVE
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, pos_label="POSITIVE"
)

recall = recall_score(
    y_true, y_pred, pos_label="POSITIVE"
)

f1 = f1_score(
    y_true, y_pred, pos_label="POSITIVE"
)

print("\nMÉTRICAS")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-score:  {f1:.2%}")

print("\nRELATÓRIO DE CLASSIFICAÇÃO")
print(classification_report(
    y_true,
    y_pred,
    labels=classes,
    zero_division=0,
))