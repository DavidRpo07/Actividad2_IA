import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, ConfusionMatrixDisplay

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocesamiento import cargar_y_seleccionar, hacer_split, definir_columnas, construir_preprocesador

# 1) Cargar datos y hacer split en train/test
X, y = cargar_y_seleccionar("../data.csv")
X_train, X_test, y_train, y_test = hacer_split(X, y, test_size=0.2, seed=42)

# 2) Preprocesamiento
num_cols, cat_cols = definir_columnas(X)
pre = construir_preprocesador(num_cols, cat_cols)

# 3) Red Neuronal (MLPClassifier)
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

pipe_mlp = Pipeline([("pre", pre), ("mlp", mlp)])
pipe_mlp.fit(X_train, y_train)

# 4) Hacer predicciones
y_pred = pipe_mlp.predict(X_test)

# Métricas de rendimiento
acc  = accuracy_score(y_test, y_pred)
prec_survived    = precision_score(y_test, y_pred, pos_label=1)
prec_not_survive = precision_score(y_test, y_pred, pos_label=0)

print("Métricas de rendimiento Red Neuronal (MLP):")
print(f"Accuracy : {acc:.4f}")
print(f"Precision (Sobrevivió = 1):    {prec_survived:.4f}")
print(f"Precision (No sobrevivió = 0): {prec_not_survive:.4f}")

# Matriz de confusión
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Purples")
plt.title("Matriz de confusión Red Neuronal")
plt.grid(False)
plt.savefig("matrizConf_mlp.png", dpi=150)
plt.close()

print("\nSe guardó: matrizConf_mlp.png")
