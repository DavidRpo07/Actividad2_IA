import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, ConfusionMatrixDisplay 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocesamiento import cargar_y_seleccionar, hacer_split, definir_columnas, construir_preprocesador

# 1) Cargar datos y hacer split en train/test
X, y = cargar_y_seleccionar("../data.csv") 
X_train, X_test, y_train, y_test = hacer_split(X, y, test_size=0.2, seed=42)

# 2) Preprocesamiento
num_cols, cat_cols = definir_columnas(X)
pre = construir_preprocesador(num_cols, cat_cols)

# 3) Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

pipe_rf = Pipeline([("pre", pre), ("rf", rf)])
pipe_rf.fit(X_train, y_train)

# 4) Hacer predicciones
y_pred = pipe_rf.predict(X_test)

# Métricas de rendimiento
acc  = accuracy_score(y_test, y_pred)
prec_survived    = precision_score(y_test, y_pred, pos_label=1)  
prec_not_survive = precision_score(y_test, y_pred, pos_label=0) 

print("Métricas de rendimiento Random Forest:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision (Sobrevivió = 1):    {prec_survived:.4f}")
print(f"Precision (No sobrevivió = 0): {prec_not_survive:.4f}")

# Matriz de confusión
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title("Matriz de confusión Random Forest")
plt.grid(False)
plt.savefig("matrizConf_rf.png", dpi=150)
plt.close()

# Importancia de características
rf_fit = pipe_rf.named_steps["rf"]
feat_names = pipe_rf.named_steps["pre"].get_feature_names_out()
importancias = rf_fit.feature_importances_
orden = np.argsort(importancias)[::-1]

top_k = 12
print("\nTop features por importancia (RandomForest):")
for idx in orden[:top_k]:
    print(f"{feat_names[idx]}: {importancias[idx]:.4f}")

# Gráfico de importancias
plt.figure()
plt.bar(range(min(top_k, len(feat_names))), importancias[orden][:top_k])
plt.xticks(range(min(top_k, len(feat_names))), feat_names[orden][:top_k], rotation=90)
plt.title("Feature Importances RandomForest")
plt.tight_layout()
plt.savefig("feature_importances_rf.png", dpi=150)
plt.close()

print("\nSe guardaron: matrizConf_rf.png y feature_importances_rf.png")