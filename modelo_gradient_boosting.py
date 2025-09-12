from preprocesamiento import (
    cargar_y_seleccionar,
    definir_columnas,
    construir_preprocesador,
    hacer_split,
)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, classification_report, confusion_matrix, roc_auc_score
)
import numpy as np
import pandas as pd


def entrenar_gbm(X_train, y_train, pre):
    pipe = Pipeline(steps=[
        ("pre", pre),
        ("gb", GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        "gb__n_estimators": [100, 200],
        "gb__learning_rate": [0.01, 0.05, 0.1],
        "gb__max_depth": [2, 3, 4],
        "gb__subsample": [0.8, 1.0],
        "gb__min_samples_leaf": [1, 3, 5],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def evaluar(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)

    try:
        y_proba = modelo.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    print(f"\nMétricas GBM:\nAccuracy={acc:.3f}  Precision={prec:.3f}")
    if auc is not None:
        print(f"ROC-AUC={auc:.3f}")

    reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(reporte).T
    df = df[["precision"]]
    print("\nReporte de clasificación:\n", df)

    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))


def top_importancias(modelo, num_cols, cat_cols):
    onehot = modelo.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
    cat_feat = list(onehot.get_feature_names_out(cat_cols))
    feat_names = list(num_cols) + cat_feat

    importances = modelo.named_steps["gb"].feature_importances_
    orden = np.argsort(importances)[::-1]
    print("\nTop 15 características por importancia (GBM):")
    for idx in orden[:15]:
        print(f"{feat_names[idx]:>20s}: {importances[idx]:.4f}")


def main():
    X, y = cargar_y_seleccionar("data.csv")
    num_cols, cat_cols = definir_columnas(X)
    pre = construir_preprocesador(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = hacer_split(X, y, test_size=0.2, seed=42)
    modelo, params, cv_score = entrenar_gbm(X_train, y_train, pre)

    print("\nMejores hiperparámetros:", params)
    print("Mejor score CV (F1):", round(cv_score, 3))

    evaluar(modelo, X_test, y_test)
    top_importancias(modelo, num_cols, cat_cols)


if __name__ == "__main__":
    main()
