import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.pipeline import Pipeline


import preprocesamiento as prep


def _clean_conflict_lines(text: str) -> str:
    bad_prefixes = ("<<<<<<<", "=======", ">>>>>>>")
    return "\n".join(line for line in text.splitlines() if not line.startswith(bad_prefixes))

def _patch_read_csv():
    """Parchea pd.read_csv SOLO dentro de preprocesamiento para ignorar marcadores de conflicto.
    No cambia datos ni nombres de columnas. No escribe archivos."""
    orig_read_csv = pd.read_csv
    def patched_read_csv(filepath_or_buffer, *args, **kwargs):
        if isinstance(filepath_or_buffer, (str, Path)) and Path(filepath_or_buffer).exists():
            raw = Path(filepath_or_buffer).read_text(encoding="utf-8", errors="replace")
            filtered = _clean_conflict_lines(raw)
            return orig_read_csv(io.StringIO(filtered), *args, **kwargs)
        return orig_read_csv(filepath_or_buffer, *args, **kwargs)
    prep.pd.read_csv = patched_read_csv
    return orig_read_csv

def plot_and_save_confusion_matrix(y_true, y_pred, out_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Matriz de confusión")
    plt.colorbar(im, ax=ax)
    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Matriz de confusión guardada en: {out_path}")

def main():
    orig = _patch_read_csv()
    try:
        X, y = prep.cargar_y_seleccionar("data.csv")
        num_cols, cat_cols = prep.definir_columnas(X)
        pre = prep.construir_preprocesador(num_cols, cat_cols)
        X_train, X_test, y_train, y_test = prep.hacer_split(X, y)

        gb = GradientBoostingClassifier(learning_rate=0.05, n_estimators=200, max_depth=3)
        clf = Pipeline([("pre", pre), ("gb", gb)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print("\nMétricas (conjunto de prueba):")
        print(f"Accuracy = {accuracy_score(y_test, y_pred):.3f}")
        print(f"Precision = {precision_score(y_test, y_pred):.3f}")

        plot_and_save_confusion_matrix(y_test, y_pred)
    finally:
        prep.pd.read_csv = orig

if __name__ == "__main__":
    main()
