import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def cargar_y_seleccionar(ruta_csv: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(ruta_csv)

    # Target
    y = df["Survived"].astype(int)
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].copy()

    return X, y


def definir_columnas(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = ["Age", "SibSp", "Parch", "Fare"]
    cat_cols = ["Pclass", "Sex", "Embarked"]
    return num_cols, cat_cols


def construir_preprocesador(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])

    cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    return pre


def hacer_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y 
    )
    return X_train, X_test, y_train, y_test


def preprocesar_titanic(ruta_csv: str = "data.csv", test_size: float = 0.2, seed: int = 42):
    X, y = cargar_y_seleccionar(ruta_csv)

    num_cols, cat_cols = definir_columnas(X)

    pre = construir_preprocesador(num_cols, cat_cols)

    X_train, X_test, y_train, y_test = hacer_split(X, y, test_size=test_size, seed=seed)

    pre.fit(X_train)
    X_train_t = pre.transform(X_train)
    X_test_t  = pre.transform(X_test)

    return X_train_t, X_test_t, y_train, y_test, pre, (num_cols, cat_cols)


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, pre, cols = preprocesar_titanic()
    print("Preprocesamiento hecho:", Xtr.shape, Xte.shape)