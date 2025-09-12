# Informe Actividad 2 – Aprendizaje Supervisado (Titanic)

## 1. Descripción del dataset
- **Fuente:** [Titanic - Machine Learning from Disaster de Kaggle](https://www.kaggle.com/c/titanic/data)  
- **Número de registros:** 891 pasajeros.  
- **Número de variables:** 12 columnas iniciales.  
- **Variable objetivo:** `Survived`  
  - 0 = No sobrevivió  
  - 1 = Sobrevivió  

El objetivo del problema es predecir la supervivencia de un pasajero del Titanic a partir de características demográficas y de su boleto.

---

## 2. Preprocesamiento realizado

### a. Limpieza de datos faltantes
- **Numéricas:** `Age`, `Fare` → imputadas con la mediana.  
- **Categóricas:** `Embarked` → imputada con la moda.  
- Descartamos las columnas: (`Name`, `Ticket`, `Cabin`, `PassengerId`) porque las consideramos irrelevantes o contaban con muchos daltos faltantes.  

### b. Codificación de variables categóricas
- Variables categóricas (`Pclass`, `Sex`, `Embarked`) fueron convertidas a variables dummy utilizando OneHotEncoder.  

### c. Escalado/normalización
- Variables numéricas (`Age`, `SibSp`, `Parch`, `Fare`) fueron estandarizadas con StandardScaler.  

### d. División en train/test
- El dataset se dividió en 80% train y 20% test.  
- Usamos `stratify=y` para mantener la proporción de clases (61.6% no sobrevivió, 38.4% sobrevivió).

---

## 3. Modelos entrenados

1. **Random Forest**  
   - Parámetros principales: `n_estimators=300`, `class_weight="balanced"`, `random_state=42`.  
   - Utilizamos (`n_jobs=-1`) para acelerar el entrenamiento.

---

## 4. Evaluación de resultados

### a. Métricas de rendimiento
Se utilizaron las siguientes métricas:
- **Accuracy:** porcentaje de aciertos totales.  
- **Precision (sobrevivió):** de los predichos como sobrevivientes, cuántos realmente lo fueron.  
- **Precision (no sobrevivió):** de los predichos como no sobrevivientes, cuántos realmente lo fueron.  



### b. Visualizaciones
- **Matriz de confusión** (`matrizConf_rf.png`) para observar aciertos y errores del modelo.  
- **Gráfico de importancia de características** (`feature_importances_rf.png`), donde se observa que `Fare`, `Age` y `Sex` fueron las variables más influyentes.

---
