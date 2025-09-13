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
2. **Red Neuronal (MLPClassifier)**
   - Arquitectura: Dos capas ocultas con 100 y 50 neuronas respectivamente.
   - Función de activación: `relu`.
   - Optimizador: `adam`.
   - Iteraciones máximas: 500.
   - Semilla aleatoria: 42.
   - Implementado con `sklearn.neural_network.MLPClassifier` (ver `neural/mlp.py`).
3. **Gradient Boosting**
   - Implementado con `sklearn.ensemble.GradientBoostingClassifier`.  
   - Parámetros principales: `n_estimators=200`, `learning_rate=0.05`, `max_depth=4`, `min_samples_leaf=5`, `subsample=1.0`.  
   - Semilla aleatoria: 42.  
   - Mismo preprocesamiento que los otros modelos.  
   - Característica principal: construye árboles de manera secuencial, donde cada nuevo árbol intenta corregir los errores del anterior, lo que permite capturar relaciones complejas en los datos.

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
- **Matriz de confusión (MLP)** (`matrizConf_mlp.png`) para la red neuronal, mostrando la distribución de aciertos y errores.
- Se reportan las mismas métricas (accuracy y precisión por clase) para comparar directamente con Random Forest.
- **Matriz de confusión (Gradient Boosting)** (`matrizConf_gb.png`) para comparar su rendimiento frente a los otros algoritmos.

---

## 5. Análisis comparativo: Random Forest vs Red Neuronal vs Gradient Boostiong

Ambos modelos fueron entrenados y evaluados bajo las mismas condiciones de preprocesamiento y división de datos. A continuación, se resumen las observaciones principales:

- **Random Forest**
   - Tiende a ser más robusto ante el sobreajuste y requiere menos ajuste de hiperparámetros.
   - Es interpretable gracias a la importancia de características.
   - Suele converger más rápido y es menos sensible a la escala de los datos.

- **Red Neuronal (MLPClassifier)**
   - Puede modelar relaciones más complejas y no lineales entre las variables que hay.
   - Su rendimiento depende mucho de la normalización y de la configuración de hiperparámetros (número de capas, neuronas, tasa de aprendizaje, etc.).
   - Puede requerir más tiempo de entrenamiento y es más sensible a los datos mas desbalanceados o mas ruidosos.

- **Gradient Boosting**  
   - Ofrece un equilibrio entre complejidad y rendimiento, puede superar a Random Forest en datasets pequeños y medianos.  
   - Puede capturar patrones más sutiles al construir árboles de forma secuencial, pero es más sensible al ajuste de hiperparámetros (tasa de aprendizaje, profundidad, número de árboles).  
   - Es más costoso en tiempo de entrenamiento que Random Forest, aunque menos que una red neuronal compleja (MLP).  


**Comparación de resultados:**
- Si bien ambos modelos pueden alcanzar resultados similares en accuracy, Random Forest suele ser más estable y fácil de interpretar en este tipo de problemas tabulares, dando en este caso una prediccion mas precisa.
- La red neuronal puede superar a Random Forest si se ajusta cuidadosamente y si existen patrones complejos en los datos, pero también puede sobreajustar si no se regula adecuadamente.
- Gradient Boosting logró un rendimiento competitivo, con buen balance entre precisión y capacidad de generalización, destacándose como una opción sólida si hay tiempo para optimización fina.

Recomendamos revisar las métricas y las matrices de confusión generadas para elegir el modelo más adecuado según el objetivo del análisis.

---
