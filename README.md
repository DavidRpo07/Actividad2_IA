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
1. **Random Forest**
  - Exactitud: 0.8156
  - Precisión(Sobrevivió): 0.7903
  - Precisión(No Sobrevivió): 0.8291
2. **Red Neuronal (MLPClassifier)**
  - Exactitud: 0.7877
  - Precisión(Sobrevivió): 0.7385
  - Precisión(No Sobrevivió): 0.8158
3. **Gradient Boosting**
  - Exactitud: 0.804
  - Precisión(Sobrevivió): 0.804
    
### b. Visualizaciones
Las siguientes visualizaciones se encuentran en el directorio de cada modelo.
- **Matriz de confusión (Random Forest)** (`matrizConf_rf.png`). 
- **Gráfico de importancia de características(Random Forest)** (`feature_importances_rf.png`), donde se observa que `Fare`, `Age` y `Sex` fueron las variables más influyentes.
- **Matriz de confusión (MLP)** (`matrizConf_mlp.png`).
- **Matriz de confusión (Gradient Boosting)** (`matrizConf_gb.png`).

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


El Random Forest fue el modelo con mejor rendimiento global y el más balanceado, mostrando resultados consistentes tanto en la detección de sobrevivientes como de no sobrevivientes. El Gradient Boosting, en cambio, presentó un comportamiento algo más desbalanceado: logró un mayor número de aciertos en la clase de no sobrevivientes respecto al Random Forest, pero tuvo un desempeño más débil en la detección de sobrevivientes, siendo además el modelo con mayor cantidad de falsos negativos. Finalmente, la Red Neuronal fue la de menor rendimiento global, aunque mantuvo un comportamiento relativamente equilibrado en ambas clases, presentó el mayor número de falsos positivos (17), lo que muestra una tendencia a sobreestimar la clase de sobrevivientes.

---
## 6. Conclusiones

Este trabajo nos permitió llevar a cabo un proceso completo de entrenamiento de modelos, desde la preparación de los datos hasta el entrenamiento y la comparación de resultados entre distintos algoritmos de clasificación. Se pudo evidenciar la relevancia del preprocesamiento, ya que aspectos como la limpieza, la estandarización y el manejo de datos faltantes resultaron determinantes para lograr un buen entrenamiento y, por ende , obtener resultados confiables.

En cuanto al desempeño, el modelo que se destacó como el más adecuado para este problema fue el Random Forest, ya que presentó la mayor exactitud y un balance consistente en la detección tanto de sobrevivientes como de no sobrevivientes. Su robustez frente al sobreajuste lo convierte en una herramienta muy útil para datasets de tamaño moderado, con un nivel de ruido controlado y que combinan variables numéricas y categóricas.

En conclusión, esta actividad no solo permitió comprender el flujo completo de entrenamiento y evaluación de modelos, sino también reforzó la importancia de elegir el algoritmo en función de las características del problema y del conjunto de datos disponible.

