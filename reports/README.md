# Proyecto de Machine Learning: Accidentes en Canada
**Máster en Data Science para finanzas <br />**
**Autores: <br />**
González Díaz, Guillermo <br />
Sebastiani, Carlos  <br />


## Objetivos del trabajo:
El objectivo del trabajo será crear un modelo de Machine Learning que, habiendo una persona implicada en un accidente, determine la probabilidad de que haya al menos una víctima mortal en dicho accidente.Para esto se han utilizado disintos modelos, para los cuales se ha optimizado el que ha tenido mejores metricas, en nuestro caso, el **RandomForest**

## Partes del trabajo
El trabajo consiste en:
- EDA acerca del dataset sacado del gobierno canadiense, el cual contiene información relenvante acerca de las variables y las características del mismo.
- Preprocesado de valores missing
- Preprocesado de tratameinto de variables para hacer el encoding de las mismas. 
- Distintos modelos dee machine learning:
    - Modelo Base (Decision Tree)
    - LinearRegression (Logistic) + Lasso
    - RandomForest
    - XGBoost
    - LightBoost
- Interpretabilidad utilizando SHAP




## Evaluación de modelos para escoger el ganador
Con el mejor modelo hemos logrado predecir con las siguientes métricas
**Accurancy:** de todos las predicciones cuantas hace bien, mientras más cerca de 1 mejor.  <br /> 
**Recall:** del total de accidentes mortales que hay, cuantos ha predicho correctamente, mientras más cerca de 1 mejor. <br /> 
**Precision:**  del total de accidentes que ha predicho como mortales, cuantos lo han sido en realidad, mientras más cerca de 1 mejor. <br />
**F1 score:** es una combinación entre el precision y el recall que utiliza ambas métricas.

Las mértricas nos interesan para evaluar el modelo son el **recall, el presicion y el f1 score.** <br /> 

También nos va a interesar el tipo de error que se comete. Ya que no es lo mismo predecir que hay un muerto y que no lo haya habido a no predecirlo y que si lo haya habido. Por lo cual en cuanto más alto nos salga en la **matriz de confusión**  el True Positive, mejor será nuestra predicción ya que  no estamos tan interesados en el accuracy, pero si en **predecir correctamente las muertes.**

Cabe destacar que esto habría que someterlo a un análisis en base a la **matriz de coste** que no tenemos.
