import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


#Datos
m2 = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
habitaciones = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
alquiler = [300, 330, 360, 395, 430, 460, 490, 520, 550, 580, 610, 640, 670, 700, 730]

#DataFrame
data = {"m2": m2, "habitaciones": habitaciones, "alquiler": alquiler}
df = pd.DataFrame(data)
print(df.head(10))

#preparamos para poder entrenar las dos variables de entrada en este caso m2 y habitaciones 
X = df[["m2", "habitaciones"]].values
y = df["alquiler"].values
#Entrenamos algoritmo
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Predecimos valor de theta0, 1, 2
print("El valor de theta 0 es:", lin_reg.intercept_)
print("El valor de theta 1 es:", lin_reg.coef_[0])
print("El valor de theta 2 es:", lin_reg.coef_[1])

# Valores extremos de m2 y habitaciones
min_values = [df["m2"].min(), df["habitaciones"].min()]
max_values = [df["m2"].max(), df["habitaciones"].max()]

# Juntás los valores en un array 2D
X_min_max = np.array([min_values, max_values])

# Predicción
alquiler_pred = lin_reg.predict(X_min_max)

#gráfico
plt.figure(figsize=(8,6))
scatter = plt.scatter(df["m2"], df["alquiler"], c=df["habitaciones"], cmap="viridis", s=80)
plt.xlabel("Metros cuadrados (m2)")
plt.ylabel("Alquiler")
plt.title("Relación entre m2, habitaciones y alquiler")
legend = plt.legend(*scatter.legend_elements(), title="Habitaciones")
plt.gca().add_artist(legend)
plt.show()
