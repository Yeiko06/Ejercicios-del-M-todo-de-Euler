import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros del circuito RC
R = 1000  # Ohms
C = 0.001  # Farads
V_fuente = 5  # Voltios

# Definición de la EDO: dV/dt = (V_fuente - V)/(R*C)
def f(t, V):
    return (V_fuente - V)/(R*C)

# Condiciones iniciales
t0 = 0
V0 = 0
tf = 5
n = 20

# Paso
h = (tf - t0)/n

# Inicialización de listas para almacenar resultados
t_vals = [t0]
V_vals = [V0]

# Método de Euler
t = t0
V = V0
for i in range(n):
    V = V + h * f(t, V)
    t = t + h
    t_vals.append(t)
    V_vals.append(V)

# Solución analítica
def sol_analitica(t):
    return V_fuente * (1 - np.exp(-t/(R*C)))

# Calcular solución analítica
V_analitica = [sol_analitica(t) for t in t_vals]

# Crear tabla de resultados
data = {
    "t": t_vals,
    "V_aproximada": V_vals,
    "V_analitica": V_analitica,
    "Error": [abs(V_vals[i] - V_analitica[i]) for i in range(len(t_vals))]
}
df = pd.DataFrame(data)
csv_path = "ejercicio1_resultados.csv"
df.to_csv(csv_path, index=False)

# Graficar ambas soluciones
plt.figure(figsize=(10, 6))
plt.plot(t_vals, V_vals, 'o-', label='Solución aproximada (Euler)', color='blue')
plt.plot(t_vals, V_analitica, '-', label='Solución analítica', color='red')
plt.title('Carga de un capacitor - Circuito RC')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.grid(True)
plt.legend()
image_path = "ejercicio1_solucion.png"
plt.savefig(image_path)
plt.show()

# Mostrar tabla de resultados
print("Tabla de resultados para el Ejercicio 1:")
print(df)