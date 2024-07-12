
import numpy as np
import matplotlib.pyplot as plt
import time

# Constantes
e = 1.602176634e-19  # Carga del electrón en C
m = 9.1093837139e-31  # Masa del electrón en kg
d = 1e-3  # Distancia en m (1 mm)
f = 1.35e9  # Frecuencia en Hz (1.35 GHz)
V = 27.9  # Voltaje en V
alpha = 0  # Fase inicial en radianes (0 grados)
x0 = 0  # Posición inicial en m
t0 = 0  # Tiempo inicial en s

# Parámetros derivados
omega = 2 * np.pi * f  # Frecuencia angular en rad/s

# Calcular la aceleración
def acceleration(x, t):
    return (e * V / (m * d)) * np.cos(omega * t + alpha)

# Función para calcular la aceleración en cada instante de tiempo
def compute_acceleration(times):
    return [acceleration(0, t) for t in times]

# Parámetros de la simulación
delta_t = 1e-12  # Paso de tiempo en s
T = 2.82e-10  # Tiempo total de simulación en s
num_steps = int(T / delta_t)
times = np.linspace(t0, T, num_steps)

# Calcular la aceleración en cada instante de tiempo
accelerations = compute_acceleration(times)

# Graficar los resultados
plt.figure(figsize=(12, 6))

plt.plot(times, accelerations, label='Aceleración', color='red', linestyle='-', linewidth=0.7)

# Añadir la placa de metal a la derecha
plt.axvline(x=times[-1], color='gray', linestyle='-', linewidth=2)

# Anotaciones de tiempo de computación
textstr = f'Tiempo total de simulación: {T:.2e} s'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=props, color='black')

plt.xlabel('Tiempo (s)')
plt.ylabel('Aceleración (m/s²)')
plt.title('Aceleración vs. Tiempo entre Placas Metálicas')
plt.legend()

plt.tight_layout()
plt.show()