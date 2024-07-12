import numpy as np
import matplotlib.pyplot as plt
import time

# Parámetros constantes
V0 = 30.8  # Voltios
d = 0.001  # metros
f = 1e9  # Hz
omega = 2 * np.pi * f  # frecuencia angular
e = 1.602e-19  # Carga del electrón
m = 9.109e-31  # Masa del electrón
alpha = 99.8 * np.pi / 180  # Convertir grados a radianes
Ek_eV = 3.68  # Energía cinética en eV
Ek_J = Ek_eV * 1.602e-19  # Energía cinética en Joules

# Velocidad inicial en m/s
v0 = np.sqrt(2 * Ek_J / m)
print("Velocidad inicial v0 (calculada):", v0)

# Tiempo total y paso de tiempo
T = 1 / f
dt = T / 1000  # Paso de tiempo
num_steps = 1000  # Número de pasos de tiempo

# Inicializar arrays para almacenar resultados
t_values = np.linspace(0, T, num_steps)
a_values_verlet = np.zeros(num_steps)
a_values_rk4 = np.zeros(num_steps)
a_values_analitica = np.zeros(num_steps)

# Función para calcular la aceleración
def aceleracion(t):
    return - (e * V0) / (m * d) * np.cos(omega * t)

# Calcular la aceleración usando la solución analítica
for i in range(num_steps):
    t = t_values[i]
    a_values_analitica[i] = aceleracion(t)

# Métodos numéricos para calcular la aceleración
# Método Velocity Verlet
start_time_verlet = time.time()
for i in range(num_steps):
    t = t_values[i]
    a_new = aceleracion(t)
    a_values_verlet[i] = a_new
end_time_verlet = time.time()
verlet_time = end_time_verlet - start_time_verlet

# Método RK4
start_time_rk4 = time.time()
for i in range(num_steps):
    t = t_values[i]
    
    k1_a = aceleracion(t)
    k2_a = aceleracion(t + 0.5*dt)
    k3_a = aceleracion(t + 0.5*dt)
    k4_a = aceleracion(t + dt)
    
    a_values_rk4[i] = (1/6) * (k1_a + 2*k2_a + 2*k3_a + k4_a)
end_time_rk4 = time.time()
rk4_time = end_time_rk4 - start_time_rk4

# Mostrar resultados
print(f"Tiempo de computación - Velocity Verlet: {verlet_time:.6f} segundos")
print(f"Tiempo de computación - RK4: {rk4_time:.6f} segundos")

# Comparar todas las soluciones en una gráfica
plt.figure(figsize=(10, 6))
plt.plot(t_values / T, a_values_verlet, label='Velocity Verlet')
plt.plot(t_values / T, a_values_rk4, label='Método RK4')
plt.plot(t_values / T, a_values_analitica, label='Solución Analítica', linestyle='--')
plt.xlabel('t/T')
plt.ylabel('Aceleración (m/s^2)')
plt.title('Comparación entre Velocity Verlet, Método RK4 y Solución Analítica para la Aceleración')
plt.legend()
plt.grid(True)

# Anotar los resultados en la gráfica
plt.text(0.3, 0.9, f"Tiempo de computación - Velocity Verlet: {verlet_time:.6f} s", transform=plt.gca().transAxes)
plt.text(0.3, 0.85, f"Tiempo de computación - RK4: {rk4_time:.6f} s", transform=plt.gca().transAxes)

plt.show()
