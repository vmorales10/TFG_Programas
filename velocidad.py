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
dt = T / 1000 # Paso de tiempo
num_steps = 1000  # Número de pasos de tiempo

# Inicializar arrays para almacenar resultados
t_values = np.linspace(0, T, num_steps)
x_values_verlet = np.zeros(num_steps)
v_values_verlet = np.zeros(num_steps)
a_values_verlet = np.zeros(num_steps)

x_values_rk4 = np.zeros(num_steps)
v_values_rk4 = np.zeros(num_steps)

# Función para calcular la aceleración
def acceleration(t):
    return - (e * V0) / (m * d) * np.cos(omega * t)

# Encontrar el primer punto donde la aceleración es positiva
start_index = 0
for i in range(num_steps):
    t = t_values[i]
    a_new = acceleration(t)
    if a_new > 0:
        start_index = i
        break

# Ajustar los valores iniciales para comenzar desde el primer punto con aceleración positiva
x_values_verlet[0] = 0
v_values_verlet[0] = v0
a_values_verlet[0] = acceleration(t_values[start_index])

x_values_rk4[0] = 0
v_values_rk4[0] = v0

# Método Velocity Verlet a partir del primer punto con aceleración positiva
start_time_verlet = time.time()
for i in range(1, num_steps - start_index):
    # Actualización de la posición
    x_values_verlet[i] = x_values_verlet[i-1] + v_values_verlet[i-1] * dt + 0.5 * a_values_verlet[i-1] * dt**2
    
    # Calcular nueva aceleración
    t = t_values[start_index + i]
    a_new = acceleration(t)
    
    # Actualización de la velocidad
    v_values_verlet[i] = v_values_verlet[i-1] + 0.5 * (a_values_verlet[i-1] + a_new) * dt
    
    # Actualizar aceleración
    a_values_verlet[i] = a_new
end_time_verlet = time.time()
verlet_time = end_time_verlet - start_time_verlet

# Método RK4
start_time_rk4 = time.time()
for i in range(1, num_steps - start_index):
    t = t_values[start_index + i]
    v = v_values_rk4[i-1]
    
    k1 = dt * acceleration(t)
    k2 = dt * acceleration(t + 0.5*dt)
    k3 = dt * acceleration(t + 0.5*dt)
    k4 = dt * acceleration(t + dt)
    
    v_values_rk4[i] = v + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
end_time_rk4 = time.time()
rk4_time = end_time_rk4 - start_time_rk4

# Recortar los valores para graficar solo a partir del primer punto con aceleración positiva
t_values = t_values[start_index:]
v_values_verlet = v_values_verlet[:num_steps - start_index]
v_values_rk4 = v_values_rk4[:num_steps - start_index]

# Función para calcular la velocidad usando la solución analítica
def analytical_velocity(t):
    return v0 + (e * V0) / (m * omega * d) * np.sin(alpha) - (e * V0) / (m * omega * d) * np.sin(omega * t)

# Calcular velocidad usando la solución analítica
analytical_v_values = analytical_velocity(t_values)

# Calcular el error cuadrático medio (MSE)
mse_verlet = np.mean((v_values_verlet - analytical_v_values)**2)
mse_rk4 = np.mean((v_values_rk4 - analytical_v_values)**2)

# Mostrar resultados
print(f"Tiempo de computación - Velocity Verlet: {verlet_time:.6f} segundos")
print(f"Tiempo de computación - RK4: {rk4_time:.6f} segundos")
print(f"Error cuadrático medio (MSE) - Velocity Verlet: {mse_verlet:.6e}")
print(f"Error cuadrático medio (MSE) - RK4: {mse_rk4:.6e}")

# Comparar todas las soluciones en una gráfica
plt.figure(figsize=(10, 6))
plt.plot(t_values / T, v_values_verlet, label='Velocity Verlet')
plt.plot(t_values / T, v_values_rk4, label='Método RK4')
plt.plot(t_values / T, analytical_v_values, label='Solución Analítica', linestyle='--')
plt.xlabel('t/T')
plt.ylabel('v (m/s)')
plt.title('Comparación entre Velocity Verlet, Método RK4 y Solución Analítica')
plt.legend()
plt.grid(True)

# Anotar los resultados en la gráfica
plt.text(0.5, 0.25, f"Tiempo de computación - Velocity Verlet: {verlet_time:.6f} s", transform=plt.gca().transAxes)
plt.text(0.5, 0.2, f"Tiempo de computación - RK4: {rk4_time:.6f} s", transform=plt.gca().transAxes)
plt.text(0.5, 0.15, f"MSE - Velocity Verlet: {mse_verlet:.6e}", transform=plt.gca().transAxes)
plt.text(0.5, 0.1, f"MSE - RK4: {mse_rk4:.6e}", transform=plt.gca().transAxes)

plt.show()

