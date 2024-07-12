
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

# Solución analítica
def analytical_solution(t):
    term1 = x0
    term2 = (0 + (e * V / (m * omega * d)) * np.sin(omega * t0 + alpha)) * (t - t0)
    term3 = (e * V / (m * d * omega**2)) * (np.cos(omega * t + alpha) - np.cos(omega * t0 + alpha))
    term4 = -(e * V / (2 * m)) * (t - t0)**2
    return term1 + term2 - term3 + term4

# Función para la derivada del sistema
def f(y, t):
    x, v = y
    a = acceleration(x, t)
    return np.array([v, a])

# Método de Runge-Kutta 4
def RK_method(f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(0, len(t) - 1):
        h = t[i + 1] - t[i]
        F1 = h * f(y[i], t[i])
        F2 = h * f(y[i] + F1 / 2, t[i] + h / 2)
        F3 = h * f(y[i] + F2 / 2, t[i] + h / 2)
        F4 = h * f(y[i] + F3, t[i] + h)
        y[i + 1] = y[i] + (F1 + 2 * F2 + 2 * F3 + F4) / 6
    return y

# Método de Verlet
def verlet_method(acceleration, x0, v0, t):
    num_steps = len(t)
    x = x0
    v = v0
    a = acceleration(x, t[0])
    
    positions = np.zeros(num_steps)
    for i in range(num_steps):
        x_new = x + v * (t[1] - t[0]) + 0.5 * a * (t[1] - t[0])**2
        a_new = acceleration(x_new, t[i] + (t[1] - t[0]))
        v_new = v + 0.5 * (a + a_new) * (t[1] - t[0])
        
        # Actualizar para el siguiente paso
        x = x_new
        v = v_new
        a = a_new
        
        # Almacenar resultados
        positions[i] = x
    
    return positions

# Función para calcular el tiempo de computación
def compute_time(method, *args):
    start_time = time.time()
    result = method(*args)
    end_time = time.time()
    return result, end_time - start_time

# Función para calcular el MSE (Error)
def compute_error(numerical_solution, analytical_solution):
    return np.mean((numerical_solution - analytical_solution) ** 2)

# Condiciones iniciales
v0 = 0  # Velocidad inicial en m/s
y0 = [x0, v0]  # Estado inicial [posición, velocidad]

# Parámetros de la simulación
delta_t = 1e-12  # Paso de tiempo en s
T = 2.82e-10  # Tiempo total de simulación en s
num_steps = int(T / delta_t)
times = np.linspace(t0, T, num_steps)

# Ejecutar métodos y calcular tiempos de computación
positions_verlet, time_verlet = compute_time(verlet_method, acceleration, x0, v0, times)
y_rk4, time_rk4 = compute_time(RK_method, f, y0, times)
positions_rk4 = y_rk4[:, 0]

# Calcular la solución analítica
analytical_positions = np.array([analytical_solution(t) for t in times])

# Calcular Error (MSE)
error_verlet = compute_error(positions_verlet, analytical_positions)
error_rk4 = compute_error(positions_rk4, analytical_positions)

# Graficar los resultados
plt.figure(figsize=(12, 6))

plt.plot(times, positions_verlet, label='Verlet', color='orange', linestyle='-', linewidth=0.7)
plt.plot(times, positions_rk4, label='Runge-Kutta 4', color='green', linestyle='-', linewidth=0.7)
plt.plot(times, analytical_positions, label='Analítica', color='blue', linestyle='dashed', linewidth=0.7)

# Añadir la placa de metal a la derecha
plt.axvline(x=times[-1], color='gray', linestyle='-', linewidth=2)

# Anotaciones de tiempo de computación y errores medios cuadrados
textstr = '\n'.join((
    f'Error - Verlet: {error_verlet:.6e}',
    f'Error - RK4: {error_rk4:.6e}',
    f'Tiempo de computación - Verlet: {time_verlet:.6f} s',
    f'Tiempo de computación - RK4: {time_rk4:.6f} s'
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.65, 0.25, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=props, color='black')

plt.xlabel('Tiempo (s)')
plt.ylabel('Posición (m)')
plt.title('Posición vs. Tiempo entre Placas Metálicas')
plt.legend()

plt.tight_layout()
plt.show()