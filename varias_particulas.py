import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import fsolve

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

# Número de partículas
num_particles = 5

# Generar condiciones iniciales aleatorias para las partículas
np.random.seed(42)  # Para reproducibilidad
initial_positions = np.random.rand(num_particles) * 0.01  # Posiciones iniciales entre 0 y 0.01 metros
initial_velocities = np.random.rand(num_particles) * v0  # Velocidades iniciales entre 0 y v0

# Tiempo total y paso de tiempo
T = 1 / f
dt = T / 1000  # Paso de tiempo
num_steps = 10000 # Número de pasos de tiempo

# Inicializar arrays para almacenar resultados
t_values = np.linspace(0, T, num_steps)
x_values_verlet = np.zeros((num_particles, num_steps))
v_values_verlet = np.zeros((num_particles, num_steps))
a_values_verlet = np.zeros((num_particles, num_steps))

x_values_rk4 = np.zeros((num_particles, num_steps))
v_values_rk4 = np.zeros((num_particles, num_steps))

# Función para calcular la aceleración
def acceleration(t):
    return - (e * V0) / (m * d) * np.cos(omega * t)

# Método Velocity Verlet para múltiples partículas
start_time_verlet = time.time()
for p in range(num_particles):
    # Condiciones iniciales para cada partícula
    x_values_verlet[p, 0] = initial_positions[p]
    v_values_verlet[p, 0] = initial_velocities[p]
    a_values_verlet[p, 0] = acceleration(0)
    
    for i in range(1, num_steps):
        # Actualización de la posición
        x_values_verlet[p, i] = x_values_verlet[p, i-1] + v_values_verlet[p, i-1] * dt + 0.5 * a_values_verlet[p, i-1] * dt**2
        
        # Calcular nueva aceleración
        t = t_values[i]
        a_new = acceleration(t)
        
        # Actualización de la velocidad
        v_values_verlet[p, i] = v_values_verlet[p, i-1] + 0.5 * (a_values_verlet[p, i-1] + a_new) * dt
        
        # Actualizar aceleración
        a_values_verlet[p, i] = a_new
end_time_verlet = time.time()
verlet_time = end_time_verlet - start_time_verlet

# Método RK4 para múltiples partículas
start_time_rk4 = time.time()
for p in range(num_particles):
    # Condiciones iniciales para cada partícula
    x_values_rk4[p, 0] = initial_positions[p]
    v_values_rk4[p, 0] = initial_velocities[p]
    
    for i in range(1, num_steps):
        t = t_values[i]
        v = v_values_rk4[p, i-1]
        x = x_values_rk4[p, i-1]
        
        k1_v = dt * acceleration(t)
        k1_x = dt * v
        
        k2_v = dt * acceleration(t + 0.5*dt)
        k2_x = dt * (v + 0.5*k1_v)
        
        k3_v = dt * acceleration(t + 0.5*dt)
        k3_x = dt * (v + 0.5*k2_v)
        
        k4_v = dt * acceleration(t + dt)
        k4_x = dt * (v + k3_v)
        
        v_values_rk4[p, i] = v + (1/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        x_values_rk4[p, i] = x + (1/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
end_time_rk4 = time.time()
rk4_time = end_time_rk4 - start_time_rk4

# Calcular la energía cinética final en eV para múltiples partículas
Ek_final_verlet = 0.5 * m * v_values_verlet[:, -1]**2 / e
Ek_final_rk4 = 0.5 * m * v_values_rk4[:, -1]**2 / e

# Función para calcular Θ(γ0)
def theta(gamma):
    k1 = 0.56
    k2 = 0.25
    return (k1 + k2) / 2 - (k1 - k2) / np.pi * np.arctan(np.pi * np.log(gamma))

# Ecuación para γ0
def gamma0_eq(gamma, delta_max):
    return delta_max * gamma * np.exp(1 - gamma) * theta(gamma)

# Función para calcular δ(E)
def calculate_delta(E, W0, W1, W2, Wmax, delta_max):
    # Resolver γ0
    gamma0_inicial = 1
    gamma0_solucion = fsolve(gamma0_eq, gamma0_inicial, args=(delta_max))[0]

    # Calcular γ2
    gamma2 = (W2 - W0) / (Wmax - W0)

    # Calcular s
    kv = 1
    s = np.log(kv * delta_max) / np.log(gamma2 / 3.6)

    # Calcular δ(E)
    delta = delta_max * (E / Wmax)**s * np.exp(s * (1 - E / Wmax))
    return delta

# Datos para los materiales
materiales = {
    'Plata': {'W0': 16, 'W1': 30, 'W2': 5000, 'Wmax': 165, 'delta_max': 2.22},
    'Aluminio': {'W0': 12.9, 'W1': 85.47, 'W2': 1414, 'Wmax': 350, 'delta_max': 1.5},
    'Cobre': {'W0': 10, 'W1': 25, 'W2': 5000, 'Wmax': 175, 'delta_max': 2.25}
}

# Calcular el coeficiente de emisión secundaria para cada partícula en cada material
delta_verlet = {material: np.zeros(num_particles) for material in materiales}
delta_rk4 = {material: np.zeros(num_particles) for material in materiales}

for material, params in materiales.items():
    for p in range(num_particles):
        delta_verlet[material][p] = calculate_delta(Ek_final_verlet[p], **params)
        delta_rk4[material][p] = calculate_delta(Ek_final_rk4[p], **params)

# Sumar los coeficientes de emisión secundaria para cada material
suma_delta_verlet = {material: np.sum(delta) for material, delta in delta_verlet.items()}
suma_delta_rk4 = {material: np.sum(delta) for material, delta in delta_rk4.items()}

# Mostrar resultados
print(f"Tiempo de computación - Velocity Verlet: {verlet_time:.6f} segundos")
print(f"Tiempo de computación - RK4: {rk4_time:.6f} segundos")
for material in materiales:
    print(f"Suma del coeficiente de emisión secundaria (Verlet) para {material}: {suma_delta_verlet[material]:.12e}")
    print(f"Suma del coeficiente de emisión secundaria (RK4) para {material}: {suma_delta_rk4[material]:.12e}")

# Comparar todas las soluciones en una gráfica
plt.figure(figsize=(10, 6))
for p in range(num_particles):
    plt.plot(t_values / T, x_values_verlet[p, :], label=f'Verlet Partícula {p+1}')
    plt.plot(t_values / T, x_values_rk4[p, :], label=f'RK4 Partícula {p+1}', linestyle='--')

plt.xlabel('t/T')
plt.ylabel('x (m)')
plt.title('Comparación de trayectorias de múltiples partículas')
plt.legend()
plt.grid(True)

plt.show()

