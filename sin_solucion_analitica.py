import numpy as np
import matplotlib.pyplot as plt

# Constantes físicas
e = 1.602e-19  # Carga del electrón en Coulombs
m_electron = 9.109e-31  # Masa del electrón en kg

def aceleracion(posicion, tiempo, V0, d, omega, masa):
    """
    Calcula la aceleración de un electrón en un campo eléctrico variable armónicamente.
    """
    return (-e * V0 / d * np.cos(omega * tiempo)) / masa

def verlet_integration(initial_position, initial_velocity, V0, d, omega, dt, total_time, masa):
    """
    Método de Verlet para calcular la posición y la velocidad en función del tiempo.
    
    Parámetros:
    - initial_position: Posición inicial de la partícula.
    - initial_velocity: Velocidad inicial de la partícula.
    - V0: Amplitud de la tensión.
    - d: Distancia entre las placas.
    - omega: Frecuencia angular.
    - dt: Paso de tiempo.
    - total_time: Tiempo total de la simulación.
    - masa: Masa de la partícula.
    
    Retorna:
    - posiciones: Lista de posiciones en cada paso de tiempo.
    - velocidades: Lista de velocidades en cada paso de tiempo.
    - tiempos: Lista de tiempos en cada paso de tiempo.
    """
    num_steps = int(total_time / dt)
    posiciones = np.zeros(num_steps)
    velocidades = np.zeros(num_steps)
    tiempos = np.zeros(num_steps)
    
    posiciones[0] = initial_position
    velocidades[0] = initial_velocity
    tiempos[0] = 0
    
    # Primer paso usando el método de Euler para inicializar
    posiciones[1] = posiciones[0] + velocidades[0] * dt + 0.5 * aceleracion(posiciones[0], tiempos[0], V0, d, omega, masa) * dt**2
    
    for i in range(1, num_steps - 1):
        tiempos[i + 1] = tiempos[i] + dt
        posiciones[i + 1] = 2 * posiciones[i] - posiciones[i - 1] + aceleracion(posiciones[i], tiempos[i], V0, d, omega, masa) * dt**2
        velocidades[i] = (posiciones[i + 1] - posiciones[i - 1]) / (2 * dt)
    
    # Calcula la velocidad en el último paso
    velocidades[-1] = (posiciones[-1] - posiciones[-2]) / dt
    
    return posiciones, velocidades, tiempos

def rk4_integration(initial_position, initial_velocity, V0, d, omega, dt, total_time, masa):
    """
    Método de Runge-Kutta de cuarto orden para calcular la posición y la velocidad en función del tiempo.
    
    Parámetros:
    - initial_position: Posición inicial de la partícula.
    - initial_velocity: Velocidad inicial de la partícula.
    - V0: Amplitud de la tensión.
    - d: Distancia entre las placas.
    - omega: Frecuencia angular.
    - dt: Paso de tiempo.
    - total_time: Tiempo total de la simulación.
    - masa: Masa de la partícula.
    
    Retorna:
    - posiciones: Lista de posiciones en cada paso de tiempo.
    - velocidades: Lista de velocidades en cada paso de tiempo.
    - tiempos: Lista de tiempos en cada paso de tiempo.
    """
    num_steps = int(total_time / dt)
    posiciones = np.zeros(num_steps)
    velocidades = np.zeros(num_steps)
    tiempos = np.zeros(num_steps)
    
    posiciones[0] = initial_position
    velocidades[0] = initial_velocity
    tiempos[0] = 0
    
    for i in range(num_steps - 1):
        tiempos[i + 1] = tiempos[i] + dt
        
        k1_v = aceleracion(posiciones[i], tiempos[i], V0, d, omega, masa) * dt
        k1_x = velocidades[i] * dt
        
        k2_v = aceleracion(posiciones[i] + 0.5 * k1_x, tiempos[i] + 0.5 * dt, V0, d, omega, masa) * dt
        k2_x = (velocidades[i] + 0.5 * k1_v) * dt
        
        k3_v = aceleracion(posiciones[i] + 0.5 * k2_x, tiempos[i] + 0.5 * dt, V0, d, omega, masa) * dt
        k3_x = (velocidades[i] + 0.5 * k2_v) * dt
        
        k4_v = aceleracion(posiciones[i] + k3_x, tiempos[i] + dt, V0, d, omega, masa) * dt
        k4_x = (velocidades[i] + k3_v) * dt
        
        velocidades[i + 1] = velocidades[i] + (1/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        posiciones[i + 1] = posiciones[i] + (1/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    
    return posiciones, velocidades, tiempos

# Entrada de parámetros por parte del usuario
initial_position = float(input("Ingrese la posición inicial (m): "))
initial_velocity = float(input("Ingrese la velocidad inicial (m/s): "))
V0 = float(input("Ingrese la amplitud de la tensión (V): "))
d = float(input("Ingrese la distancia entre las placas (m): "))
f = float(input("Ingrese la frecuencia (Hz): "))
dt = float(input("Ingrese el paso de tiempo (s): "))
total_time = float(input("Ingrese el tiempo total de simulación (s): "))

# Elección de la masa
masa_choice = input("¿Usar la masa del electrón? (si/no): ").strip().lower()
if masa_choice == 'no':
    masa = float(input("Ingrese la masa de la partícula (kg): "))
else:
    masa = m_electron

# Cálculo de la frecuencia angular
omega = 2 * np.pi * f

# Realiza la simulación usando el método de Verlet
posiciones_verlet, velocidades_verlet, tiempos_verlet = verlet_integration(initial_position, initial_velocity, V0, d, omega, dt, total_time, masa)

# Realiza la simulación usando el método de Runge-Kutta de cuarto orden
posiciones_rk4, velocidades_rk4, tiempos_rk4 = rk4_integration(initial_position, initial_velocity, V0, d, omega, dt, total_time, masa)

# Calcula la energía final en eV
energia_final_verlet = 0.5 * masa * velocidades_verlet[-1]**2 / e
energia_final_rk4 = 0.5 * masa * velocidades_rk4[-1]**2 / e

# Grafica los resultados
plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
plt.plot(tiempos_verlet, posiciones_verlet, label="Posición (m) - Verlet")
plt.plot(tiempos_rk4, posiciones_rk4, label="Posición (m) - RK4", linestyle='dashed')
plt.xlabel("Tiempo (s)")
plt.ylabel("Posición (m)")
plt.legend()
plt.grid()

# Imprime la velocidad final y energía en la gráfica de posición
plt.text(0.25 * total_time, 0.95 * max(max(posiciones_verlet), max(posiciones_rk4)),
         f"Vf (Verlet): {velocidades_verlet[-1]:.6f} m/s\nEf (Verlet): {energia_final_verlet:.6f} eV\n"
         f"Vf (RK4): {velocidades_rk4[-1]:.6f} m/s\nEf (RK4): {energia_final_rk4:.6f} eV",
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.subplot(2, 1, 2)
plt.plot(tiempos_verlet, velocidades_verlet, label="Velocidad (m/s) - Verlet", color="orange")
plt.plot(tiempos_rk4, velocidades_rk4, label="Velocidad (m/s) - RK4", color="red", linestyle='dashed')
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (m/s)")
plt.legend()
plt.grid()

# Imprime la velocidad final y energía en la gráfica de velocidad
plt.text(0.25 * total_time, 0.95 * max(max(velocidades_verlet), max(velocidades_rk4)),
         f"Vf (Verlet): {velocidades_verlet[-1]:.6f} m/s\nEf (Verlet): {energia_final_verlet:.6f} eV\n"
         f"Vf (RK4): {velocidades_rk4[-1]:.6f} m/s\nEf (RK4): {energia_final_rk4:.6f} eV",
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()

# Imprime la velocidad final y energía en la consola
print(f"Velocidad final (Verlet): {velocidades_verlet[-1]:.6f} m/s")
print(f"Energía final (Verlet): {energia_final_verlet:.6f} eV")
print(f"Velocidad final (RK4): {velocidades_rk4[-1]:.6f} m/s")
print(f"Energía final (RK4): {energia_final_rk4:.6f} eV")
