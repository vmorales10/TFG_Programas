import numpy as np
import matplotlib.pyplot as plt

# Constantes
e = 1.602176634e-19  # Carga del electrón en Coulombs
m = 9.1093837139e-31  # Masa del electrón en kg

# Datos de materiales
materiales = {
    "Plata": {"W0": 16, "W1": 30, "W2": 5000, "Wmax": 165, "delta_max": 2.22},
    "Aluminio": {"W0": 12.9, "W1": 85.47, "W2": 1414, "Wmax": 350, "delta_max": 1.5},
    "Cobre": {"W0": 10, "W1": 25, "W2": 5000, "Wmax": 175, "delta_max": 2.25},
}

# Funciones auxiliares
def velocity_verlet(x, v, a, dt):
    x_new = x + v * dt + 0.5 * a * dt**2
    a_new = acceleration(x_new)
    v_new = v + 0.5 * (a + a_new) * dt
    return x_new, v_new, a_new

def acceleration(x):
    k = 1.0  # Constante del resorte para la fuerza de restauración
    return -k * x  # Fuerza restauradora tipo oscilador armónico

def calcular_probabilidad_emision(W, material):
    W0 = materiales[material]["W0"]
    W1 = materiales[material]["W1"]
    W2 = materiales[material]["W2"]
    Wmax = materiales[material]["Wmax"]
    delta_max = materiales[material]["delta_max"]
    
    gamma0 = calcular_gamma0(W0, W1, Wmax, delta_max)
    gamma2 = (W2 - W0) / (Wmax - W0)
    s = np.log(delta_max) / np.log(gamma2 / 3.6)
    r = delta_max * 3.6**s
    
    delta = delta_max * (W / Wmax) * np.exp(1 - W / Wmax) * (1 / (1 + (W / W0)**gamma2))**(s * (W / Wmax))
    return delta

def calcular_gamma0(W0, W1, Wmax, delta_max):
    k1, k2 = 0.56, 0.25
    gamma0 = (Wmax / (W1 - W0)) * ((Wmax - W0) / (Wmax - W1))**delta_max
    return gamma0

# Solicitud de parámetros al usuario
material = input("Ingrese el material de la placa (Plata, Aluminio, Cobre): ")
f = float(input("Ingrese la frecuencia f en Hz: "))
v0_ev = float(input("Ingrese la velocidad inicial v0 en eV: "))
x0 = float(input("Ingrese la posición inicial x0 en m: "))
V = float(input("Ingrese el voltaje en V: "))
d = float(input("Ingrese la distancia d en m: "))
n = int(input("Ingrese el número de ciclos n: "))
T = float(input("Ingrese el periodo T en s: "))

# Conversión de unidades y cálculo de parámetros derivados
v0 = np.sqrt(2 * v0_ev * e / m)  # Convertir de eV a m/s
omega = 2 * np.pi * f
dt = T / 1000  # Paso de tiempo arbitrario

# Inicialización de variables para Velocity Verlet
t = 0
x = x0
v = v0
a = acceleration(x)
trayectoria = []

while t < n * T:
    x, v, a = velocity_verlet(x, v, a, dt)
    t += dt
    trayectoria.append((t, x))
    if abs(x) > d:
        v = -v  # Rebote
        W = np.abs(0.5 * m * v**2 / e)
        delta_emision = calcular_probabilidad_emision(W, material)
        print(f"Probabilidad de emisión: {delta_emision:.4f}")

# Graficar la trayectoria de la posición
tiempos = [p[0] for p in trayectoria]
posiciones = [p[1] for p in trayectoria]

plt.figure(figsize=(10, 5))
plt.plot(tiempos, posiciones, label="Posición (Verlet)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Posición (m)")
plt.title("Trayectoria de la Partícula con Método Verlet")
plt.legend()
plt.grid(True)
plt.show()

print(f"Posición final: {x} m")
print(f"Velocidad final: {v} m/s")
