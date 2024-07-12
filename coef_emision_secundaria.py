import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constantes
k1 = 0.56
k2 = 0.25
kv = 1

# Función para calcular Θ(γ0)
def theta(gamma):
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

# Energía primaria de electrones
E = np.linspace(0, 600, 1000)

# Graficar
plt.figure(figsize=(10, 6))

for material, params in materiales.items():
    delta = calculate_delta(E, **params)
    plt.plot(E, delta, label=material)

# Solicitar al usuario la energía primaria de electrones
energia_usuario = float(input("Ingrese la energía primaria de electrones en eV: "))

# Calcular y mostrar el coeficiente de emisión para la energía ingresada
for material, params in materiales.items():
    delta_usuario = calculate_delta(np.array([energia_usuario]), **params)[0]
    plt.plot(energia_usuario, delta_usuario, 'o', label=f'{material} a {energia_usuario} eV: {delta_usuario:.4f}')

plt.xlabel('Energía Primaria de Electrones (eV)')
plt.ylabel('Coeficiente de Emisión Secundaria')
plt.title('Coeficiente de Emisión Secundaria para Diferentes Materiales')
plt.grid(True)
plt.legend()
plt.show()
