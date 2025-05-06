import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ------------------------------- #
# Definir el modelo
# ------------------------------- #
def modeloFeda1(t, x):
    # Parámetros
    So = 4      # g/L
    Xo = 2      # g/L
    Um = 0.1   # 1/día
    D = 0.029   # 1/día
    Kd = 0.02   # 1/día
    Ks = 300    # g/L
    Ki = 0.1    # g/L

    Yx = 0.82   # g/g
    Ksx = 0.983 # g/g
    Kmx = 0.4   # g/g
    Ys = 4.35   # g/g

    Ymet = 1.2  # g/g
    YCo2 = 0.4  # g/g
    YH2 = 0.005  # g/g

    # Variables
    X = x[0]  # Biomasa
    S = x[1]  # Sustrato
    CO2 = x[2]
    Met = x[3]
    H2 = x[4]

    # Prevenir división por cero
    if S == 0:
        S = 1e-6

    # Velocidad específica de crecimiento con inhibición
    mu = Um / (1 + Ks/S + S/Ki)

    dxdt = np.zeros(5)

    # Ecuaciones diferenciales
    dxdt[0] = D * (Xo - X) + mu * X - Kd * X
    dxdt[1] = D * (So - S) - (mu * X)/Yx - Ksx * X * mu - Kmx * X * mu - (1/Ys) * (CO2 + Met + H2)
    dxdt[2] = YCo2 * mu * X
    dxdt[3] = Ymet * mu * X
    dxdt[4] = YH2 * mu * X

    return dxdt

# ------------------------------- #
# Resolver el sistema
# -------------------------------#
t_span = (0, 80)
t_eval = np.linspace(*t_span, 500)
x0 = [23, 20, 0, 0, 0]

sol = solve_ivp(modeloFeda1, t_span, x0, t_eval=t_eval)

# -------------------------------#
# Graficar resultados
# -------------------------------#
t = sol.t
x = sol.y

plt.figure(figsize=(10,6))
plt.plot(t, x[2], 'g', label='CO2', linewidth=2)
plt.plot(t, x[3], 'y', label='Metano', linewidth=2)
plt.plot(t, x[4], 'b', label='Hidrógeno', linewidth=2)

# (Opcional)
# plt.plot(t, x[0], 'm--', label='Biomasa')
# plt.plot(t, x[1], 'c--', label='Sustrato')

plt.xlabel('Tiempo (días)')
plt.ylabel('Concentración (g/L)')
plt.title('Producción de CO₂, Metano e Hidrógeno')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
