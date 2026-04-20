import numpy as np
import matplotlib.pyplot as plt
import math
import time


# Изначальная функция f(x) = sin(x)
def f(x):
    return np.sin(x)


# Функция для аппроксимации g(x) = sin(x) + 1
def g(x):
    return np.sin(x) + 1.0


# g_n(x)
# использую p.floor(n * gx) / n, так как если i/n<=g(x)<(i+1)/n,
# то np.floor(n * gx) / n = i/n
def g_n_values(x, n):
    gx = g(x)
    return np.where(gx >= n, n, np.floor(n * gx) / n)


# \tilde g_n(x) = max(g_1(x), ..., g_n(x))
def g_tilde(x, n):
    current_max = 0
    for k in range(1, n + 1):
        current_max = np.maximum(current_max, g_n_values(x, k))
    return current_max


# f_n(x) = \tilde g_n(x) - 1
def f_n(x, n):
    return g_tilde(x, n) - 1.0


# Точные значения интегралов
I_exact = 1.0 - math.cos(4.0)

atoms = [0, 1, 2, 3, 4]
I_st_exact = sum(math.sin(k) for k in atoms)

print("Точное значение интеграла Лебега:")
print(f"I = {I_exact:.10f}")
print()

print("Точное значение интеграла Лебега–Стилтьеса:")
print(f"I_st = {I_st_exact:.10f}")
print()

# 2.1 Графики
x_plot = np.linspace(0.0, 4.0, 4000)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, f(x_plot), label='f(x)=sin(x)', linewidth=2)

for n in [1, 2, 5, 10]:
    plt.plot(x_plot, f_n(x_plot, n), label=fr'$f_{{{n}}}(x)$')

plt.title('Графики функции f(x)=sin(x) и простых функций f_n(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()

# 2.2 Интеграл Лебега от f_n
x_int = np.linspace(0.0, 4.0, 100001)

print("Интеграл Лебега от f_n по E=[0,4]:")
for n in [10, 100, 1000]:
    start = time.perf_counter()

    y = f_n(x_int, n)
    I_num = np.trapezoid(y, x_int)

    workTime = time.perf_counter() - start

    print(f"n = {n}")
    print(f"  Численное значение     = {I_num:.10f}")
    print(f"  Точное значение        = {I_exact:.10f}")
    print(f"  Абсолютная погрешность = {abs(I_num - I_exact):.10f}")
    print(f"  Время работы           = {workTime:.6f} сек")
    print()

# 2.3 Интеграл Лебега–Стилтьеса от f_n
print("Интеграл Лебега–Стилтьеса от f_n по E=[0,4]:")
for n in [10, 100, 1000]:
    start = time.perf_counter()

    I_st_num = sum(f_n(a, n) for a in atoms)

    workTime = time.perf_counter() - start

    print(f"n = {n}")
    print(f"  Численное значение     = {I_st_num:.10f}")
    print(f"  Точное значение        = {I_st_exact:.10f}")
    print(f"  Абсолютная погрешность = {abs(I_st_num - I_st_exact):.10f}")
    print(f"  Время работы           = {workTime:.6f} сек")
    print()
