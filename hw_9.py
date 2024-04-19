# 1. Даны значения величины заработной платы заемщиков банка (zp) и значения их поведенческого кредитного скоринга (ks):
# zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110], ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832].
# Используя математические операции, посчитать коэффициенты линейной регрессии, приняв за X заработную плату
# (то есть, zp - признак), а за y - значения скорингового балла (то есть, ks - целевая переменная).
# Произвести расчет как с использованием intercept, так и без.

import numpy as np

zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

# Расчет с использованием intercept
# x_mean = np.mean(zp)
# y_mean = np.mean(ks)
# b_with_intercept = np.sum((zp - x_mean) * (ks - y_mean)) / np.sum((zp - x_mean) ** 2)
# a_with_intercept = y_mean - b_with_intercept * x_mean

# Расчет без использования intercept
# b_without_intercept = np.sum(zp * ks) / np.sum(zp ** 2)

# print("Свободный член (intercept):", a_with_intercept)
# print("Коэффициент наклона (slope):", b_with_intercept)
# print("\nКоэффициент линейной регрессии без использования intercept:")
# print("Коэффициент наклона (slope):", b_without_intercept)

# 2. Посчитать коэффициент линейной регрессии при заработной плате (zp), используя градиентный спуск (без intercept).
# def mse_loss(b, x, y):
#    return np.mean((y - b * x) ** 2)

# def gradient(b, x, y):
#    return -2 * np.mean(x * (y - b * x))

# def gradient_descent(x, y, learning_rate=0.0001, epochs=1000):
#    b = 0
#    for _ in range(epochs):
#        grad = gradient(b, x, y)
#        b -= learning_rate * grad
#    return b

# b_gradient_descent = gradient_descent(zp, ks)

# print("Коэффициент линейной регрессии (slope) с использованием градиентного спуска (без intercept):", b_gradient_descent)

# 3. Произвести вычисления как в пункте 2, но с вычислением intercept. Учесть, что изменение коэффициентов должно производиться
# на каждом шаге одновременно (то есть изменение одного коэффициента не должно влиять на изменение другого во время одной итерации).

def mse_loss(b0, b1, x, y):
    return np.mean((y - (b0 * x + b1)) ** 2)

def gradient(b0, b1, x, y):
    grad_b0 = -2 * np.mean(x * (y - (b0 * x + b1)))
    grad_b1 = -2 * np.mean(y - (b0 * x + b1))
    return grad_b0, grad_b1

def gradient_descent_with_intercept(x, y, learning_rate=0.0001, epochs=1000):
    b0 = 0  # Начальное значение коэффициента наклона
    b1 = 0  # Начальное значение intercept
    for _ in range(epochs):
        grad_b0, grad_b1 = gradient(b0, b1, x, y)
        b0 -= learning_rate * grad_b0  # Обновление коэффициента наклона
        b1 -= learning_rate * grad_b1  # Обновление intercept
    return b0, b1

b0_gradient_descent, b1_gradient_descent = gradient_descent_with_intercept(zp, ks)

print("Коэффициенты линейной регрессии (slope и intercept) с использованием градиентного спуска:")
print("Свободный член (intercept):", b1_gradient_descent)
print("Коэффициент наклона (slope):", b0_gradient_descent)