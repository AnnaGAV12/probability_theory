# 1. Даны значения зарплат из выборки выпускников: 100, 80, 75, 77, 89, 33, 45, 25, 65, 17, 30, 24, 57, 55, 70, 75, 65, 84, 90, 150. Посчитать (желательно без использования статистических методов наподобие std, var, mean) среднее арифметическое, среднее квадратичное отклонение, смещенную и несмещенную оценки дисперсий для данной выборки.Даны значения зарплат из выборки выпускников: 100, 80, 75, 77, 89, 33, 45, 25, 65, 17, 30, 24, 57, 55, 70, 75, 65, 84, 90, 150. Посчитать (желательно без использования статистических методов наподобие std, var, mean) среднее арифметическое, среднее квадратичное отклонение, смещенную и несмещенную оценки дисперсий для данной выборки.

# salaries = [100, 80, 75, 77, 89, 33, 45, 25, 65, 17, 30, 24, 57, 55, 70, 75, 65, 84, 90, 150]

# mean = sum(salaries) / len(salaries)
# print("Среднее арифметическое:", mean)

# std_dev = (sum((x - mean) ** 2 for x in salaries) / len(salaries)) ** 0.5
# print("Среднее квадратичное отклонение (стандартное отклонение):", std_dev)

# var_biased = sum((x - mean) ** 2 for x in salaries) / len(salaries)
# print("Смещенная оценка дисперсии:", var_biased)

# var_unbiased = sum((x - mean) ** 2 for x in salaries) / (len(salaries) - 1)
# print("Несмещенная оценка дисперсии:", var_unbiased)

# 2. В первом ящике находится 8 мячей, из которых 5 - белые. Во втором ящике - 12 мячей, из которых 5 белых. Из первого ящика вытаскивают случайным образом два мяча, из второго - 4. Какова вероятность того, что 3 мяча белые?
# import math

# Вероятность вытащить 2 белых мяча из первого ящика
# prob_white_from_box1 = math.comb(5, 2) / math.comb(8, 2)

# Вероятность вытащить 1 белый мяч из второго ящика
# prob_white_from_box2 = math.comb(5, 1) / math.comb(12, 4)

# Вероятность события A: 2 белых мяча из первого ящика и 1 белый мяч из второго
# prob_A = prob_white_from_box1 * prob_white_from_box2

# Вероятность вытащить 1 белый мяч из первого ящика
# prob_white_from_box1 = math.comb(5, 1) / math.comb(8, 2)

# Вероятность вытащить 2 белых мяча из второго ящика
# prob_white_from_box2 = math.comb(5, 2) / math.comb(12, 4)

# Вероятность события B: 1 белый мяч из первого ящика и 2 белых мяча из второго
# prob_B = prob_white_from_box1 * prob_white_from_box2

# Общая вероятность
# total_prob = prob_A + prob_B

# print("Вероятность того, что 3 мяча белые:", total_prob)

# 3.На соревновании по биатлону один из трех спортсменов стреляет и попадает в мишень. Вероятность попадания для первого спортсмена равна 0.9, для второго — 0.8, для третьего — 0.6. Найти вероятность того, что выстрел произведен: a). первым спортсменом б). вторым спортсменом в). третьим спортсменом.
# # Вероятности попадания для каждого спортсмена
# prob_A1 = 0.9
# prob_A2 = 0.8
# prob_A3 = 0.6
# # Вероятность для каждого спортсмена
# prob_A = 1/3
#
# #Вероятность попадания в мишень
# prob_B = prob_A1 * prob_A + prob_A2 * prob_A + prob_A3 * prob_A
#
# # Вероятность попадания в мишень, если выстрел сделал первый спортсмен
# prob_B_given_A1 = prob_A1 * prob_A / prob_B
#
# # Вероятность попадания в мишень, если выстрел сделал второй спортсмен
# prob_B_given_A2 = prob_A2 * prob_A / prob_B
#
# # Вероятность попадания в мишень, если выстрел сделал третий спортсмен
# prob_B_given_A3 = prob_A3 * prob_A / prob_B
#
# print("Вероятность попадания в мишень, если выстрел сделал первый спортсмен:", prob_B_given_A1)
# print("Вероятность попадания в мишень, если выстрел сделал второй спортсмен:", prob_B_given_A2)
# print("Вероятность попадания в мишень, если выстрел сделал третий спортсмен:", prob_B_given_A3)

# 4. В университет на факультеты A и B поступило равное количество студентов, а на факультет C студентов поступило столько же, сколько на A и B вместе. Вероятность того, что студент факультета A сдаст первую сессию, равна 0.8. Для студента факультета B эта вероятность равна 0.7, а для студента факультета C - 0.9. Студент сдал первую сессию. Какова вероятность, что он учится: a). на факультете A б). на факультете B в). на факультете C?
# # Вероятности сдачи сессии для каждого факультета
# prob_pass_A = 0.8
# prob_pass_B = 0.7
# prob_pass_C = 0.9
#
# # Вероятности поступления на каждый факультет
# prob_A = 1/4
# prob_B = 1/4
# prob_C = 2/4
#
# # вероятность сдачи сессии
# prob_pass = prob_pass_A * prob_A + prob_pass_B * prob_B + prob_pass_C * prob_C
#
# # Вероятность того, что студент с факультета A сдал сессию
# prob_A_given_pass = prob_pass_A * prob_A / prob_pass
#
# # Вероятность того, что студент с факультета B сдал сессию
# prob_B_given_pass = prob_pass_B * prob_B / prob_pass
#
# # Вероятность того, что студент с факультета C сдал сессию
# prob_C_given_pass = prob_pass_C * prob_C / prob_pass
#
# print("Вероятность того, что студент учится на факультете A:", prob_A_given_pass)
# print("Вероятность того, что студент учится на факультете B:", prob_B_given_pass)
# print("Вероятность того, что студент учится на факультете C:", prob_C_given_pass)

# 5. Устройство состоит из трех деталей. Для первой детали вероятность выйти из строя в первый месяц равна 0.1, для второй - 0.2, для третьей - 0.25. Какова вероятность того, что в первый месяц выйдут из строя: а). все детали б). только две детали в). хотя бы одна деталь г). от одной до двух деталей?
# # Вероятности выхода из строя для каждой детали
# prob_fail_1 = 0.1
# prob_fail_2 = 0.2
# prob_fail_3 = 0.25
#
# # Вероятность того, что все детали выйдут из строя
# prob_all_fail = prob_fail_1 * prob_fail_2 * prob_fail_3
#
# # Вероятность того, что выйдут из строя только две детали
# prob_only_two_fail = (prob_fail_1 * prob_fail_2 * (1 - prob_fail_3) +
#                       prob_fail_1 * (1 - prob_fail_2) * prob_fail_3 +
#                       (1 - prob_fail_1) * prob_fail_2 * prob_fail_3)
#
# # Вероятность того, что хотя бы одна деталь выйдет из строя
# prob_at_least_one_fail = 1 - (1 - prob_fail_1) * (1 - prob_fail_2) * (1 - prob_fail_3)
#
# # Вероятность того, что выйдет из строя от одной до двух деталей
# prob_one_to_two_fail = prob_at_least_one_fail - prob_all_fail
#
# print("Вероятность того, что все детали выйдут из строя:", prob_all_fail)
# print("Вероятность того, что выйдут из строя только две детали:", prob_only_two_fail)
# print("Вероятность того, что хотя бы одна деталь выйдет из строя:", prob_at_least_one_fail)
# print("Вероятность того, что выйдет из строя от одной до двух деталей:", prob_one_to_two_fail)