# Есть ли статистически значимые различия в росте дочерей?
# Ответ: нет

import numpy as np
import scipy.stats as stats


mothers = [172, 177, 158, 170, 178, 175, 164, 160, 169, 165]
daughters = [173, 175, 162, 174, 175, 168, 155, 170, 160, 163]

stats.ttest_rel(mothers, daughters)