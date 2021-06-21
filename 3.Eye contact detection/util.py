import math


def calculate_mcc(tp, fp, fn, tn):
    numerator = (tp * tn) - (fp * fn)  # 马修斯相关系数公式分子部分
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))  # 马修斯相关系数公式分母部分
    result = numerator / denominator
    return result
