import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import sympy
import math
from concurrent.futures import ProcessPoolExecutor

def calc_pi(roop_count, pi):
    """素数かどうか判断するための整数を生成
    
    Args:
        roop_count (int ):  main関数のwhile文をループした回数
        pi        (float):  円周率...小数点以下の桁数は(17 + roop_count)
    
    Returns:
        int: 素数判定をするための整数
    """
    
    # 小数点以下を取得
    prime_candidate = str(pi).split('.')[1]
    # while文をループした回数分の先頭桁数と丸めさを避けるために取得した余分の1桁を無視
    prime_candidate = int(prime_candidate[roop_count:-1])

    return prime_candidate

# def is_prime(prime_candidate):
#     """素数判定

#     Args:
#         prime_candidate   (int): 素数候補の整数

#     Returns:
#         True or False (boolean):prime_candidateが素数かどうかをブール値で返す
#     """
#     i = 2
    
#     while i * i <= prime_candidate:
#         if prime_candidate % i == 0:
#             return False
        
#         i += 1

#     return True
            

if __name__ == '__main__':
    
    # 円周率の桁数(整数3の1桁と小数点以下の末尾の丸め誤差を避けるために小数点以下は17桁を取得する)
    PI_DIGIT = 18
    # 素数を取得したい回数
    MAX_PRIME_NUM = 2024
    # 現在見つかった素数の個数
    prime_count = 0
    # while文をループした回数
    roop_count = 0
    # 最終的な答え(2024個目の素数)
    answer = 0
    
    while prime_count != MAX_PRIME_NUM:
        
        pi = sympy.pi.evalf(PI_DIGIT + roop_count)
        prime_candidate = calc_pi(roop_count, pi)
        
        # if is_prime(prime_candidate):
        if sympy.ntheory.isprime(prime_candidate):
            prime_count += 1
            
            if prime_count == MAX_PRIME_NUM:
                answer = prime_candidate
        
        roop_count += 1
        
    print(f'{MAX_PRIME_NUM}個目の素数は{answer}')