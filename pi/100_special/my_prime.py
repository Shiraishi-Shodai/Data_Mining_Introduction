def isprime2( n ):
    if n==1:
        return False
    if n==2:
        return True
    elif n >= 3 :
        for i in range( 3, int( n**0.5) +1,2)  : 
            if n % i == 0 :
                return False
        return True

import math
from decimal import Decimal, getcontext

def chudnovsky_algorithm():
    C = 426880 * Decimal(math.sqrt(10005))
    K = 6
    M = 1
    X = 1
    L = 13591409
    S = L

    for i in range(1, 5001):  # 計算の繰り返し回数（精度に応じて調整可能）
        M = (K**3 - 16*K) * M // i**3
        L += 545140134
        X *= -262537412640768000
        S += Decimal(M * L) / X
        K += 12

    pi = C / S
    return pi

def get_PI(n):
    # 精度を設定（例えば、1000桁）
    getcontext().prec = n #100002
    return  chudnovsky_algorithm() 
import sys
from decimal import *

def Gauss_Legendre():
    a = Decimal(1)
    b = Decimal(1) / Decimal(2).sqrt()
    t = Decimal(1) / Decimal(4)
    p = Decimal(1)
    r = Decimal(0)
    rn = Decimal(3)
    while r != rn:
        r = rn
        an = (a + b) / 2
        bn = (a * b).sqrt()
        tn = t - p * (a - an) * (a - an)
        pn = 2 * p
        rn = ((a + b) * (a + b)) / (4 * t)
        a = an
        b = bn
        t = tn
        p = pn
    return rn

def get_PI2(n):
    # 精度を設定（例えば、1000桁）
    getcontext().prec = n #100002
    return  Gauss_Legendre() 

