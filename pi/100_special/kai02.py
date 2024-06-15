from mpmath import mp
from sympy import isprime
import time
start = time.time()

mp.dps = 1000000 + 2  # 1000000桁+ "3."
pi = str(mp.pi)[2:]  # "3." を除く

prime_th = 0
for i in range(0,1000000):
    t_num = int(pi[i:i+16]) 
    if isprime(t_num):
        prime_th +=1
        print(f"{i}: {t_num} is {prime_th} th prime")
    
    if prime_th==2024:
        break
print(f"{time.time()-start:.3f}sec")