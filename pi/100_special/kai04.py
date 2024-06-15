
from sympy import isprime
from my_prime import isprime2,get_PI,get_PI2
import time
start = time.time()
pi = get_PI2(1000000)
prime_th = 0
print(f"{time.time()-start:.3f}sec")
for i in range(0,10000000):
    t_num = int(pi[i:i+16]) 
    if isprime2(t_num):
        prime_th +=1
        print(f"{i}: {t_num} is {prime_th} th prime")
    
    if prime_th==2024:
        break

print(f"{time.time()-start:.3f}sec")