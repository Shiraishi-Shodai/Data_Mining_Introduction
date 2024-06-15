
from sympy import isprime
from my_prime import isprime2,get_PI,get_PI2
import time
start = time.time()
pi = str(get_PI2(10000))
print(pi[0:100])

print(f"{time.time()-start:.3f}sec")