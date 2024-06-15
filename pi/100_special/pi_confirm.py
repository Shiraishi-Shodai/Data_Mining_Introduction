from my_prime import isprime2,get_PI,get_PI2
import time
start = time.time()
digits = 520000
pi = ""
with open('pi.dat') as f:
    pi = f.read()
pi = pi.replace(' ', '').replace("\n", '').replace("\r", '')
pi = "3" + pi
pi2 = str(get_PI(digits)).replace('.', '')

#print(pi[0:digits])
#print(pi2[0:digits])
print(f"{time.time()-start:.3f}sec")
for i in range(0,max(len(pi),len(pi2))):
    if pi[0:i] == pi2[0:i]:
        #print(f"{i} => pass")
        pass
    else:
        print(f"i={i}th not same!")
        break

print(f"{time.time()-start:.3f}sec")