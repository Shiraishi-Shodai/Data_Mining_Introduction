from decimal import *
def get_PI_basel(n,prec=10000):
    getcontext().prec = prec
    sum = 0
    for i in range(1,n+1):
        sum = sum + 1/i**2
    
    return (sum*6)**0.5

print(get_PI_basel(10000))