import matplotlib.pyplot as plt
import numpy as np

data_ex =[
    [ 1.0 , 3.1 ] ,
    [ 2.0 , 5.1 ] ,
    [ 3.0 , 6.8 ] ,
    [ 4.0 , 8.9 ] ,
    [ 5.0 , 11.5 ] ,
    [ 6.0 , 13.8 ] ,
]
def cost_func(data_ex,a,b):
    E=0
    for li in data_ex:
        E += (li[1]-(a*li[0]+b))**2
    return E

def get_a_b(data):
	x_vec =[]
	y_vec =[]

	for vec in data:
		x_vec.append(vec[0])
		y_vec.append(vec[1])
	
	#平均
	x_mu = sum(x_vec)/len(x_vec)
	y_mu = sum(y_vec)/len(y_vec)
	
	Sxy = 0
	Sxx = 0
	for i in range(0,len(x_vec)) :
		Sxy = (x_vec[i]-x_mu)*(y_vec[i]-y_mu) + Sxy
	
	for i in range(0,len(x_vec)) :
		Sxx = (x_vec[i]-x_mu)*(x_vec[i]-x_mu) + Sxx
	
	# a,b は1次回帰直線の傾きと係数
	a = Sxy / Sxx
	b = y_mu - a* x_mu

	return [a,b]

a , b = get_a_b(data_ex)
E_min = cost_func(data_ex,a,b)
print(f'a= {a},b={b},E={E_min}')