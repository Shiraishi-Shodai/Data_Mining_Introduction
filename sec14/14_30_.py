import sympy
import numpy as np

data_ex=np.array([
    [1.0,3.1],
    [2.0,5.1],
    [3.0,6.8],
    [4.0,8.9],
    [5.0,11.5],
    [6.0,13.8],
])

E=0
for li in data_ex:
    sympy.var('a b')
    E += (li[1]-(a*li[0]+b))**2

print(E)
print(sympy.expand(E)) #式の展開を確認
print(sympy.diff(E,a)) # aでの偏微分を確認
print(sympy.diff(E,b)) # bでの偏微分を確認

#連立方程式を解く
a_m , b_m = sympy.solve([sympy.diff(E,b), sympy.diff(E,a)], [a, b])
print(sympy.solve([sympy.diff(E,b), sympy.diff(E,a)], [a, b]))
print(f"a= {a_m} , b ={b_m}")
print(E.subs([(a, 2.13714285714286), (b, 0.72)]))