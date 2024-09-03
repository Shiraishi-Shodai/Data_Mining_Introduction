import numpy as np

a = np.array([1, 1, 0.8])
b = np.array([0.45, 0.4, 0.6])

c = np.array([0.5, 0.4, 0.6])
d = np.array([-1.0, -0.79, -1.3])

e = np.array([-1.5, 1.4, 3.6])
f = np.array([-4.5, 2.9, 0.6])

def cos_sim(a, b):
    return a @ b /(np.linalg.norm(a, ord=2)) * (np.linalg.norm(b, ord=2))

print(cos_sim(a, b))
print(cos_sim(c, d))
print(cos_sim(e, f))