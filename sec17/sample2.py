import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
A = 1
B = 0

# x の範囲
x = np.linspace(-2, 2, 400)

# 関数の定義
y1 = 10**(A * x + B)
log_y1 = np.log10(y1)

y2 = 10**(-A * x + B) + 65.0
log_y2 = np.log10(y2)

# グラフのプロット
plt.figure(figsize=(10, 6))

# 1. log10(y) = A * x + B のプロット
plt.plot(x, log_y1, label=r'$\log_{10}(y) = A \cdot x + B$', color='blue')

# 2. log10(y) = log10(10^(-A * x + B) + 65.0) のプロット
plt.plot(x, log_y2, label=r'$\log_{10}(y) = \log_{10}(10^{-A \cdot x + B} + 65.0)$', color='red')

# グラフの設定
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\log_{10}(y)$')
plt.title(r'Comparison of $\log_{10}(y) = A \cdot x + B$ and $\log_{10}(y) = \log_{10}(10^{-A \cdot x + B} + 65.0)$')

plt.show()
