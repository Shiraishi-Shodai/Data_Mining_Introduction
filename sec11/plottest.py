import matplotlib.pyplot as plt
import numpy as np

fig1 = plt.figure()  # まず、figureを作る。

ax1 = fig1.add_subplot(111) # 次に、fig1 にsubplot（３行３列の１番）を加える。
x = np.linspace(-3, 4, 100)
ax1.plot(x, x**2) # グラフax1を描画する。
plt.savefig('plottest1.png')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111) # 次に、fig1 にsubplot（３行３列の７番）を加える。
icecream_ratio = [10, 20, 50, 40]
ax2.pie(icecream_ratio,autopct='%.2f%%') # グラフax2を描画する。
plt.savefig('plottest2.png')

# ax3 = fig1.add_subplot(339) # 次に、fig1 にsubplot（３行３列の９番）を加える。
# ax3.hist([2, 3, 5]) # グラフax3を描画する。

# X =[1.0, 2.0, 3.0, 4.1, 4.9 , 6.0 , 7.1]
# y =[2.0, 4.0, 6.0, 7.9, 10.1, 11.9, 14.3]
# ax4 = fig1.add_subplot(335) # 次に、fig1 にsubplot（３行３列の５番）を加える。
# ax4.scatter(X,y) # グラフax4を描画する。

# # plt.show() # 全体を描画する。
# plt.savefig('plottest.png')