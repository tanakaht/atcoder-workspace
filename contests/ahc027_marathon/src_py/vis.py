import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
sys.setrecursionlimit(1000000)
anss = []
for line in sys.stdin:
    if line == '\n':
        break
    anss.append(line)
DIR = {"R": (0, 1), "D": (1, 0), "L": (0, -1), "U": (-1, 0)}

# 文字列に基づいて動きを計算する関数
def get_movements(commands_list):
    for commands in commands_list:
        yield commands

# アニメーションの更新関数
def update(commands):
    cur = (0, 0)
    xdata, ydata = [0], [0]
    for t, x in enumerate(commands):
        di, dj = DIR[x]
        cur = (cur[0] + di, cur[1] + dj)
        xdata.append(cur[0]+t/len(commands)/10)
        ydata.append(cur[1]+t/len(commands)/10)
    line.set_data(xdata, ydata)
    return line,

# アニメーションの初期化関数
def init():
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    return line,


# アニメーションの準備
fig, ax = plt.subplots()
xdata, ydata = [], []
line, = ax.plot([], [], lw=2)

# アニメーションの開始
ani = FuncAnimation(fig, update, frames=get_movements(anss), init_func=init, blit=True, interval=500)

plt.show()
