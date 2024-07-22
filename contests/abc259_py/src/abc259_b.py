import sys
import math
import numpy as np

a, b, d = map(int, input().split())
theta = d*math.pi*2/360
M = np.array([
    [math.cos(theta), -math.sin(theta)],
    [math.sin(theta), math.cos(theta)]
    ])
x = M@np.array([a, b])
print(*x)
