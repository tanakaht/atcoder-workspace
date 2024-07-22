import sys
import math
from itertools import product

h1, h2, h3, w1, w2, w3 = map(int, input().split())
ans = 0
for x in product(range(1, 31), repeat=4):
    c = min(x[0], x[1], h1-x[0]-x[1],
            x[2], x[3], h2-x[2]-x[3],
            w1-x[0]-x[2], w2-x[1]-x[3], w3-(h1-x[0]-x[1])-( h2-x[2]-x[3])
    )
    ans += (c>=1) and ((w3-(h1-x[0]-x[1])-( h2-x[2]-x[3]))==(h3-( w1-x[0]-x[2])-(w2-x[1]-x[3])))
print(ans)
