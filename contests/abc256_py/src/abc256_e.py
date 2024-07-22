import sys
import math

N = int(input())
X = list(map(lambda x: int(x)-1, input().split()))
C = list(map(int, input().split()))
ans = 0
appeared = set()
for i in range(N):
    if i in appeared:
        continue
    # ループを見つける
    cur_app = set()
    x = i
    appeared.add(x)
    cur_app.add(x)
    loop_s = None
    while True:
        if X[x] in cur_app:
            loop_s = X[x]
            break
        if X[x] in appeared:
            break
        appeared.add(x)
        cur_app.add(x)
        x = X[x]
    if loop_s is not None:
        minc = C[loop_s]
        x = X[loop_s]
        while x != loop_s:
            minc = min(minc, C[x])
            x = X[x]
        ans += minc
print(ans)
