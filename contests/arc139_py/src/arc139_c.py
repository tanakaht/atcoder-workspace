import sys
import math

N, M = map(int, input().split())
# 3x+y=k1, x+3y=k2とする, x,yを出す
def ks2xy(k1, k2):
    x=(3*k1-k2)//8
    y=(-k1+3*k2)//8
    return (x, y)

def k1tok2s(k1):
    # 最小は(1, x) or
    pass

# 小さい方から取れば良い?一旦お試し
cur = {i: (3*i)%8-32 for i in range(8)}
anss = []
for k1 in range(4, 3*N+M+1):
    k2 = cur[k1%8]+8
    cnt = 0
    while True:
        cnt += 1
        if cnt>100:
            break
        x, y = ks2xy(k1, k2)
        if not (1<=x<=N and 1<=y<=M):
            k2 += 8
        else:
            break
    if (1<=x<=N and 1<=y<=M):
        anss.append((x, y))
        cur[k1%8] = k2
print(len(anss))
for x, y in anss:
    print(x, y)
