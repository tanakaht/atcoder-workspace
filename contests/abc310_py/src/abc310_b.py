import sys
import math

N, M = map(int, input().split())
PCF = [list(map(int, input().split())) for _ in range(N)]
for i, (p, c, *f) in enumerate(PCF):
    f = set(f)
    for j, (p_, c_, *f_) in enumerate(PCF):
        if i==j:
            continue
        flg = True
        for x in f_:
            flg = flg and (x in f)
        if flg and p<p_:
            print("Yes")
            sys.exit()
        if flg and p==p_ and len(f)>len(f_):
            print("Yes")
            sys.exit()
print("No")
