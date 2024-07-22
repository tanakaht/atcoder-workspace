import sys
import math

Ha, Wa = map(int, input().split())
A = [list(map(lambda x: x=="#", input()))+[False]*10 for _ in range(Ha)]+[[False*20] for _ in range(10)]
Hb, Wb = map(int, input().split())
B = [list(map(lambda x: x=="#", input()))+[False]*10 for _ in range(Hb)]+[[False*20] for _ in range(10)]
Hx, Wx = map(int, input().split())
X = [list(map(lambda x: x=="#", input())) for _ in range(Hx)]

for ha in range(20):
    for wa in range(20):
        for hb in range(20):
            for wb in range(20):
                for hx in range(10):
                    for wx in range(10):
                        if sorted([ha, hb. hx, wa, wb, wx])[1]!=0:
                            continue
                        flg = True
                        for h in range(hx, hx+Hx):
                            for w in range(wx, wx+Wx):
                                flg = flg and (X[h-hx][w-wx]==(A[h-ha][w-wb] or B[h-hb][w-wb]))
                        if flg:
                            print("Yes")
                            sys.exit(0)

print("No")
