import sys
import math

Ha, Wa = map(int, input().split())
A = [list(map(lambda x: 1 if x=="#" else 0, input()))+[0]*(30-Wa) for _ in range(Ha)]+[[0]*30 for _ in range(30)]
Hb, Wb = map(int, input().split())
B = [list(map(lambda x: 1 if x=="#" else 0, input()))+[0]*(30-Wb) for _ in range(Hb)]+[[0]*30 for _ in range(30)]
Hx, Wx = map(int, input().split())
X = [list(map(lambda x: 1 if x=="#" else 0, input())) for _ in range(Hx)]
A_set = set()
B_set = set()
for h in range(Ha):
    for w in range(Wa):
        if A[h][w]:
            A_set.add((h, w))
for h in range(Hb):
    for w in range(Wb):
        if B[h][w]:
            B_set.add((h, w))

cnt = 0
for a in A:
    cnt += sum(a)
for b in B:
    cnt += sum(b)

hx, wx = 9, 9
X_set = set()
for h in range(Hx):
    for w in range(Wx):
        if X[h][w]:
            X_set.add((h, w))

for ha in range(19):
    for wa in range(19):
        for hb in range(19):
            for wb in range(19):
                flg = True
                tmpcnt = 0
                for h in range(hx, hx+Hx):
                    for w in range(wx, wx+Wx):
                        black_cnt = ((h-ha, w-wa) in A_set) + ((h-hb, w-wb) in B_set)
                        flg = flg and ((h-hx, w-wx) in X_set)==(black_cnt >= 1)
                        tmpcnt += black_cnt
                if flg and (tmpcnt==cnt):
                    print("Yes")
                    sys.exit(0)

print("No")
