import sys
import math
import itertools

def tentosu(idxs):
    ret = 0
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            if idxs[i] > idxs[j]:
                ret += 1
    return ret

H, W = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(H)]
B = [list(map(int, input().split())) for _ in range(H)]
ans = math.inf
for i_idxs in itertools.permutations(range(H), H):
    for j_idxs in itertools.permutations(range(W), W):
        is_ok = True
        for i in range(H):
            for j in range(W):
                if A[i][j] != B[i_idxs[i]][j_idxs[j]]:
                    is_ok = False
        if is_ok:
            tmpans = tentosu(i_idxs) + tentosu(j_idxs)
            if tmpans < ans:
                ans = tmpans
if ans == math.inf:
    print(-1)
else:
    print(ans)
