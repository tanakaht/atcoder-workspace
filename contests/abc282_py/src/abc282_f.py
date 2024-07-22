import sys
import math

N = int(input())
# M
M = 12*N
print(M)
# l, r *M
lrs = []
for i in range(12):
    w = pow(2, i)
    for l in range(1, N+1):
        r = min(N, l+w-1)
        print(l, r)
        lrs.append((l, r))
Q = int(input())
for _ in range(Q):
    L, R = map(int, input().split())
    if L==R:
        ans = (L, L)
        print(*ans)
        assert lrs[ans[0]-1][0]==L and lrs[ans[1]-1][1]==R and lrs[ans[1]-1][0]<=lrs[ans[0]-1][1] and lrs[ans[0]-1][1]<=R and lrs[ans[1]-1][0]>=L
        # print(lrs[ans[0]-1], lrs[ans[1]-1])
        continue
    # a, b
    i = int(math.log2(R-L+1))
    w = pow(2, i)
    ans = N*i+L, N*i+R-w+1
    print(*ans)
    assert lrs[ans[0]-1][0]==L and lrs[ans[1]-1][1]==R and lrs[ans[1]-1][0]<=lrs[ans[0]-1][1] and lrs[ans[0]-1][1]<=R and lrs[ans[1]-1][0]>=L
    # print(lrs[ans[0]-1], lrs[ans[1]-1])
