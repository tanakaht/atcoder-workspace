import sys

input = sys.stdin.readline
T = int(input())
for caseid in range(1, T+1):
    N, M, Q = map(int, input().split())
    ABC = [list(map(int, input().split())) for _ in range(M)]
    XY = [list(map(int, input().split())) for _ in range(Q)]
    anss = [0]*Q
    ans = 0
    g =
    for i in range(N):
        if S[i]=='F':
            pass
        elif S[i]=='O':
            if pre is None or pre=='O':
                pass
            else:
                ans += 1
            pre = S[i]
        elif S[i]=='X':
            if pre is None or pre=='X':
                pass
            else:
                ans += 1
            pre = S[i]
    print(f'Case #{caseid}: {ans}')
