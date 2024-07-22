import sys
import math

T = int(input())
for _ in range(T):
    N, K = map(int, input().split())
    S = input()
    S2 = S.replace("?", "1")
    cnt1 = S.count("1")
    S2 = list(S.replace("?", "1"))
    S = list(S)
    if cnt1==0:
        maxcnt = 0
        cnt = 0
        for i in range(N):
            if S2[i]=="1":
                cnt += 1
                maxcnt = max(maxcnt, cnt)
            else:
                cnt = 0
        if maxcnt == K:
            print("Yes")
        else:
            print("No")
    elif cnt1==1:
        maxcnt = 0
        hasone = False
        cnt = 0
        for i in range(N):
            if S2[i]=="1":
                if S[i]=="1":
                    hasone = True
                cnt += 1
                if hasone:
                    maxcnt = max(maxcnt, cnt)
            else:
                cnt = 0
                hasone = False
        if maxcnt == K:
            print("Yes")
        else:
            print("No")
    else:
        l, r = math.inf, -math.inf
        for i in range(N):
            if S[i]=="1":
                l = min(l, i)
                r = max(r, i)
        if "".join(S2[l:r+1]).count("0")!=0:
            print("No")
        else:
            if r-l==+1==K:
                print("Yes")
            else:
                maxcnt = 0
                hasone = False
                cnt = 0
                for i in range(N):
                    if S2[i]=="1":
                        if S[i]=="1":
                            hasone = True
                        cnt += 1
                        if hasone:
                            maxcnt = max(maxcnt, cnt)
                    else:
                        cnt = 0
                        hasone = False
                if maxcnt == K:
                    print("Yes")
                else:
                    print("No")
