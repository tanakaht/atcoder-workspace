import sys
import math

N, T, M = map(int, input().split())
AB = [list(map(lambda x: int(x)-1, input().split())) for _ in range(M)]
teams = []

def dfs(i, teams):
    if i==N:
        if len(teams)!=T:
            return 0
        else:
            belongs = [0]*N
            for team in teams:
                for x in team:
                    belongs[x] = team
            for a, b in AB:
                if belongs[a]==belongs[b]:
                    return 0
            return 1
    else:
        ret = 0
        for j in range(len(teams)+1):
            if j==len(teams):
                teams.append([i])
            else:
                teams[j].append(i)
            ret += dfs(i+1, teams)
            teams[j].remove(i)
            if not teams[j]:
                teams.pop(j)
        return ret
print(dfs(0, []))
