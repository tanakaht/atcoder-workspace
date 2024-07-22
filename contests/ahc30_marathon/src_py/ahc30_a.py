import random
# read prior information
line = input().split()
N = int(line[0])
M = int(line[1])
eps = float(line[2])
fields = []
allcnt = 0
for _ in range(M):
    line = input().split()
    ps = []
    allcnt += int(line[0])
    for i in range(int(line[0])):
        ps.append((int(line[2*i+1]), int(line[2*i+2])))
    fields.append(ps)

# drill every square
cnt = 0
has_oil = []
not_appeared = set()
for i in range(N):
    for j in range(N):
        not_appeared.add((i, j))
q = []
while True:
    if q:
        i, j = q.pop()
    else:
        i, j = random.choice(list(not_appeared))
        not_appeared.remove((i, j))
    print("q 1 {} {}".format(i, j))
    resp = int(input())
    if resp!=0:
        has_oil.append((i, j))
        cnt += resp
        for i_, j_ in [i+1, j], [i-1, j], [i, j+1], [i, j-1]:
            if 0<=i_<N and 0<=j_<N:
                if (i_, j_) in not_appeared:
                    q.append((i_, j_))
                    not_appeared.remove((i_, j_))
    if cnt == allcnt:
        break

print("a {} {}".format(len(has_oil), ' '.join(map(lambda x: "{} {}".format(x[0], x[1]), has_oil))))
resp = input()
assert resp == "1", f"resp: {resp}"
