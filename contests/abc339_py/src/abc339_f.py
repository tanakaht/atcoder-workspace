import sys
import math

N = int(input())
MODS = [998244353, 1000000007, 1000000009, 1000000021, 1000000033, 1000000087, 1000000093, 1000000097, 1000000103, 1000000123, 1000000181, 1000000207, 1000000223, 1000000241, 1000000271, 1000000289, 1000000297, 1000000321, 1000000349, 1000000363, 1000000403, 1000000409, 1000000411, 1000000427, 1000000433, 1000000439, 1000000447, 1000000453, 1000000459, 1000000483, 1000000513, 1000000531, 1000000579, 1000000607, 1000000613, 1000000637, 1000000663, 1000000711, 1000000753, 1000000787, 1000000801, 1000000829, 1000000861, 1000000871, 1000000891, 1000000901, 1000000919, 1000000931, 1000000933, 1000000993, 1000001011, 1000001021, 1000001053, 1000001087, 1000001099, 1000001137, 1000001161, 1000001203, 1000001213, 1000001237, 1000001263, 1000001269, 1000001273, 1000001279, 1000001311, 1000001329, 1000001333, 1000001351, 1000001371, 1000001393, 1000001413, 1000001447, 1000001449, 1000001491, 1000001501, 1000001531, 1000001537, 1000001539, 1000001581, 1000001617, 1000001621, 1000001633, 1000001647, 1000001663, 1000001677, 1000001699, 1000001759, 1000001773, 1000001789, 1000001791, 1000001801, 1000001803, 1000001819, 1000001857, 1000001887, 1000001917, 1000001927, 1000001957, 1000001963, 1000001969, 1000002043, 1000002089, 1000002103, 1000002139, 1000002149, 1000002161, 1000002173, 1000002187, 1000002193, 1000002233, 1000002239, 1000002277, 1000002307, 1000002359, 1000002361, 1000002431, 1000002449, 1000002457, 1000002499, 1000002571, 1000002581, 1000002607, 1000002631, 1000002637, 1000002649, 1000002667, 1000002727, 1000002791, 1000002803, 1000002821, 1000002823, 1000002827, 1000002907, 1000002937, 1000002989, 1000003009, 1000003013, 1000003051, 1000003057, 1000003097, 1000003111, 1000003133, 1000003153, 1000003157, 1000003163, 1000003211, 1000003241, 1000003247, 1000003253, 1000003267, 1000003271, 1000003273, 1000003283, 1000003309, 1000003337, 1000003351, 1000003357, 1000003373, 1000003379, 1000003397, 1000003469, 1000003471, 1000003513, 1000003519, 1000003559, 1000003577, 1000003579, 1000003601, 1000003621, 1000003643, 1000003651, 1000003663, 1000003679, 1000003709, 1000003747, 1000003751, 1000003769]
def v2tuple(s):
    v = int(s)
    return tuple(v%mod for mod in MODS)
def mul(a, b):
    return tuple((x*y)%mod for x, y, mod in zip(a, b, MODS))
A = [v2tuple(input()) for _ in range(N)]
cnts = dict()
for a in A:
    cnts[a] = cnts.get(a, 0) + 1
ans = 0
for i, ai in enumerate(A):
    for j, aj in enumerate(A):
        v = mul(ai, aj)
        ans += cnts.get(v, 0)
print(ans)