import sys
import numpy as np

n = int(sys.argv[1])
X = 2**np.random.uniform(-996.5784284662087, +996.5784284662087, n)
print(n, end=' ')
print(*X, sep=' ')
