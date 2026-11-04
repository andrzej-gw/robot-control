import random
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def value(alpha = 0.95):
    x = 1000 * random.random()
    y = 0.5 * x + random.gauss(0, 3) + 150
    if random.random() < alpha:
        y = random.uniform(150, 650)
    return x, y

values = [value() for _ in range(1000)]
values.sort()

t = 10

bestFit = None
bestPointCount = 0
for k in range(1000):
    sample = random.sample(values, 2)
    #  print(sample)
    
    x = np.array([v[0] for v in sample])
    y = np.array([v[1] for v in sample])
    maybeModel = stats.linregress(x, y)
    alsoInliers = []
    for x, y in values:
        y_pred = maybeModel.intercept + maybeModel.slope * x
        distance_squared = (y - y_pred)**2 / (1 + maybeModel.slope**2)
        if (x, y) not in sample and distance_squared < t:        
            alsoInliers.append([x, y])
    if bestPointCount < len(sample) + len(alsoInliers):
        new_sample = sample + alsoInliers
        x = np.array([v[0] for v in new_sample])
        y = np.array([v[1] for v in new_sample])
        bestFit = stats.linregress(x, y)
        bestPointCount = len(new_sample)
        
print("bestPointCount:", bestPointCount)
print(bestFit)
x = np.array([v[0] for v in values])
y = np.array([v[1] for v in values])
res = bestFit
plt.plot(x, y, 'o', label='original data')
plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
plt.legend()
plt.show()
