import numpy as np
import boto3
from io import BytesIO
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


results = []

results.append((3, 97.38))
results.append((5, 97.88))
results.append((7, 98.13))
results.append((10, 98.44))
results.append((13, 98.95))
results.append((16, 98.98))
results.append((20, 98.96))
results.append((25, 98.93))
results.append((30, 98.87))
results.append((40, 98.98))
results.append((50, 99.03))
results.append((60, 98.98))
results.append((70, 98.91))
results.append((80, 98.95))
results.append((90, 98.96))
results.append((100, 99.04))

xSize = []
yPercent = []
for values in results:
    xSize.append(values[0])
    yPercent.append(values[1])

plt.figure()
plt.suptitle('Pourcentage de classifications réussies selon la taille du train set en 1 vs all')
plt.xlabel('Taille du train set')
plt.ylabel('Pourcentage de réussite')
plt.plot(xSize, yPercent, 'r')
plt.xticks(xSize)
graph = BytesIO()
plt.savefig(graph, format='jpg')
graph.seek(0)
s3 = boto3.resource('s3')
s3.Object('oc-plawson', 'toto.jpg').put(Body=graph)
