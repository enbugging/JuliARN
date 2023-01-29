import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

res = np.array([])
res = np.append(res, 2*np.ones(12))
res = np.append(res, 3*np.ones(41))
res = np.append(res, 4*np.ones(22))
res = np.append(res, 5*np.ones(25))
res = np.append(res, 6*np.ones(11))
res = np.append(res, 7*np.ones(5))
res = np.append(res, 8*np.ones(1))
res = np.append(res, 10*np.ones(1056))
res = np.append(res, 14*np.ones(3))
res = np.append(res, 20*np.ones(3))
res = np.append(res, 22*np.ones(2))
res = np.append(res, 24*np.ones(5))
res = np.append(res, 28*np.ones(2))
sns.displot(res, kind="ecdf")
plt.show()
