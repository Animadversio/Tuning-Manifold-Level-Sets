"""
Script for making schematics of the landscape in figure 1
"""
import matplotlib.pyplot as plt
# 3d plot of surface
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from skimage.measure import find_contours
from core.utils.plot_utils import saveallforms

#%%
figdir = r"E:\OneDrive - Harvard University\NeurRep2022_NeurIPS\Figures\Schematics\src"
XX, YY = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
ZZ = np.exp(5*(- XX**2 - YY**2))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(XX, YY, ZZ, cmap='viridis')
ax.contour3D(XX, YY, ZZ, 25, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
saveallforms(figdir, "test_contour3D", fig)
plt.show()
#%% 3d plot of surface
def gauss_function(XX, YY, center, precision):
    XXc = XX - center[0]
    YYc = YY - center[1]
    ZZ = np.exp(- 0.5*(XXc**2 * precision[0, 0] +
                       YYc**2 * precision[1, 1] +
                2 * XXc * YYc * precision[0, 1]))
    return ZZ

XX, YY = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
fun1 = gauss_function(XX, YY, center=[0.5, 0.8],
                  precision=np.array([[2.5, 1], [1, 6]]))
fun2 = gauss_function(XX, YY, center=[-0.5, -0.3],
                  precision=np.array([[3, 1.5], [1.5, 2]]))
fun3 = gauss_function(XX, YY, center=[0.5, -0.3],
                  precision=np.array([[0.2, 0], [0, 0.5]]))
ZZ = 0.6 * fun1 + fun2 + 0.04405 * fun3
#%%
fig = plt.figure()
plt.contour(XX, YY, ZZ, 17, cmap='viridis', alpha=0.7)
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('image')
saveallforms(figdir, "test_contour2D", fig)
plt.show()
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(XX, YY, ZZ, cmap='viridis')
ax.contour3D(XX, YY, ZZ, 18, cmap='viridis', alpha=0.7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# change view angle
ax.view_init(22, -45)
saveallforms(figdir, "test_contour3D", fig)
plt.show()
ax.view_init(90, 0)
saveallforms(figdir, "test_contour3D_topview", fig)
plt.show()
#%%