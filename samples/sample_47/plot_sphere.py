import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

xs = []
ys = []
zs = []

with open("./build/sample_47/sphere.txt") as f:
    for line in f:
        x, y, z = map(float, line.split())
        xs.append(x)
        ys.append(y)
        zs.append(z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs, s=10)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
    
ax.set_box_aspect([1,1,1]) # equal scaling

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()