import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots()
arc = patches.Arc(
    [0,0], 5, 5, angle = 0.0, theta1 = 0.0, theta2 = 50.0,
    linewidth = 1, color='tab:red'
)
arc.center = [1,1]
arc.angle = 50
arc.theta1 = 100
arc.theta2 = 150
print(arc.__dict__)
ax.add_patch(arc)
ax.plot([0,10], [0,10])
plt.show()


