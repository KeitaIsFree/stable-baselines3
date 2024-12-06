import numpy
import matplotlib.pyplot as plt

def rew(action):
    rew = 0
    for act_i in range(2):
        a = action[act_i] * 3.0
        if a < 0:
            rew += a + 1
        elif a < 1:
            rew += -a + 1
        elif a < 1.5:
            rew += 4 * a - 4
        else:
            rew += -4 * a + 8
    return rew

q_values = []
for y in numpy.arange(1.00,-1.00, -0.02):
    q_values.append([])
    for x in numpy.arange(-1.00,1.00, 0.02):
        vl = rew((x,y))
        q_values[-1].append(vl)

plt.clf()
plt.imshow(q_values, cmap='viridis', aspect='auto', extent=[-1, 1, -1, 1])
plt.colorbar()  # Adds a colorbar to the side
plt.title('Heatmap')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig(f'absexp_rew.pdf')