import numpy as np
import experiment.gym_wrappers as gym_wrappers

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib.patches import Arc
from experiment.matplotlib_tools import circarrow, update_circarrow


class Animator(object):
    RAD_90 = 3.14159265/2

    def __init__(self, start_state_mode=gym_wrappers.StartStateMode.DESIGNATED_POSITIONS,
                 start_states=[0.0],
                 add_noise=True,
                 seed=5334):
        self.env = gym_wrappers.make("cartpole", "zero", "0", 
                                    start_state_mode=start_state_mode,
                                    start_states=start_states,
                                    add_noise=add_noise,
                                    seed=seed)
        
    def init(self):
        self.rect.set_visible(False)
        self.pole.set_visible(False)
        self.arc.set_visible(False)
        return self.rect, self.pole, self.arc # , self.start_arrow, self.end_arrow

    def update(self, i):
        self.rect.set_visible(True)
        self.pole.set_visible(True)
        self.arc.set_visible(True)
        state = self.states[i][0]
        self.rect.set_xy([state[0]-0.15,-0.12])
        self.pole.set_xdata([state[0], state[0] + np.cos(state[2] + Animator.RAD_90)])
        self.pole.set_ydata([0, np.sin(state[2] + Animator.RAD_90)])
        self.arrow.remove()
        self.arrow = self.ax.arrow(state[0], -0.25, state[1], 0, color='tab:red', head_width=0.08)        
        if state[3] < 0.0:
            self.arc, self.start_arrow, self.end_arrow = \
                update_circarrow(self.ax, self.arc, self.start_arrow, self.end_arrow, 0.3, 
                          state[0] + np.cos(state[2] + Animator.RAD_90),
                          np.sin(state[2] + Animator.RAD_90), 
                          np.rad2deg(state[2] + Animator.RAD_90 + state[3] * 6), 
                          np.rad2deg(-state[3] * 6), 
                          startarrow=True, head_width=0.05, color='tab:red')
        else:
            self.arc, self.start_arrow, self.end_arrow = \
                update_circarrow(self.ax, self.arc, self.start_arrow, self.end_arrow, 0.3, 
                          state[0] + np.cos(state[2] + Animator.RAD_90), 
                          np.sin(state[2] + Animator.RAD_90), 
                          np.rad2deg(state[2] + Animator.RAD_90),
                          np.rad2deg(state[3] * 6), 
                          endarrow=True, head_width=0.05, color='tab:red')
        return self.rect, self.pole, self.arc

    def display_anim(self, agent, state=None, total_steps=50):            
            obs = self.env.reset()
            if state is not None:
                self.env.manual_set(state)

            self.states = [(obs,0,False)]
            for i in range(total_steps):
                action, _states = agent.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)                
                self.states += [(obs, reward, done)]
                if done:
                    break
            
            self.fig,self.ax = plt.subplots(1)
            state = self.states[0][0]
            self.rect = patches.Rectangle(
                [state[0]-0.15,-0.12],0.3,0.24,linewidth=1.5,edgecolor='black',facecolor='white')
            self.ax.add_patch(self.rect)
            self.pole, = self.ax.plot(
                [state[0], state[0] + np.cos(state[2] + Animator.RAD_90)],
                [0, np.sin(state[2] + Animator.RAD_90)],
                color='black', linewidth=2.5)
            self.ax.axhline(y=0)
            self.ax.set_ylim([-0.8, 2.4])
            self.ax.set_xlim([-2.4, 2.4])
            self.arrow = self.ax.arrow(state[0], -0.25, state[1], 0, color='tab:red', head_width=0.08)
            if state[3] < 0.0:
                self.arc, self.start_arrow, self.end_arrow = \
                    circarrow(self.ax, 0.3,
                            state[0] + np.cos(state[2] + Animator.RAD_90),
                            np.sin(state[2] + Animator.RAD_90),
                            np.rad2deg(state[2] + Animator.RAD_90 + state[3] * 6),
                            np.rad2deg(-state[3] * 6),
                            startarrow=True, head_width=0.05, color='tab:red')
            else:
                self.arc, self.start_arrow, self.end_arrow = \
                    circarrow(self.ax, 0.3,
                            state[0] + np.cos(state[2] + Animator.RAD_90),
                            np.sin(state[2] + Animator.RAD_90),
                            np.rad2deg(state[2] + Animator.RAD_90),
                            np.rad2deg(state[3] * 6),
                            endarrow=True, head_width=0.05, color='tab:red')
            
            self.anim = animation.FuncAnimation(self.fig, self.update, init_func=self.init,
                                frames=total_steps, interval=100, blit=True)
