import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import HTML
from matplotlib.patches import Arc
from experiment.matplotlib_tools import circarrow, update_circarrow
from scipy.stats import wasserstein_distance
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
    import pathlib


def get_index(div_ranges, obs):
    result = []
    for i, _range in enumerate(div_ranges):
        idx = -1
        for j, x in enumerate(_range):
            if obs[i] <= x:
                idx = j
                break
        if idx == -1:
            result.append(len(_range) - 1)
        else:
            result.append(idx)
    return result


def load_data(data_dirs):
    #data_dirs = ['/home/joon/xai/causal/stable-baselines3/outputs/collected_21_inferior/data.pickle',
    #             '/home/joon/xai/causal/stable-baselines3/outputs/collected_21_superior/data.pickle']

    data = list()
    ranges = list()
    counts = list()

    idx = 0
    for data_dir in data_dirs:    
        with open(data_dir, 'rb') as handle:
            data.append(pickle.load(handle))
        ranges.append(data[idx]["ranges"])
        counts.append(data[idx]["counts"])
        idx += 1
    return ranges, counts

def sample_state(ranges, counts):
    pd = np.sum(np.sum(counts, axis=0), axis=0).flatten()
    pd = pd / np.sum(pd)
    pick = int(np.nonzero(np.random.multinomial(1, pd))[0])

    dp = []
    for i in range(4):
        dp = [pick % len(ranges[i])] + dp
        pick = pick // len(ranges[i])
    #dp = [10, 9, 10, 11]

    real_dp = []
    for i in range(4):
        real_dp += [(ranges[i][dp[i]-1] + ranges[i][dp[i]])/2]

    return real_dp, dp

def exp_gen(ranges, counts, state=None):
    avg = []
    stderr = []

    dp = []
    real_dp = []
    if state is None:
        real_dp, dp = sample_state(ranges, counts)
    else:
        dp = get_index(ranges, state)       
        for i in range(4):
            real_dp += [(ranges[i][dp[i]-1] + ranges[i][dp[i]])/2]
    
    probs = []
    errs = []
    to_probs = []
    to_errs = []

    print("State and binned state value:")
    print(state)
    print(real_dp)
    print(dp)

    for action in range(2):
        probs += [np.average(np.sum(counts[:, action, dp[0], :, :, :], axis=(1,2,3)) / np.sum(counts[:, action, :, :, :, :], axis=(1,2,3,4)))]
        errs += [np.std(np.sum(counts[:, action, dp[0], :, :, :], axis=(1,2,3)) / np.sum(counts[:, action, :, :, :, :], axis=(1,2,3,4)))]
        probs += [np.average(np.sum(counts[:, action, :, dp[1], :, :], axis=(1,2,3)) / np.sum(counts[:, action, :, :, :, :], axis=(1,2,3,4)))]
        errs += [np.std(np.sum(counts[:, action, :, dp[1], :, :], axis=(1,2,3)) / np.sum(counts[:, action, :, :, :, :], axis=(1,2,3,4)))]
        probs += [np.average(np.sum(counts[:, action, :, :, dp[2], :], axis=(1,2,3)) / np.sum(counts[:, action, :, :, :, :], axis=(1,2,3,4)))]
        errs += [np.std(np.sum(counts[:, action, :, :, dp[2], :], axis=(1,2,3)) / np.sum(counts[:, action, :, :, :, :], axis=(1,2,3,4)))]
        probs += [np.average(np.sum(counts[:, action, :, :, :, dp[3]], axis=(1,2,3)) / np.sum(counts[:, action, :, :, :, :], axis=(1,2,3,4)))]
        errs += [np.std(np.sum(counts[:, action, :, :, :, dp[3]], axis=(1,2,3)) / np.sum(counts[:, action, :, :, :, :], axis=(1,2,3,4)))]

        avg += [np.average(counts[:, action, :, dp[1], dp[2], dp[3]], axis=0)]
        stderr += [np.std(counts[:, 0, :, dp[1], dp[2], dp[3]], axis=0)]
        avg += [np.average(counts[:, action, dp[0], :, dp[2], dp[3]], axis=0)]
        stderr += [np.std(counts[:, 0, dp[0], :, dp[2], dp[3]], axis=0)]
        avg += [np.average(counts[:, action, dp[0], dp[1], :, dp[3]], axis=0)]
        stderr += [np.std(counts[:, 0, dp[0], dp[1], :, dp[3]], axis=0)]
        avg += [np.average(counts[:, action, dp[0], dp[1], dp[2], :], axis=0)]
        stderr += [np.std(counts[:, 0, dp[0], dp[1], dp[2], :], axis=0)]
    to_probs += [np.average(np.sum(counts[:, :, dp[0], :, :, :], axis=(1,2,3,4)) / np.sum(counts[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
    to_errs += [np.std(np.sum(counts[:, :, dp[0], :, :, :], axis=(1,2,3,4)) / np.sum(counts[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
    to_probs += [np.average(np.sum(counts[:, :, :, dp[1], :, :], axis=(1,2,3,4)) / np.sum(counts[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
    to_errs += [np.std(np.sum(counts[:, :, :, dp[1], :, :], axis=(1,2,3,4)) / np.sum(counts[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
    to_probs += [np.average(np.sum(counts[:, :, :, :, dp[2], :], axis=(1,2,3,4)) / np.sum(counts[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
    to_errs += [np.std(np.sum(counts[:, :, :, :, dp[2], :], axis=(1,2,3,4)) / np.sum(counts[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
    to_probs += [np.average(np.sum(counts[:, :, :, :, :, dp[3]], axis=(1,2,3,4)) / np.sum(counts[:, :, :, :, :, :], axis=(1,2,3,4,5)))]
    to_errs += [np.std(np.sum(counts[:, :, :, :, :, dp[3]], axis=(1,2,3,4)) / np.sum(counts[:, :, :, :, :, :], axis=(1,2,3,4,5)))]

    probs = np.array(probs)
    errs = np.array(errs)

    to_probs = np.array(to_probs)
    to_errs = np.array(to_errs)

    fig,ax = plt.subplots(1)
    rad_90 = 3.14159265/2
    rect = patches.Rectangle((real_dp[0]-0.15,-0.12),0.3,0.24,linewidth=1.5,edgecolor='black',facecolor='white')
    ax.add_patch(rect)
    ax.plot([real_dp[0], real_dp[0] + np.cos(real_dp[2] + rad_90)], [0, np.sin(real_dp[2] + rad_90)], color='black', linewidth=2.5)
    ax.arrow(real_dp[0], -0.25, real_dp[1], 0, color='tab:red', head_width=0.08)
    if real_dp[3] < 0.0:
        circarrow(ax, 0.3, real_dp[0] + np.cos(real_dp[2] + rad_90), np.sin(real_dp[2] + rad_90), 
                 np.rad2deg(real_dp[2] + rad_90 + real_dp[3] * 6), 
                 np.rad2deg(-real_dp[3] * 6), 
                 startarrow=True, head_width=0.05, color='tab:red')
    else:
        circarrow(ax, 0.3, real_dp[0] + np.cos(real_dp[2] + rad_90), np.sin(real_dp[2] + rad_90), 
                 np.rad2deg(real_dp[2] + rad_90), 
                 np.rad2deg(real_dp[3] * 6), 
                 endarrow=True, head_width=0.05, color='tab:red')
    ax.axhline(y=0)
    ax.set_ylim([-0.8, 2.4])
    ax.set_xlim([-2.4, 2.4])
    plt.savefig("cart.png")
    plt.show()

    names = ['cart lateral pos', 'cart lateral vel', 'pole angle', 'pole angular vel']

    print(probs[0:4])
    plt.bar(names, probs[0:4], yerr=errs[0:4])
    plt.savefig("prob_a0_res.png")
    plt.show()

    print(probs[4:8])
    plt.bar(names, probs[4:8], yerr=errs[4:8])
    plt.savefig("prob_a1_res.png")
    plt.show()

    print(to_probs)
    plt.bar(names, to_probs, yerr=to_errs)
    plt.savefig("prob_to_res.png")
    plt.show()

    saves = []
    scale = [2.4, 4.8, 0.209, 0.418]
    for i in range(4):
        output_str = ""
        for j in range(4):
            if i != j:
                output_str += " x[{}]={:.4f},".format(j, real_dp[j])            
        output_str = output_str[:-1]
        plt.title(names[i] + "\n" + "p(x[{}]| action = right(red) or left(blue),{})".format(i, output_str))
        diff = (ranges[i][1] - ranges[i][0]) / 2

        yesyes = False
        print("Yes yes = {}, {}".format(avg[i], avg[i+4]))
        if np.sum(avg[i]) < 1e-8:
            plt.plot(ranges[i] - diff, avg[i])
        else:
            yesyes = True
            plt.plot(ranges[i] - diff, avg[i] / np.sum(avg[i]))
            #plt.fill_between(ranges[i] - diff, avg[i]/ np.sum(avg[i])-stderr[i]/ np.sum(avg[i]), avg[i]/ np.sum(avg[i])+stderr[i]/ np.sum(avg[i]), fc='b')
        if np.sum(avg[i+4]) <= 1e-8:
            plt.plot(ranges[i] - diff, avg[i+4], c='r')
            yesyes = False
        else:
            plt.plot(ranges[i] - diff, avg[i+4]/np.sum(avg[i+4]), c='r')
            yesyes = True
            #plt.fill_between(ranges[i] - diff, avg[i+4]/np.sum(avg[i+4])-stderr[i+4]/np.sum(avg[i+4]), avg[i+4]/np.sum(avg[i+4])+stderr[i+4]/np.sum(avg[i+4]), fc='r')

        #Testing with Point Biserial Correlation
        r_pb = None
        ws_1 = None
        ws_2 = None
        if yesyes:
            ss0 = []
            ss1 = []
            trange0 = []
            freq0 = []
            trange1 = []
            freq1 = []
            #print(avg[i])
            #print(avg[i+4])
            for idx, val in enumerate(avg[i]):
                if idx is not 0:
                    ss0 += [(ranges[i][idx-1] + ranges[i][idx])/2] * int(val)
                    if avg[i][idx] > 0.0:
                        trange0 += [(ranges[i][idx-1] + ranges[i][idx])/2]
                        freq0 += [val]
                    ss1 += [(ranges[i][idx-1] + ranges[i][idx])/2] * int(avg[i+4][idx])
                    if avg[i+4][idx] > 0.0:
                        trange1 += [(ranges[i][idx-1] + ranges[i][idx])/2]
                        freq1 += [avg[i+4][idx]]
            m0 = np.average(ss0)
            m1 = np.average(ss1)
            sy = np.std(ss0+ss1)
            n0 = np.sum(avg[i])
            n1 = np.sum(avg[i+4])            
            nn = n0 + n1
            r_pb = (m0-m1)/sy * np.sqrt(n0/nn * n1/nn)
            #print("pbc -- ")
            #print(n0, n1, nn)
            ws_1 = wasserstein_distance(ss0, ss1) / scale[i]
            #print(trange0)
            #print(freq0)
            #print(trange1)
            #print(freq1)
            ws_2 = wasserstein_distance(trange0, trange1, freq0, freq1)


        print(dp[i])
        #print("difference", abs(avg[i][dp[i]]/np.sum(avg[i]) - avg[i+4][dp[i]]/np.sum(avg[i+4])))
        plt.axvline(x=[ranges[i][dp[i]]-diff], c='g')
        plt.ylim([-0.05, 1.1])
        plt.savefig("result_{}.png".format(i))
        plt.show()
        if yesyes:        
            print("point biserial coefficient = {}".format(r_pb))
            print("wasserstein distance = {}".format(ws_1))
            if r_pb > 0.5:
                saves += [(i, r_pb)]        
        
    print("")
    print("According to the causal analysis between state variables and the agent's action, ")
    for t in saves:
        i, rp = t[0], t[1]
        print("{} (coeff={})".format(names[t[0]], rp))
    print("are the main causes of the current chosen action to take place.")
    print("")

    actions = ["left" ,"right"]
    for t in saves:
        i, rp = t[0], t[1]
        left = avg[i][dp[i]]/np.sum(avg[i]) # 0 
        right = avg[i+4][dp[i]]/np.sum(avg[i+4]) # 1
        choice = 0 if left >= right else 1
            
        #print("left = {}, right = {}".format(left, right))
        print("The agent has taken the {} action".format(actions[choice]))
        #print("difference", abs(left-right))
        
        # left search
        j = dp[i] - 1
        while j > 0:
            _left = avg[i][j]/np.sum(avg[i]) # 0 
            _right = avg[i+4][j]/np.sum(avg[i+4]) # 1
            _choice = 0 if _left >= _right else 1
            if (choice is not _choice) and (abs(_left-_right)>0.00001):
                print("If {} were less than {:.5f} then the agent would have taken the {} action".format(names[i], (ranges[i][j-1] + ranges[i][j])/2, actions[_choice]))
                break
            j -= 1
        # right search
        j = dp[i] + 1
        print("The current {} is at {:.5f}".format(names[i], (ranges[i][dp[i]-1] + ranges[i][dp[i]])/2))
        while j <= 19:
            _left = avg[i][j]/np.sum(avg[i]) # 0 
            _right = avg[i+4][j]/np.sum(avg[i+4]) # 1
            _choice = 1 if _left >= _right else 0
            if (choice is not _choice) and (abs(_left-_right)>0.00001):            
                print("If {} were greater than {:.5f} then the agent would have taken the {} action".format(names[i], (ranges[i][j-1] + ranges[i][j])/2, actions[_choice]))
                break
            j += 1
