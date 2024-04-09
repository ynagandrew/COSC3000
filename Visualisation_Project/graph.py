import numpy as np
import glob
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from scipy.stats import sem
from scipy import stats

data = pd.read_csv('out.csv')
data_directory = "./vis_project_data/"
soc_weights = np.array([-1, 0, 1, 2, 2, 3, 5])
soc_names = np.array(['Negative', 'No Interaction', 'Passive-low', 'Passive-high', 'Unilateral', 'Active-low', 'Active-high'])

cog_weights = np.array([0, -1, 1, 2, 2, 3, 5])
cog_names = np.array(['Non-play', 'Stereotype', 'Exploratory', 'Functional', 'Constructive', 'Symbolic', 'Rule-governed'])

colors = ['red', '0.4', (0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.466, 0.674, 0.188), (0.494, 0.184, 0.556), (0.856, 0.184, 0.408)]

# Types of sessions/files:
session_types = ["cog", "so"]
whatdoors = ["indoor", "outdoor"]
whichs = ["base", "inter"]

# Combine to single iterable list
combined_scenarios = [
    (ses_type, whatdoor, which)
    for ses_type in session_types
    for whatdoor in whatdoors
    for which in whichs
]

cas = ["albert", "barry", "chris", "ellie"]
ca_num = ['CA1', 'CA2', 'CA3', 'CA5']

untrained_peers = ['lydia', 'mario', 'nellie', 'oscar', 'peter']
trained_peers = ['ulrich', 'viola', 'wendy', 'xavier', 'yoshi', 'zara']


def scalar_to_rgb(value, cmap_name='bwr'):
    # Define the boundaries for clamping the value
    min_value = -1
    max_value = 1

    # Clamp the value within the range [-1, 1]
    value = np.clip(value, min_value, max_value)

    # Reverse the colormap to swap blue and red
    cmap = plt.get_cmap(cmap_name)
    reversed_cmap = cmap.reversed()

    # Sample colors from the reversed colormap
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)
    rgba_color = reversed_cmap(norm(value))
    return rgba_color[:3]  # Keep only RGB values, excluding alpha



def draw_a_box(base, ax, mean, error):
    if base:
        left = 0
        right = 0.5
    else:
        left = 0.5
        right = 1

    top_error = ax.transData.transform((0, mean+error))[1]
    bot_error = ax.transData.transform((0, mean-error))[1]

    ax.axhline(y=mean, xmin=left, xmax=right, color='black', linewidth=1)
    ax.axhline(y=mean - error, xmin=left, xmax=right, color='black', linewidth=0.5)
    ax.axhline(y=mean + error, xmin=left, xmax=right, color='black', linewidth=0.5)
    ax.axvline(x=left, ymin=bot_error, ymax=top_error, color='black', linewidth=0.5)
    ax.axvline(x=right, ymin=bot_error, ymax=top_error, color='black', linewidth=0.5)
    #ax.fill_between([left, right], mean-error, mean+error, color='pink')

def default_graph(trained, whatdoor, session_type):
    if trained:
        peers = trained_peers
    else:
        peers = untrained_peers

    if session_type == 'cog':
        weights = np.array([0, -1, 1, 2, 2, 3, 5])
    else:
        weights = np.array([-1, 0, 1, 2, 2, 3, 5])

    base_averages = []
    base_sem = []
    base_lens = []

    inter_averages = []
    inter_sem = []
    inter_lens = []

    fig, axs = plt.subplots(1, 5, sharey=True, gridspec_kw={'width_ratios': [1, 1, 1, 1, 2]})
    fig.subplots_adjust(wspace=0)

    i = 0
    for x in cas:
        print(x)
        base = np.divide(((data.loc[(data['ca'] == x) &
                        (data['peer'].isin(peers)) &
                        (data['whatdoor'] == whatdoor) &
                        (data['which'] == 'base') &
                        (data['cog'] == session_type)].iloc[:, 2:9].values) * weights).sum(axis=1), 360)

        inter = np.divide(((data.loc[(data['ca'] == x) &
                        (data['peer'].isin(peers)) &
                        (data['whatdoor'] == whatdoor) &
                        (data['which'] == 'inter') &
                        (data['cog'] == session_type)].iloc[:, 2:9].values) * weights).sum(axis=1), 360)

        print(np.average(base), sem(base))
        base_averages.append(np.average(base))
        base_sem.append(sem(base))
        base_lens.append(len(base))

        print(np.average(inter), sem(inter))
        inter_averages.append(np.average(inter))
        inter_sem.append(sem(inter))
        inter_lens.append(len(inter))

        axs[i].axvline(x=1, color='b', linestyle='dotted', linewidth=1)
        #axs[i].errorbar(0.9, np.average(base), yerr=sem(base), fmt='_')
        draw_a_box(True, axs[i], np.average(base), sem(base))
        draw_a_box(False, axs[i], np.average(inter), sem(inter))
        #axs[i].errorbar(1.1, np.average(inter), yerr=sem(inter), fmt='_')
        axs[i].set_xlim(0.8, 1.2)

        axs[i].set_xticks([1])
        t_stat, p_value = stats.ttest_ind(base, inter)

        string = ca_num[i]
        if p_value < 0.05:
            string = string + "**"
        axs[i].set_xticklabels([string])

        axs[i].tick_params(axis='x', direction='in', top=True, bottom=True)
        axs[i].tick_params(axis='y', top=False, bottom=False, left=False, right=False)

        i += 1

    # For Average plot:
    draw_a_box(True, axs[i], np.average(base_averages), np.average(base_sem))
    draw_a_box(False, axs[i], np.average(inter_averages), np.average(inter_sem))

    axs[i].set_xlim(0.8, 1.2)
    axs[i].axvline(x=1, color='b', linestyle='dotted', linewidth=1)
    axs[i].set_xticks([1])

    b1 = np.average(base_averages) + np.average(base_sem)
    b2 = np.average(base_averages) - np.average(base_sem)

    i1 = np.average(inter_averages) + np.average(inter_sem)
    i2 = np.average(inter_averages) - np.average(inter_sem)

    string = "All CAs"
    if not (b1 >= i2 and b2 <= i1) or (i1 >= b2 and i2 <= b1):
        string = string

    axs[i].set_xticklabels([string])
    axs[i].tick_params(axis='x', direction='in', top=True, bottom=True)
    axs[i].tick_params(axis='y', top=False, bottom=False, left=False, right=False)

    # Final settings:
    if session_type == 'cog':
        axs[0].set_yticks(np.arange(-1.0, 3.1, 1))
    else:
        axs[0].set_yticks(np.arange(0.0, 2.1, 0.5))
    axs[0].tick_params(axis='y', direction='in', left=True, right=False)
    axs[4].tick_params(axis='y', direction='in', left=False, right=True)

    if trained:
        fig.suptitle("Trained Dyads - "+ whatdoor)
    else:
        fig.suptitle("Untrained Dyads - "+ whatdoor)

    if session_type == 'cog':
        axs[0].set_ylabel('Cognitive play')
    else:
        axs[0].set_ylabel('Social interaction')

    fig.text(0.68, 0.04, "Asterisks represent statistical significance", fontsize=6)
    plt.savefig("1.png", bbox_inches="tight", dpi=300)
    plt.show()

def individual_sessions(ca, trained):

    if trained == 'trained':
        peers = trained_peers
    else:
        peers = untrained_peers

    fig = plt.figure()
    subfigs = fig.subfigures(2, 2)

    for i in range(2):
        for j in range(2):
            plot = data.loc[(data['ca'] == ca) &
                               (data['peer'].isin(peers)) &
                               (data['whatdoor'] == whatdoors[j]) &
                               (data['cog'] == session_types[i])]
            plot.sort_values(by=['8'])

            plotted = plot.iloc[:, [2, 3, 4, 5, 6, 7, 8]].values
            plotted = np.transpose(plotted)

            for x in plotted:
                np.append(x, x[-1])

            which = plot.iloc[:, [15]].values
            change_indices = [i for i in range(1, len(which)) if which[i] != which[i-1]]

            if session_types[i] == 'cog':
                names = cog_names
            else:
                names = soc_names

            neg_data = {name: plotted[0:2][i].tolist() for i, name in reversed(list(enumerate(names[0:2])))}
            pos_data = {name: plotted[2:][i].tolist() for i, name in enumerate(names[2:])}

            neg_colours = ['0.4', 'red']

            graphs = subfigs[i, j].subplots(2, 1, sharex=True)
            graphs[0].stackplot(np.arange(len(plotted[0])), pos_data.values(), step='post', labels=pos_data.keys(), colors=[(0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.466, 0.674, 0.188), (0.494, 0.184, 0.556), (0.856, 0.184, 0.408)])
            graphs[1].stackplot(np.arange(len(plotted[0])), neg_data.values(), step='post', labels=neg_data.keys(), colors=neg_colours)

            graphs[0].set_ylim(0, 360)
            graphs[1].set_ylim(0, 360)

            graphs[1].invert_yaxis()
            print(which)
            print(change_indices[0])
            graphs[0].axvline(x=change_indices[0], color='black', linestyle='dotted', linewidth=1)
            graphs[1].axvline(x=change_indices[0], color='black', linestyle='dotted', linewidth=1)

            graphs[0].set_yticks([])
            graphs[1].set_yticks([])

            if j == 0:
                graphs[0].set_ylabel(session_types[i], fontsize=15, y=0)

            if j == 1:
                graphs[0].legend(bbox_to_anchor=(1.04, 0), loc="lower left", reverse=True)
                graphs[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            if i == 0:
                graphs[0].set_title(f'{whatdoors[j]}', fontsize=15)

    plt.subplots_adjust(hspace=0.0)

    fig.suptitle(ca + ', ' + trained, y=0)
    plt.savefig(ca + trained + ".png", bbox_inches = "tight", dpi=300)
    plt.show()
def heatmap(cog_new_weights,soc_new_weights, comparison):

    fig, ax = plt.subplots()
    heat_array = []
    for i in range(2):
        session_type = session_types[i]
        if i == 0:
            default_weights = cog_weights
            new_weights = cog_new_weights
        else:
            default_weights = soc_weights
            new_weights = soc_new_weights
        for j in range(4):
            if j < 1.5:
                peers = untrained_peers
            else:
                peers = trained_peers
            if (j == 0 or j == 2):
                whatdoor = whatdoors[0]
            else:
                whatdoor = whatdoors[1]
            for x in cas:
                base = np.divide(((data.loc[(data['ca'] == x) &
                                            (data['peer'].isin(peers)) &
                                            (data['whatdoor'] == whatdoor) &
                                            (data['which'] == 'base') &
                                            (data['cog'] == session_type)].iloc[:, 2:9].values) * default_weights).sum(axis=1), 360)

                inter = np.divide(((data.loc[(data['ca'] == x) &
                                             (data['peer'].isin(peers)) &
                                             (data['whatdoor'] == whatdoor) &
                                             (data['which'] == 'inter') &
                                             (data['cog'] == session_type)].iloc[:, 2:9].values) * default_weights).sum(axis=1), 360)

                default_heat = np.average(inter) - np.average(base)

                if not comparison:
                    heat_array.append(default_heat)
                else:

                    base = np.divide(((data.loc[(data['ca'] == x) &
                                                (data['peer'].isin(peers)) &
                                                (data['whatdoor'] == whatdoor) &
                                                (data['which'] == 'base') &
                                                (data['cog'] == session_type)].iloc[:,2:9].values) * new_weights).sum(axis=1), 360)

                    inter = np.divide(((data.loc[(data['ca'] == x) &
                                                 (data['peer'].isin(peers)) &
                                                 (data['whatdoor'] == whatdoor) &
                                                 (data['which'] == 'inter') &
                                                 (data['cog'] == session_type)].iloc[:,2:9].values) * new_weights).sum(axis=1), 360)

                    new_heat = np.average(inter) - np.average(base)

                    heat = new_heat - default_heat

                    heat_array.append(heat)

    new_order = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 16, 17, 20, 21, 24, 25, 28, 29, 18, 19, 22, 23, 26, 27, 30, 31]
    final_array = []

    for x in new_order:
        final_array.append(heat_array[x])

    print(max(final_array))
    print(min(final_array))

    final_array = np.reshape(final_array, (4, 8))
    final_array = np.array([[scalar_to_rgb(scalar) for scalar in row] for row in final_array])

    im = ax.imshow(final_array)
    plt.suptitle("Default Weights vs Shifted Weights [-2, -1, 0, 1, 2, 3, 5]", fontsize=15)
    plt.title("Untrained                        Trained", fontsize=15)
    ax.set_xticks([0.5, 2.5, 4.5, 6.5])
    ax.set_yticks([])
    ax.axhline(y=1.5, color='black', linewidth=1)
    ax.axvline(x=1.5, color='black', linewidth=1)
    ax.axvline(x=3.5, color='black', linewidth=1)
    ax.axvline(x=5.5, color='black', linewidth=1)
    ax.set_xticklabels(["indoor", "outdoor", "indoor", "outdoor"])
    ax.set_ylabel("soc              cog", y=0.5, fontsize=15)
    plt.savefig("hello.png", bbox_inches="tight", dpi=300)
    plt.show()

def heat_legend():
    fig, ax = plt.subplots()
    arr = [['CA1', 'CA2'], ['CA3', 'CA5']]
    ax.imshow()


cog_weights = np.array([0, -1, 1, 2, 2, 3, 5])
soc_weights = np.array([-1, 0, 1, 2, 2, 3, 5])

no_high_weights_cog = np.array([0, -1, 1, 1, 2, 2, 2])
no_high_weights_soc = np.array([-1, 0, 1, 1, 2, 2, 2])

high_weighted_cog = np.array([0, -1, 1, 2, 4, 6, 8])
high_weighted_soc = np.array([-1, 0, 1, 2, 4, 6, 8])

early_zero_cog = np.array([-1, -2, 0, 1, 2, 3, 5])
early_zero_soc = np.array([-2, -1, 0, 1, 2, 3, 5])

heatmap(early_zero_cog, early_zero_soc, True)
#individual_sessions('chris', "untrained")
#individual_sessions('chris', 'trained')
#default_graph(False, 'indoor', 'cog')