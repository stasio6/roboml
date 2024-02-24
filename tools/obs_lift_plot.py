import matplotlib.pyplot as plt
import numpy as np
import os, csv
import warnings

X_NAME_TO_X_LABEL = {
    'global_step': 'Environment Steps',
    '_runtime': 'Wall-clock Time (hours)',
}

# EXTEND_CURVE = True
EXTEND_CURVE = False

def make_style(fig, ax):
    ax.xaxis.grid(True, which='major',linestyle = (0,(8,4))) 
    ax.yaxis.grid(True, which='major',linestyle = (0,(8,4))) 

def get_curves_from_csv_dir(dir_path, x_name='global_step', y_name='train/success', x_lim=None):
    curves = []
    for file_path in os.listdir(dir_path):  
        if not file_path.endswith(".csv"):
            continue
        print('Read from:', dir_path, file_path)
        with open(f"{dir_path}/{file_path}", newline='') as csvfile:
            firstline = csvfile.readline()
            while ',' not in firstline:
                firstline = csvfile.readline()
            col_names = firstline[:-1].split(",")
            if "DrQ" in dir_path:
                if x_name == 'global_step':
                    x_name = 'frame'
                elif x_name == '_runtime':
                    x_name = 'hour'
                y_name = 'episode_reward'
            x_col = col_names.index(x_name)
            y_col = col_names.index(y_name)
            reader = csv.reader(csvfile)
            sorted_rows = sorted(reader, key=lambda row: float(row[x_col]))
            x_list, y_list = [], []
            for row in sorted_rows:
                if row[x_col] != "" and row[y_col] != "":
                    _x = float(row[x_col])
                    _y = float(row[y_col])
                    if x_lim and _x > x_lim:
                        break
                    x_list.append(_x)
                    y_list.append(_y)
        if "DrQ" in dir_path:
            if x_name == 'hour':
                x_list = list(np.array(x_list) * 3600)
            else:
                x_list = list(np.array(x_list) * 1) # TODO: Transform frame to global_step accordingly
        curves.append({'x': x_list, 'y': y_list})

    return curves


def smooth(y, window=20):
    new_y = np.zeros_like(y)
    for i in range(len(y)):
        new_y[i] = np.mean(y[max(0, i-window):i+1])
    return new_y

def prepare_curves_for_subplot(data_for_subplot, x_name='global_step', y_name='train/success', num_bins=200, smooth_window=10):
    has_state_expert = False
    for data_for_method in data_for_subplot['methods']:
        if 'State RL' in data_for_method['label']:
            has_state_expert = True

    raw_curves_for_all_method, max_x_for_all_method = [], []
    x_lim = data_for_subplot.get('x_lim', None)
    for data_for_method in data_for_subplot['methods']:
        raw_curves = get_curves_from_csv_dir(data_for_method['dir'], x_name, y_name, x_lim=data_for_method.get(f'x_lim', x_lim))
        max_x_for_all_method.append(max([curve['x'][-1] for curve in raw_curves]))
        raw_curves_for_all_method.append(raw_curves)
    global_max_x = max(max_x_for_all_method)

    curves_for_all_method = []
    for i, raw_curves in enumerate(raw_curves_for_all_method):
        # Dirty hack
        for j in range(len(raw_curves)):
            raw_curves[j] = {'x': [0] + raw_curves[j]['x'], 'y': [0] + raw_curves[j]['y']}

        # Merge curves to a single curve
        if EXTEND_CURVE:
            linspace = np.linspace(0, global_max_x, num_bins+1)
        else:
            local_max_x = max_x_for_all_method[i]
            num_bins_for_this_curve = int(num_bins * (local_max_x / global_max_x))
            linspace = np.linspace(0, local_max_x, num_bins_for_this_curve+1)
        interp_curves = np.array([np.interp(linspace, curve['x'], smooth(curve['y'], window=smooth_window)) for curve in raw_curves])
        merged_curve = {'x': linspace, 'y': np.mean(interp_curves, axis=0), 'std': np.std(interp_curves, axis=0)}
        curve = merged_curve

        # post processing
        for key in ['y', 'std']:
            if 'success' in y_name:
                curve[key] = np.array(curve[key]) * 100
            curve[key] = smooth(curve[key], window=smooth_window)
        if x_name == '_runtime':
            curve['x'] = np.array(curve['x']) / 3600

        # specific to s2v project
        data_for_method = data_for_subplot['methods'][i]
        if 'State RL' in data_for_method['label']:
            offset = curve['x'][-1]
        if 'DAgger' in data_for_method['label'] and has_state_expert:
            curve['x'] = curve['x'] + offset

        curves_for_all_method.append(curve)
    return curves_for_all_method


def draw(
        data_for_subplots,
        save_name=None, 
        figsize=(6,4),
        x_name='global_step', 
        y_name='train/success', 
        smooth_window=10,
        outside_legend=False,
    ):
    # plt.rcParams["font.family"] = "Helvetica" # I don't have it installed locally, feel free to use it

    n_subplots = len(data_for_subplots)
    fig, axes = plt.subplots(nrows=1, ncols=n_subplots, figsize=figsize, dpi=300)
    if n_subplots == 1:
        axes = [axes]

    ##################################
    #  Make each subplot
    ##################################

    for i_plot, data_for_subplot in enumerate(data_for_subplots):
        ax = axes[i_plot]

        ##################################
        #  Prepare data
        ##################################
        curves = prepare_curves_for_subplot(data_for_subplot, x_name, y_name, smooth_window=smooth_window)
        n_method = len(curves)

        ##################################
        #  Plot
        ##################################
        for i, curve in enumerate(curves):
            x, y, std = curve['x'], curve['y'], curve['std']
            ax.plot(x, y, label=data_for_subplot['methods'][i]['label'], linewidth=3)
            ax.fill_between(x, y-std, y+std, alpha=0.2)

        ##################################
        #  Style
        ##################################
        ax.set_facecolor("#F2F2F2")
        ax.set_title(data_for_subplot['title'], fontweight='bold')
        x_lim = data_for_subplot.get('x_lim', None)
        if x_lim is None:
            ax.set_xlim(left=0)
        else:
            if x_name == '_runtime':
                x_lim = x_lim / 3600
            ax.set_xlim(0, x_lim)
        # ax.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=False) # labels along the bottom edge are off
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.set_xlabel(X_NAME_TO_X_LABEL[x_name], fontweight='bold')

        if 'success' in y_name:
            ax.set_ylim((0, 105))
            if i_plot == 0:
                ax.set_ylabel('Success Rate %', fontweight='bold')
        else:
            warnings.warn(f'Unknown Y-axis data {y_name}, so Y-axis is auto-configured.')
            if i_plot == 0:
                ax.set_ylabel(y_name, fontweight='bold')
        if not outside_legend:
            leg = ax.legend(loc='best', fontsize=10)
            # leg = ax.legend(loc='upper left', fontsize=10)
            # leg = ax.legend(loc='lower right', fontsize=10)
            # # set the linewidth of each legend object
            # for legobj in leg.legendHandles:
            #     legobj._sizes = [15]

    if outside_legend:
        x_offset = 0.5 if n_subplots % 2 == 1 else 0
        axes[n_subplots//2].legend(bbox_to_anchor=(x_offset,-0.45), loc='lower center', ncol=n_method) # 0.5 means center


    ##################################
    #  Overall Style
    ##################################
    for ax in axes:
        make_style(fig, ax)

    # if not outside_legend:
    #     fig.tight_layout()
    if save_name is None:
        save_name = data_for_subplots[0]['title']

    os.makedirs(f'{DATA_DIR}/figures', exist_ok=True)
    fig.savefig(f'{DATA_DIR}/figures/{save_name}.jpg', bbox_inches="tight")
    fig.savefig(f'{DATA_DIR}/figures/{save_name}.pdf', bbox_inches="tight")

if __name__ == "__main__":

    ##############################################
    # ManiSkill Global Parameters
    ##############################################
    DATA_DIR = 'data/obs_lift_plots'
    max_global_step = None
    max__runtime = None
    y_name = 'train/success'
    smooth_window = 20
    figsize = (4,3)

    ##############################################
    # ManiSkill Env Specific Parameters
    ##############################################

    env_name = 'PickCube'
    expert_max_global_step = 320_000
    expert_max__runtime = 0.75 * 3600
    max_global_step = 600_000
    smooth_window = 10
    max__runtime = 3 * 3600
    
    # env_name = 'StackCube'
    # expert_max_global_step = 4_000_000
    # expert_max__runtime = 10.22 * 3600
    # max_global_step = 6_000_000
    # smooth_window = 10
    # max__runtime = 15 * 3600

    # env_name = 'TurnFaucet'
    # expert_max_global_step = 5_000_000
    # expert_max__runtime = 15 * 3600
    # max_global_step = 7_000_000
    # max__runtime = 30 * 3600

    # env_name = 'PickSingleYCB'
    # expert_max_global_step = 5_000_000
    # expert_max__runtime = 13.4 * 3600
    # max_global_step = 7_000_000
    # max__runtime = 25 * 3600

    # env_name = 'OpenDrawer'
    # expert_max_global_step = 7_000_000
    # expert_max__runtime = 20 * 3600
    # max_global_step = 12_000_000
    # max__runtime = 35 * 3600
    # smooth_window = 5

    # env_name = 'MoveBucket'
    # expert_max_global_step = 25_000_000
    # expert_max__runtime = 85 * 3600
    # max_global_step = 40_000_000
    # smooth_window = 5
    # max__runtime = 130 * 3600

    # env_name = 'PegInsertion'
    # expert_max_global_step = 12_000_000
    # expert_max__runtime = 32 * 3600
    # max_global_step = 18_000_000
    # max__runtime = 55 * 3600
    # smooth_window = 10

    # env_name = 'PickClutterYCB'
    # expert_max_global_step = 8_000_000
    # expert_max__runtime = 19 * 3600
    # max_global_step = 15_000_000
    # max__runtime = 45 * 3600
    # smooth_window = 10

    # env_name = 'OpenDoor'
    # expert_max_global_step = 2_000_000
    # expert_max__runtime = 6.5 * 3600
    # max_global_step = 2_500_000
    # # max__runtime = 20 * 3600
    # smooth_window = 20

    ##############################################
    # DMControl Global Parameters
    ##############################################
    # y_name = 'train/return'
    # smooth_window = 5

    ##############################################
    # DMControl Env Specific Parameters
    ##############################################

    # env_name = 'Acrobot-Swingup'
    # expert_max_global_step = 1_500_000
    # expert_max__runtime = 153 * 60
    # max_global_step = 4_000_000
    # max__runtime = 5 * 3600

    # env_name = 'Reacher-Hard'
    # expert_max_global_step = 200_000
    # expert_max__runtime = 21 * 60
    # max_global_step = 3_000_000
    # max__runtime = 2.2 * 3600

    # env_name = 'Swimmer-6'
    # expert_max_global_step = 300_000
    # expert_max__runtime = 35 * 60
    # max_global_step = 3_000_000
    # max__runtime = 10 * 3600

    # env_name = 'Walker-Run'
    # expert_max_global_step = 1_000_000
    # expert_max__runtime = 95 * 60
    # max_global_step = 3_000_000
    # max__runtime = 3 * 3600

    # env_name = 'Hopper-Hop'
    # expert_max_global_step = 3_000_000
    # expert_max__runtime = 330 * 60
    # max_global_step = 6_000_000
    # max__runtime = 8 * 3600

    # env_name = 'Humanoid-Walk'
    # expert_max_global_step = 4_000_000
    # expert_max__runtime = 9.3 * 3600
    # max_global_step = 10_000_000
    # max__runtime = 15 * 3600

    ######################################################################
    # Plot
    ######################################################################

    for x_axis_name in ['env_steps', 'wall_time']:

        single_env_data = {
            'methods': [
                {
                    'label': 'Obs Lift Stage 1 (State RL)',
                    'dir': f'{DATA_DIR}/{env_name}_SAC-state',
                    'x_lim': expert_max_global_step if x_axis_name == 'env_steps' else expert_max__runtime,
                },
                {
                    'label': 'Obs Lift Stage 2 (State-to-Visual DAgger)',
                    'dir': f'{DATA_DIR}/{env_name}_s2v',
                },
                {
                    'label': 'Visual RL',
                    'dir': f'{DATA_DIR}/{env_name}_AAC-SAC',
                },
                # Uncomment for DrQ2 plots
                # {
                #     'label': 'DrQ-v2',
                #     'dir': f'{DATA_DIR}/{env_name}_DrQ2',
                # },
            ],
            'title': env_name,
            'x_lim': max_global_step if x_axis_name == 'env_steps' else max__runtime,
        }
        draw(
            data_for_subplots=[single_env_data], 
            save_name=f'{env_name}_{x_axis_name}',
            x_name='_runtime' if x_axis_name == 'wall_time' else 'global_step',
            y_name=y_name,
            figsize=figsize, 
            smooth_window=smooth_window,
        )
