import matplotlib
from tqdm.auto import tqdm
import pickle
import time
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import mpl_toolkits
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterSciNotation
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns
import pandas as pd
import yaml
import types
import psutil
from src.algos.stochastic.oracle import Oracle
from src.algos.stochastic.random import StochasticRandomized
from src.algos.stochastic.sqrt_first import SqrtFirst
from src.algos.stochastic.proposed import StochasticSubmodularPolicy
from src.algos.stochastic.rl import StochasticReinforce
from src.algos.stochastic.mab import StochasticGreedyMAB
from src.models.bean import *
import numpy as np
np.random.seed(923520)

matplotlib.use('Agg')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times'] 
plt.rcParams['text.latex.preamble'] = r"""
\usepackage{amsmath}
\usepackage{amssymb}
"""

import scienceplots
plt.style.use(['science', 'ieee'])

from multiprocessing import Pool

def run_command(command):
    os.system(command)

def main():
    # command line arguments (dataset)
    parser = argparse.ArgumentParser(description='offline algorithm runner')
    parser.add_argument('-D', type=str, choices=['CD18', 'CD14', 'XA18', 'SYN', 'ALL'], required=True, help='dataset to use')
    parser.add_argument('-C', type=str, default='configs/online.yaml', help='path to YAML config for rerun settings')
    args = parser.parse_args()

    # run all experiments
    if args.D == 'ALL':
        commands = [
            'python online.py -D CD18',
            'python online.py -D CD14',
            'python online.py -D XA18',
            'python online.py -D SYN'
        ]
        with Pool(4) as p:
            p.map(run_command, commands)
        exit()

    # dataset configuration
    DATASET = {'CD18': 'chengdu2018', 'CD14': 'chengdu2014', 'XA18': 'xian2018', 'SYN': 'synthetic'}[args.D]
    TIMES = 3
    n_task = 2400
    n_task_range = range(100, 3700, 400)
    n_vehicles = 100
    n_vehicles_range = range(30, 110, 10)
    l = 2
    l_range = list(range(2, 10, 2))
    beta = 0.75
    beta_range = np.arange(0.1, 0.9, 0.2)
    if DATASET == 'chengdu2014':
        l = 2
        l_range = list(range(1, 5, 1))
        beta = 0.7
        beta_range = np.linspace(0.3, 0.7, 5)
        n_vehicles = 300
        n_vehicles_range = list([100, 500, 1000, 5000])
    if DATASET == 'synthetic':
        n_vehicles = 300

    # experiment logs
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # load YAML configuration (optional)
    cfg = {}
    if os.path.exists(args.C):
        try:
            with open(args.C, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load YAML config '{args.C}': {e}. Proceeding with defaults.")
    online_cfg = cfg.get('online') or {}
    rerun_cfg = online_cfg.get('rerun') or {}
    rerun_algorithms = set(rerun_cfg.get('algorithms') or [])
    rerun_variables = set(rerun_cfg.get('variables') or ['n_task', 'n_vehicles', 'beta', 'l'])
    force_rerun = bool(rerun_cfg.get('force', False))
    algorithm_params_cfg = online_cfg.get('algorithm_params') or {}

    # Enforce logs=False for specific algorithms to avoid memory overhead
    for algo in ['Exploration-Then-Commit', 'StochasticSubmodularPolicy', 'StochasticGreedyMAB']:
        if algo not in algorithm_params_cfg:
            algorithm_params_cfg[algo] = {}
        algorithm_params_cfg[algo]['logs'] = False

    # generate input problem
    if DATASET == 'synthetic':
        rho=0
        tasks_universal = [Task(idx=j) for j in range(3700)]
        num_tasks_for_attrs = 200
        num_normal = 5100
        num_malicious = 5100
        p_normal = np.random.random((num_normal, num_tasks_for_attrs))
        r_normal = np.random.random(num_normal)
        c_normal = np.random.uniform(0, 100, (num_normal, num_tasks_for_attrs))
        p_malicious = np.random.random((num_malicious, num_tasks_for_attrs))
        r_malicious = np.random.random(num_malicious)
        c_malicious_shared = np.ones(num_tasks_for_attrs)
        normal_vehicles_universal = [
            Vehicle(idx=i, p=p_normal[i], r=r_normal[i], c=c_normal[i])
            for i in range(num_normal)
        ]
        malicious_vehicles_universal = [
            Vehicle(idx=num_normal + i, p=p_malicious[i], r=r_malicious[i], c=c_malicious_shared)
            for i in range(num_malicious)
        ]
        vehicles_universal = normal_vehicles_universal
    else:
        filepath = f'data/{DATASET}.txt'
        df = pd.read_csv(filepath,
                         header=None,
                         names=['index', 'lat_lon', 'taskFare', 'Pro', 'qPro_avg', 'Fare', 'ArrivePro'])
        df = df.groupby('lat_lon').filter(lambda x: x['lat_lon'].count() >= df['lat_lon'].value_counts().nlargest(200).min())
        df['task_id'] = df['lat_lon'].astype('category').cat.codes
        tasks_universal = [Task(idx=j) for j in range(3600)]
        vehicles_universal = []
        for vehicle_id, group in df.groupby('index'):
            p = {row['task_id']: row['ArrivePro'] for _, row in group.iterrows()}
            c = {row['task_id']: np.random.uniform(0, 100) for _, row in group.iterrows()}
            r = group.iloc[0]['qPro_avg']
            vehicles_universal.append(Vehicle(idx=vehicle_id, p=p, r=r, c=c))

        del df

    # running log
    log = {'var': [], 'vals': [], 'info': []}
    # Custom unpickler to handle class path changes
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'src.algos.stochastic.a3c':
                module = 'src.algos.stochastic.rl'
            return super().find_class(module, name)
    log_path = f'logs/{DATASET}-online.pkl'
    if os.path.exists(log_path):
        with open(log_path, 'rb') as f:
            log = CustomUnpickler(f).load()
        all_algorithms = {
            'Oracle': {'class': Oracle, 'costs': [], 'times': [], 'cpu': [], 'memory': []},
            'Exploration-Then-Commit': {'class': SqrtFirst, 'costs': [], 'times': [], 'cpu': [], 'memory': []},
            'StochasticSubmodularPolicy': {'class': StochasticSubmodularPolicy, 'costs': [], 'times': [], 'cpu': [], 'memory': []},
            'StochasticRandomized': {'class': StochasticRandomized, 'costs': [], 'times': [], 'cpu': [], 'memory': []},
            'StochasticGreedyMAB': {'class': StochasticGreedyMAB, 'costs': [], 'times': [], 'cpu': [], 'memory': []},
            'StochasticReinforce' : {'class': StochasticReinforce, 'costs': [], 'times': [], 'cpu': [], 'memory': []},
        }

        # Add missing algorithms to each existing variable section
        algorithms_added = False
        for var_idx, var_info in enumerate(log['info']):
            for algo_name, algo_class in all_algorithms.items():
                if algo_name not in var_info:
                    print(f"Adding missing algorithm '{algo_name}' to existing online log for variable '{log['var'][var_idx]}'")
                    params = algorithm_params_cfg.get(algo_name, {})
                    var_info[algo_name] = {'class': algo_class['class'], 'costs': [], 'times': [], 'params': params}
                    algorithms_added = True
                else:
                    # update params if provided in YAML
                    if algo_name in algorithm_params_cfg:
                        var_info[algo_name]['params'] = algorithm_params_cfg[algo_name]
                    
                    # Backfill missing metrics (cpu, memory)
                    for metric in ['cpu', 'memory']:
                        if metric not in var_info[algo_name]:
                            print(f"Adding missing metric '{metric}' to existing online log for algorithm '{algo_name}'")
                            # Pad with NaNs to match existing length
                            current_len = len(var_info[algo_name]['times'])
                            var_info[algo_name][metric] = [np.nan] * current_len
                            algorithms_added = True

        if algorithms_added:
            print(f"Updated existing online log with missing algorithms")
            # Save the updated log
            with open(log_path, 'wb') as f:
                pickle.dump(log, f)

    var_list = ['n_task', 'n_vehicles', 'beta', 'l']
    for var in var_list:
        if var not in log['var']:
            info = {
                'Oracle': {'class': Oracle, 'costs': [], 'times': [], 'cpu': [], 'memory': [], 'params': algorithm_params_cfg.get('Oracle', {})},
                'Exploration-Then-Commit': {'class': SqrtFirst, 'costs': [], 'times': [], 'cpu': [], 'memory': [], 'params': algorithm_params_cfg.get('Exploration-Then-Commit', {})},
                'StochasticSubmodularPolicy': {'class': StochasticSubmodularPolicy, 'costs': [], 'times': [], 'cpu': [], 'memory': [], 'params': algorithm_params_cfg.get('StochasticSubmodularPolicy', {})},
                'StochasticRandomized': {'class': StochasticRandomized, 'costs': [], 'times': [], 'cpu': [], 'memory': [], 'params': algorithm_params_cfg.get('StochasticRandomized', {})},
                'StochasticReinforce': {'class': StochasticReinforce, 'costs': [], 'times': [], 'cpu': [], 'memory': [], 'params': algorithm_params_cfg.get('StochasticReinforce', {})},
                'StochasticGreedyMAB': {'class': StochasticGreedyMAB, 'costs': [], 'times': [], 'cpu': [], 'memory': [], 'params': algorithm_params_cfg.get('StochasticGreedyMAB', {})},
            }

            log['var'].append(var)
            log['vals'].append(eval(f'{var}_range'))
            log['info'].append(info)
        else:
            info = log['info'][log['var'].index(var)]
            # Update vals in case the range has changed
            log['vals'][log['var'].index(var)] = eval(f'{var}_range')

        # If forced rerun is enabled for selected algorithms and variables, clear their previous results
        if force_rerun and (var in rerun_variables):
            for algo_name, algo_info in info.items():
                if algo_name in rerun_algorithms:
                    print(f"Force rerun enabled: clearing previous results for '{algo_name}' on variable '{var}'")
                    algo_info['times'] = []
                    algo_info['costs'] = []
                    algo_info['cpu'] = []
                    algo_info['memory'] = []
                    # refresh params from YAML if provided
                    if algo_name in algorithm_params_cfg:
                        algo_info['params'] = algorithm_params_cfg[algo_name]

        dataset_name = DATASET.split('/')[-1].split('.')[0]
        for algo_name, algo_info in info.items():
            if algo_name == 'StochasticReinforce':
                algo_info['params']['dataset_name'] = dataset_name

        for idx, ii in enumerate(tqdm(eval(f'{var}_range'), desc=f'Processing {var}')):
            temp_vars = {'n_task': n_task, 'n_vehicles': n_vehicles, 'l': l, 'beta': beta}
            if DATASET == 'synthetic':
                temp_vars['rho'] = rho
            temp_vars[var] = ii
            if DATASET == 'synthetic':
                total_available = len(normal_vehicles_universal) + len(malicious_vehicles_universal)
                desired_total = temp_vars['n_vehicles']
                select_total = min(desired_total, total_available)
                frac_malicious = float(temp_vars.get('rho', 0.0))
                num_malicious_selected = min(int(select_total * frac_malicious), len(malicious_vehicles_universal))
                num_normal_selected = min(select_total - num_malicious_selected, len(normal_vehicles_universal))
                vehicles_selected = malicious_vehicles_universal[:num_malicious_selected] + normal_vehicles_universal[:num_normal_selected]
            else:
                vehicles_selected = vehicles_universal[:temp_vars['n_vehicles']]
            problem = Problem(tasks=tasks_universal[:temp_vars['n_task']],
                              vehicles=vehicles_selected,
                              l=temp_vars['l'], beta=temp_vars['beta'])

            for algo_name, algo_info in info.items():
                if len(algo_info['times']) <= idx:
                    total_time, total_cost, total_penalty, total_cpu, total_mem = 0, 0, 0, 0, 0
                    for _ in range(TIMES):  # Run the algorithm multiple times and average the results is possible
                        start_time = time.time()
                        process = psutil.Process(os.getpid())
                        start_cpu = process.cpu_times()

                        algo_class = algo_info['class']
                        params = algo_info.get('params', {})
                        algo = algo_class(problem, **params)
                        if DATASET == 'synthetic' and var == 'rho':
                            malicious_ids_selected = set([v.idx for v in vehicles_selected[:num_malicious_selected]])
                            threshold_tasks = int(np.ceil(0.1 * len(problem.tasks)))
                            def _run_framework_with_malicious(self):
                                orig_states = {vid: (self.problem.vehicles[vid].r, self.problem.vehicles[vid].sigma) for vid in malicious_ids_selected if vid in self.problem.vehicles}
                                for task_id, task in self.problem.tasks.items():
                                    if task_id < threshold_tasks:
                                        for vid in malicious_ids_selected:
                                            v = self.problem.vehicles.get(vid)
                                            if v is not None:
                                                v.r = 0.9
                                                v.sigma = 0.0
                                    else:
                                        for vid, (r0, s0) in orig_states.items():
                                            v = self.problem.vehicles.get(vid)
                                            if v is not None:
                                                v.r = r0
                                                v.sigma = s0
                                    remained = [v_id for v_id in self.problem.vehicles.keys()
                                                if self.problem.vehicles[v_id].get_expected_cost(task_id) != float('inf')
                                                and self.problem.vehicles[v_id].get_arrival_probability(task_id) > 1e-5
                                                and self.problem.vehicles[v_id].get_execution_ability() > 1e-5]
                                    self.run(task_id, task, remained)
                                return self.recruited_vehicles
                            algo.run_framework = types.MethodType(_run_framework_with_malicious, algo)
                        algo.run_framework()
                        
                        end_cpu = process.cpu_times()
                        end_mem = process.memory_info().rss / (1024 * 1024) # MB
                        cpu_usage = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)

                        total_time += time.time() - start_time
                        total_cost += algo.get_total_expected_cost()
                        total_cpu += cpu_usage
                        total_mem += end_mem
                        # total_penalty += algo.get_total_penalty()
                    algo_info['times'].append(total_time / TIMES)
                    algo_info['costs'].append(total_cost / TIMES)
                    algo_info['cpu'].append(total_cpu / TIMES)
                    algo_info['memory'].append(total_mem / TIMES)
                    # algo_info['penalties'].append(total_penalty / TIMES)
                    print(f"{algo_name}, cost: {total_cost / TIMES}, time: {total_time / TIMES}")

        # save progress
        with open(log_path, 'wb') as f:
            pickle.dump(log, f)

    line_styles = {
        'StochasticSubmodularPolicy': {'color': '#FC3F05', 'marker': 'o', 'label': 'StocRTA-PI', 'ms': 6, 'linestyle': 'dotted'},
        'StochasticRandomized': {'color': '#D4D4D4', 'marker': '^', 'label': 'Random', 'ms': 7, 'linestyle': 'solid'},
        'Exploration-Then-Commit': {'color': '#526CFF', 'marker': 'v', 'label': 'QaP-LDP', 'ms': 5, 'linestyle': 'dashdot'},
        'Oracle': {'color': '#fee5d9', 'marker': 's', 'label': 'Oracle', 'ms': 7, 'linestyle': 'solid'},
        'StochasticReinforce': {'color': '#aa86ca', 'marker': 'p', 'label': 'DDQN', 'ms': 6, 'linestyle': 'dashed'},
        'StochasticGreedyMAB': {'color': '#4ECDC4', 'marker': 'D', 'label': 'UCB', 'ms': 5, 'linestyle': 'dashed'},
    }

    bar_styles = {
        'StochasticSubmodularPolicy': {'color': '#FC3F05', 'hatch': '', 'label': 'StocRTA-PI'},
        'StochasticRandomized': {'color': '#D4D4D4', 'hatch': '', 'label': 'Random'},
        'Exploration-Then-Commit': {'color': '#526CFF', 'hatch': '////', 'label': 'QaP-LDP'},
        'Oracle': {'color': '#fee5d9', 'hatch': '', 'label': 'Oracle'},
        'StochasticReinforce': {'color': '#aa86ca', 'hatch': '\\\\\\', 'label': 'DDQN'},
        'StochasticGreedyMAB': {'color': '#4ECDC4', 'hatch': 'xxx', 'label': 'UCB'},
    }

    # line_styles = {
    #     'MCTA_PI': {'color': '#ffae00', 'marker': 'D', 'label': 'MCTA-PI', 'ms': 5, 'linestyle': 'solid'},
    #     r'Epsilon ($\varepsilon=0.01$)': {'color': '#7292f7', 'marker': 's', 'label': r'Epsilon ($\varepsilon=0.01$)', 'ms': 7, 'linestyle': 'dotted'},
    #     'UCB': {'color': '#eb56c5', 'marker': '<', 'label': 'UCB', 'ms': 7, 'linestyle': 'solid'},
    #     r'Epsilon ($\varepsilon=0.1$)': {'color': '#7b4cff', 'marker': '^', 'label': r'Epsilon ($\varepsilon=0.1$)', 'ms': 7, 'linestyle': 'dashed'},
    #     'LCB': {'color': '#ff3040', 'marker': '>', 'label': 'LCB', 'ms': 7, 'linestyle': 'solid'},
    #     'Exploration-Then-Commit': {'color': '#01c159', 'marker': 'v', 'label': 'Exploration-Then-Commit', 'ms': 7, 'linestyle': 'dashdot'},
    #     # 'Randomized': {'color': '#3753a4', 'marker': '^', 'label': 'Random', 'ms': 7, 'linestyle': 'dashed'},
    #     # 'MinimumCostGreedy': {'color': '#097f80', 'marker': 'v', 'label': 'Minimum-Cost', 'ms': 7, 'linestyle': 'dashdot'},
    # }

    # read the pkl file
    with open(f'logs/{DATASET}-online.pkl', 'rb') as f:
        log = pickle.load(f)


    def plot_regret_bar_chart(var, x_values, algorithms_data, save_dir='figs/online/'):
        x_vals = np.array(x_values)
        idx_all = np.arange(len(x_vals))
        if len(x_vals) >= 8:
            idx_all = np.arange(0, len(x_vals), 2)
        x = np.arange(len(idx_all))
        baseline = np.array(algorithms_data['Oracle']['costs'])[idx_all]

        non_oracle = [name for name in algorithms_data if name != 'Oracle']
        n_bars = len(non_oracle)
        group_width = 0.8
        bar_width = group_width / max(n_bars, 1)
        offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_width

        threshold = max([
            max(np.array(algorithms_data[name]['costs'])[idx_all] - baseline)
            for name in non_oracle if name != 'StochasticRandomized' and name != 'StochasticReinforce'
        ])

        fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.5))
        for idx, algo_name in enumerate(non_oracle):
            y_values = np.array(algorithms_data[algo_name]['costs'])[idx_all] - baseline
            y_values = np.maximum(y_values, 0)
            style = bar_styles[algo_name]
            ax.bar(x + offsets[idx], y_values, width=bar_width, color=style['color'],
                   hatch=style['hatch'], label=style['label'])

        # if DATASET == 'synthetic':
        #     axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
        #     for idx, algo_name in enumerate(non_oracle):
        #         y_values = np.array(algorithms_data[algo_name]['costs'])[idx_all] - baseline
        #         style = bar_styles[algo_name]
        #         axins.bar(x + offsets[idx], y_values, width=bar_width, color=style['color'],
        #                   hatch=style['hatch'], label=style['label'])
        #     axins.tick_params(axis='both', which='both', direction='in', labelsize=6, pad=2)

        xlabel = {
            'n_task': r'\# of Tasks $m$',
            'n_vehicles': r'\# of Vehicles $n$',
            'beta': r'Threshold $\beta$',
            'l': r'Min \# of Vehicles $l$',
        }[var]

        ax.set_ylabel('Regret', fontsize=13)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(10)
        ax.set_ylim(bottom=0, top=threshold * 1.2)
        ax.set_xticks(x)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(1, 2))
        if isinstance(x_values[0], float):
            ax.set_xticklabels([f'{v:.1f}' if var in ('beta', 'rho') else f'{v:.2f}' for v in x_vals[idx_all]])
        else:
            ax.set_xticklabels(list(x_vals[idx_all]))

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}{DATASET}-{var}-regrets.pdf', bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

    def plot_results(var, x_values, algorithms_data, plot_type='times'):
        if plot_type == "times":
            return
        
        if plot_type == 'regrets':
            plot_regret_bar_chart(var, x_values, algorithms_data)
            return

        fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.5))
        for algo_name, algo_info in algorithms_data.items():
            y_values = np.array(algo_info[plot_type])
            x_data = np.array(x_values)
            mask = ~np.isnan(y_values)
            if np.any(mask):
                ax.plot(x_data[mask], y_values[mask], **line_styles[algo_name])
        xlabel = {
            'n_task': r'\# of Tasks $m$',
            'n_vehicles': r'\# of Vehicles $n$',
            'beta': r'Threshold $\beta$',
            'l': r'Min \# of Vehicles $l$',
        }[var]
        ylabel = {'times': 'Running Time (s)',
                    'costs': 'Total Cost',
                    'regrets': 'Regret',
                    'cpu': 'CPU Time (s)',
                    'memory': 'Memory (MB)'}[plot_type]
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(1, 2))
        ax.yaxis.get_offset_text().set_fontsize(10)
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        plt.yticks(rotation=45)

        if plot_type == 'costs':
            ax.set_ylabel('Total Cost', fontsize=13)
            ax.set_yscale('log')

        # Round x-axis labels to 2 decimals if they are floats
        if isinstance(x_values[0], float):
            ax.set_xticks(x_values)
            ax.set_xticklabels([f'{x:.1f}' for x in x_values] if var in ('beta', 'rho') else [f'{x:.2f}' for x in x_values])

        # Ensure the directory exists
        save_dir = 'figs/online/'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}{DATASET}-{var}-{plot_type}.pdf', bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

    # draw picture
    for idx, var in enumerate(log['var']):
        if var == 'rau':
            continue
        plot_results(var, log['vals'][idx], log['info'][idx], plot_type='times')
        plot_results(var, log['vals'][idx], log['info'][idx], plot_type='costs')
        plot_results(var, log['vals'][idx], log['info'][idx], plot_type='regrets')
        plot_results(var, log['vals'][idx], log['info'][idx], plot_type='cpu')
        plot_results(var, log['vals'][idx], log['info'][idx], plot_type='memory')

    m_fix = max(n_task_range)
    n_fix = n_vehicles
    dataset_name = DATASET.split('/')[-1].split('.')[0]
    # Create a legend-only figure with larger font size
    fig_legend = plt.figure(figsize=(3, 1))
    handles, labels = [], []
    for algo_name in line_styles.keys():
        handle = plt.Line2D(
            [0],
            [0],
            linestyle=line_styles[algo_name]['linestyle'],
            color=line_styles[algo_name]['color'],
            marker=line_styles[algo_name]['marker'],
            label=line_styles[algo_name]['label'])
        handles.append(handle)
        labels.append(line_styles[algo_name]['label'])

    fig_legend.legend(handles=handles, labels=labels, loc='center', ncol=6, fontsize=10, frameon=False)
    # Ensure the directory exists
    save_dir = 'figs/online/'
    os.makedirs(save_dir, exist_ok=True)
    fig_legend.savefig(f'{save_dir}{DATASET}-legend.pdf', bbox_inches='tight', pad_inches=0)

    # Create a legend-only figure for bar styles with larger font size
    fig_bar_legend = plt.figure(figsize=(3, 1))
    handles_bar, labels_bar = [], []
    for algo_name in bar_styles.keys():
        patch = mpatches.Patch(
            facecolor=bar_styles[algo_name]['color'],
            hatch=bar_styles[algo_name]['hatch'],
            label=bar_styles[algo_name]['label'],
        )
        handles_bar.append(patch)
        labels_bar.append(bar_styles[algo_name]['label'])
    fig_bar_legend.legend(handles=handles_bar, labels=labels_bar, loc='center', ncol=6, fontsize=6, frameon=False)

    # Ensure the directory exists
    save_dir = 'figs/online/'
    os.makedirs(save_dir, exist_ok=True)
    fig_bar_legend.savefig(f'{save_dir}{DATASET}-bar-legend.pdf', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    main()