from tqdm.auto import tqdm
import pickle
import time
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import psutil
from src.algos.offline.improved import SubmodularPolicyImproved
from src.algos.offline.proposed import SubmodularPolicy
from src.algos.offline.random import Randomized
from src.algos.offline.greedy import MinimumCostGreedy
from src.algos.offline.genetic import GeneticAlgorithm
from src.models.bean import *
import numpy as np
np.random.seed(42)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times'] 
plt.rcParams['text.latex.preamble'] = r"""
\usepackage{amsmath}
\usepackage{amssymb}
"""

import scienceplots
plt.style.use(['science'])

from multiprocessing import Pool

def run_command(command):
    os.system(command)

def main():
    # command line arguments (dataset)
    parser = argparse.ArgumentParser(description='offline algorithm runner')
    parser.add_argument('-D', type=str, choices=['CD18', 'CD14', 'XA18', 'SYN', 'ALL'], required=True, help='dataset to use')
    parser.add_argument('-C', type=str, default='configs/offline.yaml', help='path to YAML config for rerun settings')
    parser.add_argument('--only-legend', action='store_true')
    args = parser.parse_args()

    if args.D == 'ALL':
        commands = [
            'python offline.py -D CD18',
            'python offline.py -D CD14',
            'python offline.py -D XA18',
            'python offline.py -D SYN'
        ]
        with Pool(4) as p:
            p.map(run_command, commands)
        exit()

    # configuration
    DATASET = {'CD18': 'chengdu2018', 'CD14': 'chengdu2014', 'XA18': 'xian2018', 'SYN': 'synthetic'}[args.D]

    line_styles = {
        'SubmodularPolicy': {'color': '#060506', 'marker': 's', 'label': 'RTA-P', 'ms': 7, 'linestyle': 'solid'},
        'SubmodularPolicyImproved': {'color': '#ed1e25', 'marker': 'o', 'label': 'RTA-PI', 'ms': 7, 'linestyle': 'dotted'},
        'Randomized': {'color': '#097f80', 'marker': '^', 'label': 'Random', 'ms': 10, 'linestyle': 'dashed'},
        'MinimumCostGreedy': {'color': '#3753a4', 'marker': 'v', 'label': 'Minimum-Cost', 'ms': 10, 'linestyle': 'dashdot'},
        'GeneticAlgorithm': {'color': '#bb9727', 'marker': 'D', 'label': 'GA', 'ms': 8, 'linestyle': 'solid'},
    }

    if args.only_legend:
        save_dir = 'figs/offline/'
        os.makedirs(save_dir, exist_ok=True)
        fig_legend = plt.figure(figsize=(5, 1))
        handles, labels = [], []
        for algo_name in line_styles:
            handle = plt.Line2D([0], [0], linestyle=line_styles[algo_name]['linestyle'], color=line_styles[algo_name]['color'], marker=line_styles[algo_name]['marker'], label=line_styles[algo_name]['label'])
            handles.append(handle)
            labels.append(line_styles[algo_name]['label'])
        fig_legend.legend(handles=handles, labels=labels, loc='center', ncol=len(handles), fontsize=10, frameon=False)
        fig_legend.savefig(f'{save_dir}{DATASET}-legend.pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig_legend)

        if DATASET == 'synthetic':
            fig_legend_rau = plt.figure(figsize=(5, 1))
            handles2, labels2 = [], []
            for algo_name in line_styles:
                if algo_name == 'Randomized':
                    continue
                handle = plt.Line2D([0], [0], linestyle=line_styles[algo_name]['linestyle'], color=line_styles[algo_name]['color'], marker=line_styles[algo_name]['marker'], label=line_styles[algo_name]['label'])
                handles2.append(handle)
                labels2.append(line_styles[algo_name]['label'])
            fig_legend_rau.legend(handles=handles2, labels=labels2, loc='center', ncol=len(handles2), fontsize=10, frameon=False)
            fig_legend_rau.savefig(f'{save_dir}{DATASET}-rau-legend.pdf', bbox_inches='tight', pad_inches=0)
            plt.close(fig_legend_rau)
        return

    n_task = 50
    n_task_range = range(30, 150, 20)

    n_vehicles = 200
    n_vehicles_range = list(range(100, 1001, 100))
    
    l = 3
    l_range = list([2, 4, 6, 8])

    beta = 0.75
    beta_range = np.arange(0.1, 0.9, 0.2)

    if DATASET == 'synthetic':
        n_vehicles_range = list([100, 500, 1000, 5000, 10000])

        n_task = 20
        
        rau = 0.0
        rau_range = np.arange(0.1, 0.8, 0.2)

    # Ensure logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Load YAML configuration (optional)
    cfg = {}
    if os.path.exists(args.C):
        try:
            with open(args.C, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load YAML config '{args.C}': {e}. Proceeding with defaults.")

    offline_cfg = cfg.get('offline') or {}
    rerun_cfg = offline_cfg.get('rerun') or {}
    rerun_algorithms = set(rerun_cfg.get('algorithms') or [])
    rerun_variables = set(rerun_cfg.get('variables') or ['n_task', 'n_vehicles', 'beta', 'l'])
    force_rerun = bool(rerun_cfg.get('force', False))
    algorithm_params_cfg = offline_cfg.get('algorithm_params') or {}

    if DATASET == 'synthetic':
        tasks_universal = [Task(idx=i) for i in range(7000)]
        num_tasks_for_attrs = 200
        num_normal = 10000
        num_malicious = 10000
        p_normal = np.random.random((num_normal, num_tasks_for_attrs))
        r_normal = np.random.random(num_normal)
        c_normal = np.random.uniform(1, 100, (num_normal, num_tasks_for_attrs))
        p_malicious = np.random.random((num_malicious, num_tasks_for_attrs))
        c_malicious_shared = np.ones(num_tasks_for_attrs)
        normal_vehicles_universal = [
            Vehicle(idx=i, p=p_normal[i], r=r_normal[i], c=c_normal[i])
            for i in range(num_normal)
        ]
        malicious_vehicles_universal = [
            Vehicle(idx=num_normal + i, p=p_malicious[i], r=0.001, c=c_malicious_shared)
            for i in range(num_malicious)
        ]
    else:
        filepath = f'data/{DATASET}.txt'
        df = pd.read_csv(filepath,
                         header=None,
                         names=['index', 'lat_lon', 'taskFare', 'Pro', 'qPro_avg', 'Fare', 'ArrivePro'])
        df = df.groupby('lat_lon').filter(lambda x: x['lat_lon'].count() >= df['lat_lon'].value_counts().nlargest(200).min())
        df['task_id'] = df['lat_lon'].astype('category').cat.codes + 1

        tasks_universal = [Task(idx=task_id) for task_id in df['task_id'].unique()]
        vehicles_universal = []
        for vehicle_id, group in df.groupby('index'):
            p = {row['task_id']: row['ArrivePro'] for _, row in group.iterrows()}
            c = {row['task_id']: np.random.uniform(1, 100) for _, row in group.iterrows()}
            r = group.iloc[0]['qPro_avg']
            vehicles_universal.append(Vehicle(idx=vehicle_id, p=p, r=r, c=c))

    # running
    log = {'var': [], 'vals': [], 'info': []}

    # load existing log if it exists to support resuming
    log_path = f'logs/{DATASET}-offline.pkl'
    if os.path.exists(log_path):
        with open(log_path, 'rb') as f:
            log = pickle.load(f)

        # Check for missing algorithms and add them to existing logs
        all_algorithms = {
            'Randomized': {'class': Randomized, 'costs': [], 'times': [], 'cpu': [], 'memory': []},
            'MinimumCostGreedy': {'class': MinimumCostGreedy, 'costs': [], 'times': [], 'cpu': [], 'memory': []},
            'SubmodularPolicy': {'class': SubmodularPolicy, 'costs': [], 'times': [], 'cpu': [], 'memory': []},
            'SubmodularPolicyImproved': {'class': SubmodularPolicyImproved, 'costs': [], 'times': [], 'cpu': [], 'memory': []},
            'GeneticAlgorithm': {'class': GeneticAlgorithm, 'costs': [], 'times': [], 'cpu': [], 'memory': []}
        }

        # Add missing algorithms to each existing variable section
        algorithms_added = False
        for var_idx, var_info in enumerate(log['info']):
            for algo_name, algo_class in all_algorithms.items():
                if algo_name not in var_info:
                    print(f"Adding missing algorithm '{algo_name}' to existing log for variable '{log['var'][var_idx]}'")
                    # attach params from YAML if provided
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
                            print(f"Adding missing metric '{metric}' to existing log for algorithm '{algo_name}'")
                            # Pad with NaNs to match existing length
                            current_len = len(var_info[algo_name]['times'])
                            var_info[algo_name][metric] = [np.nan] * current_len
                            algorithms_added = True

        if algorithms_added:
            print(f"Updated existing log with missing algorithms")
            # Save the updated log
            with open(log_path, 'wb') as f:
                pickle.dump(log, f)

    var_list = ['n_task', 'n_vehicles', 'beta', 'l']
    if DATASET == 'synthetic':
        var_list.append('rau')
    for var in var_list:
        if var not in log['var']:
            info = {
                'Randomized': {'class': Randomized, 'costs': [], 'times': [], 'cpu': [], 'memory': [], 'params': algorithm_params_cfg.get('Randomized', {})},
                'MinimumCostGreedy': {'class': MinimumCostGreedy, 'costs': [], 'times': [], 'cpu': [], 'memory': [], 'params': algorithm_params_cfg.get('MinimumCostGreedy', {})},
                'SubmodularPolicy': {'class': SubmodularPolicy, 'costs': [], 'times': [], 'cpu': [], 'memory': [], 'params': algorithm_params_cfg.get('SubmodularPolicy', {})},
                'SubmodularPolicyImproved': {'class': SubmodularPolicyImproved, 'costs': [], 'times': [], 'cpu': [], 'memory': [], 'params': algorithm_params_cfg.get('SubmodularPolicyImproved', {})},
                'GeneticAlgorithm': {'class': GeneticAlgorithm, 'costs': [], 'times': [], 'cpu': [], 'memory': [], 'params': algorithm_params_cfg.get('GeneticAlgorithm', {})}}

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

        for idx, ii in enumerate(tqdm(eval(f'{var}_range'), desc=f'Processing {var}')):
            temp_vars = {'n_task': n_task, 'n_vehicles': n_vehicles, 'l': l, 'beta': beta}
            if DATASET == 'synthetic':
                temp_vars['rau'] = rau
            temp_vars[var] = ii

            if DATASET == 'synthetic':
                total_available = len(normal_vehicles_universal) + len(malicious_vehicles_universal)
                desired_total = temp_vars['n_vehicles']
                select_total = min(desired_total, total_available)
                frac_malicious = float(temp_vars.get('rau', 0.0))
                num_malicious = min(int(select_total * frac_malicious), len(malicious_vehicles_universal))
                num_normal = min(select_total - num_malicious, len(normal_vehicles_universal))
                vehicles_selected = malicious_vehicles_universal[:num_malicious] + normal_vehicles_universal[:num_normal]
            else:
                vehicles_selected = vehicles_universal[:temp_vars['n_vehicles']]
            
            problem = Problem(tasks=tasks_universal[:temp_vars['n_task']],
                              vehicles=vehicles_selected,
                              l=temp_vars['l'], beta=temp_vars['beta'])

            for algo_name, algo_info in info.items():
                need_run = False
                if len(algo_info['times']) <= idx:
                    need_run = True
                else:
                    try:
                        need_run = np.isnan(algo_info['times'][idx])
                    except Exception:
                        need_run = False
                if DATASET == 'synthetic' and var == 'n_vehicles' and algo_name == 'GeneticAlgorithm' and temp_vars['n_vehicles'] > 1000:
                    if len(algo_info['times']) <= idx:
                        algo_info['times'].append(np.nan)
                        algo_info['costs'].append(np.nan)
                        algo_info['cpu'].append(np.nan)
                        algo_info['memory'].append(np.nan)
                    else:
                        algo_info['times'][idx] = np.nan
                        algo_info['costs'][idx] = np.nan
                        algo_info['cpu'][idx] = np.nan
                        algo_info['memory'][idx] = np.nan
                    continue
                if not need_run:
                    continue
                start_time = time.time()
                process = psutil.Process(os.getpid())
                start_cpu = process.cpu_times()
                algo_class = algo_info['class']
                params = algo_info.get('params', {})
                algo = algo_class(problem, **params)
                print(f"[START] Running {algo_name} with {temp_vars}")
                algo.run_framework()
                end_cpu = process.cpu_times()
                end_mem = process.memory_info().rss / (1024 * 1024)
                cpu_usage = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
                if len(algo_info['times']) <= idx:
                    algo_info['times'].append(time.time() - start_time)
                    algo_info['costs'].append(algo.get_total_expected_cost())
                    algo_info['cpu'].append(cpu_usage)
                    algo_info['memory'].append(end_mem)
                else:
                    algo_info['times'][idx] = time.time() - start_time
                    algo_info['costs'][idx] = algo.get_total_expected_cost()
                    algo_info['cpu'][idx] = cpu_usage
                    algo_info['memory'][idx] = end_mem

        # save progress
        with open(log_path, 'wb') as f:
            pickle.dump(log, f)

    

    # line_styles = {
    #     'SubmodularPolicy': {'color': '#747474', 'marker': 's', 'label': 'MCTA-P', 'ms': 6, 'linestyle': 'solid'},
    #     'MCTA_PI': {'color': '#ec3f4c', 'marker': 'o', 'label': 'MCTA-PI', 'ms': 5, 'linestyle': 'dotted'},
    #     'Randomized': {'color': '#ffae00', 'marker': '^', 'label': 'Random', 'ms': 7, 'linestyle': 'dashed'},
    #     'MinimumCostGreedy': {'color': '#7292f7', 'marker': 'v', 'label': 'Minimum-Cost', 'ms': 7, 'linestyle': 'dashdot'},
    # }


    # read the pkl file
    with open(f'logs/{DATASET}-offline.pkl', 'rb') as f:
        log = pickle.load(f)


    def plot_results(var, x_values, algorithms_data, plot_type='times'):
        """
        Plot the results of the algorithms.

        Args:
            var (str): The variable being plotted.
            x_values (list): The x-axis values.
            algorithms_data (dict): The data for each algorithm.
            plot_type (str): The type of plot ('times' or 'costs').
        """

        if plot_type == "times":
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(2.6, 1.7))

        try:
            ax.set_yscale('log')
        except Exception:
            pass

        for algo_name, algo_info in algorithms_data.items():
            if DATASET == 'synthetic' and var == 'rau' and algo_name == 'Randomized':
                continue
            y_data = np.array(algo_info.get(plot_type, []))
            x_data = np.array(x_values)
            length = min(len(x_data), len(y_data))
            if length == 0:
                continue
            x_plot = x_data[:length]
            y_plot = y_data[:length]
            y_plot = np.clip(y_plot, 1e-8, None)
            mask = ~np.isnan(y_plot)
            if np.any(mask):
                ax.plot(x_plot[mask], y_plot[mask], **line_styles[algo_name])
        xlabel = {
            'n_task': r'\# of Tasks $m$',
            'n_vehicles': r'\# of Vehicles $n$',
            'beta': r'Threshold $\beta$',
            'l': r'Min \# of Vehicles $l$',
            'rau': r'Malicious Ratio $\rho$',
        }[var]
        ylabel = {'times': 'Running Time (s)',
                    'costs': 'Total Cost',
                    'cpu': 'CPU Time (s)',
                    'memory': 'Memory (MB)'}[plot_type]
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(1, 2))
        #ax.ticklabel_format(axis='y', style='sci', scilimits=(1, 2))
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        # ax.tick_params(axis='y', labelrotation=45)
        # plt.setp(ax.get_yticklabels(), rotation=45)

        if DATASET == 'synthetic' and var == 'n_vehicles':
            try:
                ax.set_xscale('log')
            except Exception:
                pass

        save_dir = 'figs/offline/'
        os.makedirs(save_dir, exist_ok=True)

        plt.savefig(f'{save_dir}{DATASET}-{var}-{plot_type}.pdf', bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)


    # draw picture
    for idx, var in enumerate(log['var']):
        plot_results(var, log['vals'][idx], log['info'][idx], plot_type='times')
        plot_results(var, log['vals'][idx], log['info'][idx], plot_type='costs')
        plot_results(var, log['vals'][idx], log['info'][idx], plot_type='cpu')
        plot_results(var, log['vals'][idx], log['info'][idx], plot_type='memory')

    # Create a legend-only figure with larger font size
    fig_legend = plt.figure(figsize=(5, 1))
    handles, labels = [], []
    for algo_name in line_styles:
        handle = plt.Line2D(
            [0],
            [0],
            linestyle=line_styles[algo_name]['linestyle'],
            color=line_styles[algo_name]['color'],
            marker=line_styles[algo_name]['marker'],
            label=line_styles[algo_name]['label'])
        handles.append(handle)
        labels.append(line_styles[algo_name]['label'])

    fig_legend.legend(handles=handles, labels=labels, loc='center', ncol=len(handles), fontsize=10, frameon=False)
    fig_legend.savefig(f'figs/offline/{DATASET}-legend.pdf', bbox_inches='tight', pad_inches=0)

    if DATASET == 'synthetic':
        fig_legend_rau = plt.figure(figsize=(5, 1))
        handles2, labels2 = [], []
        for algo_name in line_styles:
            if algo_name == 'Randomized':
                continue
            handle = plt.Line2D([0], [0], linestyle=line_styles[algo_name]['linestyle'], color=line_styles[algo_name]['color'], marker=line_styles[algo_name]['marker'], label=line_styles[algo_name]['label'])
            handles2.append(handle)
            labels2.append(line_styles[algo_name]['label'])
        fig_legend_rau.legend(handles=handles2, labels=labels2, loc='center', ncol=len(handles2), fontsize=10, frameon=False)
        fig_legend_rau.savefig(f'figs/offline/{DATASET}-rau-legend.pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig_legend_rau)

if __name__ == '__main__':
    main()