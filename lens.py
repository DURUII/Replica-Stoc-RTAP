import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os
import time
import scienceplots
from tqdm.auto import tqdm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from src.algos.stochastic.sqrt_first import SqrtFirst
from src.algos.stochastic.proposed import StochasticSubmodularPolicy
from src.algos.stochastic.mab import StochasticGreedyMAB
from src.algos.stochastic.random import StochasticRandomized
from src.algos.stochastic.rl import StochasticReinforce
from src.algos.stochastic.oracle import Oracle
from src.models.bean import Task, Vehicle, Problem

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times'] 
plt.rcParams['text.latex.preamble'] = r"""
\usepackage{amsmath}
\usepackage{amssymb}
"""
plt.style.use(['science', 'ieee'])

line_styles = {
        'StochasticSubmodularPolicy': {'color': '#EA4335', 'label': 'StocRTA-PI', 'linestyle': 'solid', 'zorder': 5, 'linewidth': 2},
        'SqrtFirst': {'color': '#212121', 'label': 'QaP-LDP', 'linestyle': 'solid', 'zorder': 3, 'linewidth': 2},
        'StochasticGreedyMAB': {'color': '#34A853', 'label': 'UCB', 'linestyle': 'solid', 'zorder': 4, 'linewidth': 2},
        'StochasticRandomized': {'color': '#4285F4', 'label': 'Random', 'linestyle': 'solid', 'zorder': 1, 'linewidth': 2},
        'StochasticReinforce': {'color': '#FBBC05', 'label': 'DDQN', 'linestyle': 'solid', 'zorder': 2, 'linewidth': 2},
}

def get_estimation_gap(alg, r_true, n_vehicles):
    if not hasattr(alg, 'logs') or not alg.logs:
        return np.full(len(r_true), np.nan) # Return NaNs if no logs
        
    def _estimate_vector(log_entry):
        vec = np.zeros(n_vehicles, dtype=float)
        for v_id in range(n_vehicles):
            val = log_entry.get(v_id, 0.0)
            if isinstance(val, (np.ndarray, list)):
                arr = np.array(val, dtype=float)
                s = arr.sum()
                if s <= 0:
                    vec[v_id] = 0.0
                else:
                    k = len(arr) - 1
                    weights = arr / s
                    support = np.arange(len(arr), dtype=float) / max(k, 1)
                    vec[v_id] = float(np.dot(weights, support))
            else:
                try:
                    vec[v_id] = float(val)
                except Exception:
                    vec[v_id] = 0.0
        return vec

    r_hat = np.array([_estimate_vector(le) for le in alg.logs], dtype=float)
    
    def _cosine_distance(u, v):
        du = np.linalg.norm(u)
        dv = np.linalg.norm(v)
        if du == 0 or dv == 0:
            return 1.0
        return 1.0 - float(np.dot(u, v) / (du * dv))
        
    distances = np.array([_cosine_distance(row, r_true) for row in r_hat], dtype=float)
    # Pad if necessary to match task length, though logs should match tasks
    return distances

def run_experiment(run_id, n_task=160, n_vehicles=10, l=2, beta=0.8):
    # Set seed for reproducibility of this run
    seed = 923520 + run_id
    np.random.seed(seed)
    
    tasks_universal = [Task(idx=j) for j in range(n_task)]
    vehicles_universal = [Vehicle(idx=j, p=np.random.random((n_task)), 
        c=np.random.uniform(0, 100, (n_task)), r=np.random.rand()) for j in range(n_vehicles)]
    print("[END] Generate Universal Sets")
    
    problem = Problem(tasks=tasks_universal[:n_task],
                      vehicles=vehicles_universal[:n_vehicles],
                      l=l, beta=beta)
    
    r_true = np.array([v.r for v in vehicles_universal[:n_vehicles]], dtype=float)
    
    # Run Oracle
    oracle = Oracle(problem)
    oracle.run_framework()
    oracle_recruited = oracle.recruited_vehicles
    
    results = {}
    
    # Algorithms to run
    algos = [
        (SqrtFirst, {}),
        (StochasticSubmodularPolicy, {}),
        (StochasticGreedyMAB, {}),
        (StochasticRandomized, {}),
        (StochasticReinforce, {'dataset_name': f'run_{run_id}'})
    ]
    
    for AlgoClass, kwargs in algos:
        if AlgoClass == StochasticReinforce:
            # Ensure fresh start for RL
            model_path = f'logs/ddqn_model_run_{run_id}_{n_vehicles}.pth'
            if os.path.exists(model_path):
                os.remove(model_path)
                
        alg_name = AlgoClass.__name__
        print(f'[BEGIN] {alg_name}')
        t_start = time.process_time()
        alg = AlgoClass(problem, **kwargs)
        alg.run_framework()
        t_end = time.process_time()
        print(f'[END] {alg_name}')
        
        # Metrics
        recruited = alg.recruited_vehicles
        
        # 1. Estimation Gap (if applicable)
        est_gap = get_estimation_gap(alg, r_true, n_vehicles)
        
        # 2. Decision Gap (Jaccard)
        decision_gaps = []
        for t_id in range(n_task):
            set_a = recruited.get(t_id, set())
            if isinstance(set_a, list):
                set_a = set(set_a)
            
            set_o = oracle_recruited.get(t_id, set())
            if isinstance(set_o, list):
                set_o = set(set_o)
                
            union = len(set_a.union(set_o))
            if union == 0:
                decision_gaps.append(0.0)
            else:
                jaccard = len(set_a.intersection(set_o)) / union
                decision_gaps.append(1.0 - jaccard)
        
        # 3. Cost Gap
        cost_gaps = []
        for t_id in range(n_task):
            vehicles_a = recruited.get(t_id, set())
            vehicles_o = oracle_recruited.get(t_id, set())
            
            cost_a = sum(problem.vehicles[v].get_expected_cost(t_id) for v in vehicles_a)
            cost_o = sum(problem.vehicles[v].get_expected_cost(t_id) for v in vehicles_o)
            cost_gaps.append(cost_a - cost_o)
            
        results[alg_name] = {
            'estimation_gap': est_gap,
            'decision_gap': np.array(decision_gaps),
            'cost_gap': np.array(cost_gaps),
            'recruited': recruited,
            'cpu_time': t_end - t_start
        }
        
    return results, oracle_recruited, problem



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun', action='store_true', help='Rerun the experiments even if logs exist')
    args = parser.parse_args()

    n_runs = 5
    n_task = 160000
    n_vehicles = 10
    l, beta = 2, 0.8
    pickle_file = f'./logs/synthetic-lens-m{n_task}-n{n_vehicles}.pkl'
    
    if os.path.exists(pickle_file) and not args.rerun:
        print(f"Loading results from {pickle_file}")
        with open(pickle_file, 'rb') as f:
            all_results = pickle.load(f)
    else:
        print(f"Running {n_runs} experiments...")
        all_results = []
        for i in tqdm(range(n_runs)):
            res, oracle_rec, prob = run_experiment(i, n_task, n_vehicles, l, beta)
            # We only store the metrics and the last run's detailed data for heatmap
            all_results.append({
                'metrics': res,
                'oracle_recruited': oracle_rec if i == 0 else None, # Store only for first run
                'problem_vehicles': prob.vehicles if i == 0 else None # Store only for first run
            })
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(all_results, f)
            
def process_and_plot(all_results, n_task, n_vehicles):
    # Prepare DataFrame for plotting
    data_est = []
    data_dec = []
    data_cost = []
    data_acc_cost = []
    
    for run_idx, run_data in enumerate(all_results):
        metrics = run_data['metrics']
        for alg_name, res in metrics.items():
            # Estimation Gap
            if len(res['estimation_gap']) > 0 and not np.all(np.isnan(res['estimation_gap'])):
                # Handle case where logs might be shorter than n_task (though shouldn't be)
                length = min(n_task, len(res['estimation_gap']))
                for t in range(length):
                    data_est.append({
                        'Task Index': t,
                        'Algorithm': alg_name,
                        'Run': run_idx,
                        'Value': res['estimation_gap'][t]
                    })
            
            # Decision Gap
            for t in range(n_task):
                data_dec.append({
                    'Task Index': t,
                    'Algorithm': alg_name,
                    'Run': run_idx,
                    'Value': res['decision_gap'][t]
                })
                
            # Cost Gap
            for t in range(n_task):
                data_cost.append({
                    'Task Index': t,
                    'Algorithm': alg_name,
                    'Run': run_idx,
                    'Value': res['cost_gap'][t]
                })

    df_est = pd.DataFrame(data_est)
    df_dec = pd.DataFrame(data_dec)
    df_cost = pd.DataFrame(data_cost)
    df_acc_cost = pd.DataFrame(data_acc_cost)
    
    os.makedirs('figs/lens', exist_ok=True)
    
    # Plotting Helper
    def plot_metric(df, filename, ylabel, yscale='linear', smoothing_window=None):
        if df.empty:
            return
        
        # Smoothing
        if smoothing_window is not None and smoothing_window > 1:
            # Apply smoothing per algorithm per run using pandas rolling
            # This preserves the error bands (variance between smoothed runs)
            df['Value'] = df.groupby(['Algorithm', 'Run'])['Value'].transform(
                lambda x: x.rolling(window=smoothing_window, min_periods=1, center=True).mean()
            )

        plt.figure(figsize=(3, 2))
        # Map algorithm names to labels
        df['Algorithm Label'] = df['Algorithm'].apply(lambda x: line_styles.get(x, {}).get('label', x))
        
        # Get unique algorithms present in df
        algos = df['Algorithm'].unique()
        
        for alg in algos:
            subset = df[df['Algorithm'] == alg]
            style = line_styles.get(alg, {})
            # Use seaborn for error bands
            sns.lineplot(data=subset, x='Task Index', y='Value', 
                         label=style.get('label', alg), color=style.get('color'), 
                         linestyle=style.get('linestyle'), zorder=style.get('zorder', 1),
                         linewidth=style.get('linewidth', 1.5))
            
        plt.xlabel(r'Round $j$', fontsize=13)
        plt.ylabel(ylabel, fontsize=13)
        # plt.legend(loc='upper right') #, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.legend('', frameon=False)
        if yscale == 'log':
            plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(f'figs/lens/{filename}')
        plt.close()

    # Adjustable smoothing strength
    smooth_window = 500 if n_task > 10000 else None

    # Fig 1. Estimation Gap
    plot_metric(df_est, f'estimation-gap-m{n_task}-n{n_vehicles}.pdf', r'Estimation Gap', yscale='log', smoothing_window=None)
    
    # Fig 2. Decision Gap
    plot_metric(df_dec, f'decision-gap-m{n_task}-n{n_vehicles}.pdf', r'Action Gap', smoothing_window=smooth_window)
    
    # Fig 3. Cost Gap
    cost_yscale = 'log' if n_vehicles > 1000 else 'linear'
    plot_metric(df_cost, f'cost-gap-m{n_task}-n{n_vehicles}.pdf', r'Cost Gap', yscale=cost_yscale, smoothing_window=smooth_window)
    
    # Legend
    fig_legend = plt.figure(figsize=(3, 1))
    handles, labels = [], []
    for algo_name in line_styles.keys():
        handle = plt.Line2D(
            [0],
            [0],
            linestyle=line_styles[algo_name]['linestyle'],
            color=line_styles[algo_name]['color'],
            label=line_styles[algo_name]['label'])
        handles.append(handle)
        labels.append(line_styles[algo_name]['label'])

    fig_legend.legend(handles=handles, labels=labels, loc='center', ncol=6, fontsize=10, frameon=False)
    plt.savefig(f'figs/lens/legend.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Fig 4. Selection Heatmap (For the first run)
    if n_task <= 200 and n_vehicles <= 200:
        first_run = all_results[0]
        oracle_rec = first_run['oracle_recruited']
        metrics = first_run['metrics']
        
        # Define colormap
        # 0: None (#34a853 - Green? Wait, user said #34a853: Not selected)
        # 1: Online (#4285f4 - Blue)
        # 2: Oracle (#ea4335 - Red)
        # 3: Both (#673ab7 - Purple)
        cmap = ListedColormap(['#34a853', '#4285f4', '#ea4335', '#673ab7'])
        bounds = [0, 1, 2, 3, 4]
        norm = BoundaryNorm(bounds, cmap.N)
        
        for alg_name, res in metrics.items():
            recruited = res['recruited']
            
            # Create grid
            grid = np.zeros((n_vehicles, n_task), dtype=int)
            
            for t in range(n_task):
                set_a = recruited.get(t, set())
                if isinstance(set_a, list):
                    set_a = set(set_a)
                
                set_o = oracle_rec.get(t, set())
                if isinstance(set_o, list):
                    set_o = set(set_o)
                
                for v in range(n_vehicles):
                    in_a = v in set_a
                    in_o = v in set_o
                    
                    if not in_a and not in_o:
                        val = 0
                    elif in_a and not in_o:
                        val = 1
                    elif not in_a and in_o:
                        val = 2
                    else: # Both
                        val = 3
                    grid[v, t] = val
                    
            plt.figure(figsize=(4, 2))
            # pcolormesh
            plt.pcolormesh(grid, cmap=cmap, norm=norm, edgecolors='w', linewidth=0.05)
            plt.xlabel(r'Round $j$', fontsize=13)
            plt.ylabel(r'Vehicle Index', fontsize=13)
            # plt.title(line_styles.get(alg_name, {}).get('label', alg_name))
            
            # Create legend manually
            patches = [
                mpatches.Patch(color='#34a853', label='None'),
                mpatches.Patch(color='#4285f4', label='Online'),
                mpatches.Patch(color='#ea4335', label='Oracle'),
                mpatches.Patch(color='#673ab7', label='Both')
            ]
            #plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
            plt.legend('', frameon=False)
            plt.tight_layout()
            plt.savefig(f'figs/lens/heatmap-{alg_name}-m{n_task}-n{n_vehicles}.pdf')
            plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun', action='store_true', help='Rerun the experiments even if logs exist')
    parser.add_argument('--update', action='store_true', help='Update plots from all existing logs')
    args = parser.parse_args()

    if args.update:
        import re
        log_dir = './logs'
        # Pattern to match synthetic-lens-m{n_task}-n{n_vehicles}.pkl
        pattern = re.compile(r'synthetic-lens-m(\d+)-n(\d+)\.pkl')
        
        if not os.path.exists(log_dir):
            print("No logs directory found.")
            return

        files = [f for f in os.listdir(log_dir) if pattern.match(f)]
        print(f"Found {len(files)} log files to process.")

        for filename in files:
            match = pattern.match(filename)
            n_task_extracted = int(match.group(1))
            n_vehicles_extracted = int(match.group(2))
            
            filepath = os.path.join(log_dir, filename)
            print(f"Processing {filepath}...")
            try:
                with open(filepath, 'rb') as f:
                    all_results = pickle.load(f)
                process_and_plot(all_results, n_task_extracted, n_vehicles_extracted)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                
    else:
        n_runs = 5
        n_task = 160000
        n_vehicles = 10
        l, beta = 2, 0.8
        pickle_file = f'./logs/synthetic-lens-m{n_task}-n{n_vehicles}.pkl'
        
        if os.path.exists(pickle_file) and not args.rerun:
            print(f"Loading results from {pickle_file}")
            with open(pickle_file, 'rb') as f:
                all_results = pickle.load(f)
        else:
            print(f"Running {n_runs} experiments...")
            all_results = []
            for i in tqdm(range(n_runs)):
                res, oracle_rec, prob = run_experiment(i, n_task, n_vehicles, l, beta)
                # We only store the metrics and the last run's detailed data for heatmap
                all_results.append({
                    'metrics': res,
                    'oracle_recruited': oracle_rec if i == 0 else None, # Store only for first run
                    'problem_vehicles': prob.vehicles if i == 0 else None # Store only for first run
                })
            
            with open(pickle_file, 'wb') as f:
                pickle.dump(all_results, f)
        
        process_and_plot(all_results, n_task, n_vehicles)

if __name__ == '__main__':
    main()