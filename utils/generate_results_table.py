#!/usr/bin/env python3
"""
Generate results table from wandb runs or log files.

Usage:
    python utils/generate_results_table.py --wandb-project RL_Apr2 --wandb-group sac_rnn_grid_search_v1
    python utils/generate_results_table.py --log-dir output/slurm_logs/grid-search/354150
"""

import os
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import wandb
import wandb.apis.public as wandb_api


# Define the structure
TASKS = ["HalfCheetah-v5", "Ant-v5", "Hopper-v5", "Walker2d-v5"]
METHODS = ["Vanilla SAC", "DR-Critic", "SA-MLP+DR-Critic", "SAID (Ours)"]
DELAYS = [0, 4, 8, 16]

# Method mapping from experiment config to display name
METHOD_MAPPING = {
    "dummy": "Vanilla SAC",
    "oracle_critic": "DR-Critic",
    "cat_mlp": "Address",  # User's method
    "sac_rnn": "Address",
    "pred_detach": "Address",
    "pred_nodetach": "Address",
    "encode_detach": "Address",
    "encode_nodetach": "Address",
    "symmetric": "Address",
}

# Default to "Address" if method not found
DEFAULT_METHOD = "Address"


def parse_log_file(log_file: Path) -> Optional[Dict]:
    """Parse a log file to extract final evaluation results."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract environment name
        env_match = re.search(r'env\.name:\s*(\S+)', content)
        if not env_match:
            return None
        env_name = env_match.group(1)
        
        # Extract delay
        delay_match = re.search(r'env\.delay:\s*(\d+)', content)
        if not delay_match:
            return None
        delay = int(delay_match.group(1))
        
        # Extract seed
        seed_match = re.search(r'seed:\s*(\d+)', content)
        if not seed_match:
            return None
        seed = int(seed_match.group(1))
        
        # Extract experiment/method - check both experiment= and experiment: patterns
        exp_match = re.search(r'experiment[=:]\s*(\S+)', content)
        method = None
        
        if exp_match:
            method = exp_match.group(1)
        else:
            # Try to find method_type in config tree (format: method_type\n    └── value)
            method_match = re.search(r'method_type\s*\n\s*└──\s*(\S+)', content)
            if method_match:
                method = method_match.group(1)
            else:
                # Try alternative pattern
                method_match = re.search(r'method_type:\s*(\S+)', content)
                if method_match:
                    method = method_match.group(1)
        
        if not method:
            method = "unknown"
        
        # Extract final eval/rew_mean_deterministic (this is the metric we want)
        # Look for the last occurrence in the progress bar or final output
        eval_matches = list(re.finditer(r'eval/rew_mean_deterministic\s+([\d.]+)', content))
        if not eval_matches:
            # Try alternative pattern
            eval_matches = list(re.finditer(r'eval/rew_mean_deterministic[:\s]+([\d.]+)', content))
        
        if not eval_matches:
            return None
        
        final_reward = float(eval_matches[-1].group(1))
        
        return {
            "env": env_name,
            "delay": delay,
            "seed": seed,
            "method": method,
            "reward": final_reward
        }
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return None


def collect_from_logs(log_dir: str) -> Dict:
    """Collect results from log files."""
    results = {}
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return results
    
    # Find all .out files
    log_files = list(log_path.rglob("*.out"))
    print(f"Found {len(log_files)} log files")
    
    for log_file in log_files:
        data = parse_log_file(log_file)
        if data is None:
            continue
        
        env = data["env"]
        delay = data["delay"]
        method = METHOD_MAPPING.get(data["method"], DEFAULT_METHOD)
        seed = data["seed"]
        reward = data["reward"]
        
        # Create key: (env, method, delay, seed)
        key = (env, method, delay, seed)
        if key not in results:
            results[key] = []
        results[key].append(reward)
    
    return results


def collect_from_wandb(project: str, group: Optional[str] = None) -> Dict:
    """Collect results from wandb API."""
    results = {}
    
    api = wandb_api.Api()
    
    # Get runs
    filters = {}
    if group:
        filters["group"] = group
    
    print(f"Fetching runs from wandb project: {project}, group: {group}")
    runs = api.runs(project, filters=filters)
    print(f"Found {len(runs)} wandb runs")
    
    for run in runs:
        try:
            # Get config - wandb stores config in a nested structure
            config = run.config
            
            # Extract environment name - wandb uses slash notation for nested configs
            env = None
            # Wandb stores nested configs as "env/name", "env/delay", etc.
            for key in ["env/name", "env.name", "env_name"]:
                if key in config:
                    env = config[key]
                    if isinstance(env, str):
                        break
            
            # Fallback: try dict structure
            if not env and "env" in config:
                if isinstance(config["env"], dict):
                    env = config["env"].get("name", None)
                elif isinstance(config["env"], str):
                    env = config["env"]
            
            # Extract delay - wandb uses "env/delay"
            delay = None
            for key in ["env/delay", "env.delay", "delay"]:
                if key in config:
                    delay_val = config[key]
                    if isinstance(delay_val, (int, float)):
                        delay = int(delay_val)
                        break
            
            # Fallback: try dict structure
            if delay is None and "env" in config and isinstance(config["env"], dict):
                delay_val = config["env"].get("delay", None)
                if delay_val is not None:
                    delay = int(delay_val) if isinstance(delay_val, (int, float)) else None
            
            # Extract seed
            seed = config.get("seed", None)
            
            # Extract method/experiment
            method_key = None
            if "experiment" in config:
                method_key = config["experiment"]
            elif "method_type" in config:
                method_key = config["method_type"]
            elif "global_cfg" in config:
                # Check global_cfg for method info
                global_cfg = config.get("global_cfg", {})
                if isinstance(global_cfg, dict):
                    actor_input = global_cfg.get("actor_input", {})
                    if isinstance(actor_input, dict):
                        method_key = actor_input.get("history_merge_method", None)
            
            if not method_key:
                # Try to infer from run name or tags
                if hasattr(run, "tags") and run.tags:
                    for tag in run.tags:
                        if tag in METHOD_MAPPING:
                            method_key = tag
                            break
            
            method = METHOD_MAPPING.get(method_key, DEFAULT_METHOD) if method_key else DEFAULT_METHOD
            
            # Get final evaluation reward (eval/rew_mean_deterministic)
            summary = run.summary
            
            # Try multiple possible keys
            reward = None
            for key in ["eval/rew_mean_deterministic", "eval/rew_mean_", "eval/rew_mean", "eval_reward"]:
                if key in summary:
                    reward = summary[key]
                    break
            
            # If not in summary, try to get from history (last value)
            if reward is None:
                try:
                    history = run.history(keys=["eval/rew_mean_deterministic", "eval/rew_mean_"], pandas=False)
                    if history and len(history) > 0:
                        # history is a list of dicts when pandas=False
                        # Get the last non-null value
                        for row in reversed(history):
                            for key in ["eval/rew_mean_deterministic", "eval/rew_mean_"]:
                                if key in row and row[key] is not None:
                                    reward = row[key]
                                    break
                            if reward is not None:
                                break
                except Exception as e:
                    # Try with pandas if available
                    try:
                        history = run.history(keys=["eval/rew_mean_deterministic", "eval/rew_mean_"], pandas=True)
                        if hasattr(history, 'empty') and not history.empty:
                            for col in history.columns:
                                if "rew_mean" in col.lower():
                                    reward = history[col].dropna().iloc[-1] if len(history[col].dropna()) > 0 else None
                                    if reward is not None:
                                        break
                    except:
                        pass
            
            if env and delay is not None and seed is not None and reward is not None:
                key = (env, method, delay, seed)
                if key not in results:
                    results[key] = []
                results[key].append(float(reward))
                print(f"  ✓ Run {run.id}: {env}, delay={delay}, seed={seed}, method={method}, reward={reward:.2f}")
            else:
                missing = []
                if not env: missing.append("env")
                if delay is None: missing.append("delay")
                if seed is None: missing.append("seed")
                if reward is None: missing.append("reward")
                print(f"  ✗ Run {run.id}: Missing {', '.join(missing)}")
                
        except Exception as e:
            print(f"  ✗ Error processing run {run.id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def aggregate_results(results: Dict) -> Dict:
    """Aggregate results by (env, method, delay) and compute mean±std."""
    aggregated = {}
    
    for (env, method, delay, seed), rewards in results.items():
        key = (env, method, delay)
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].extend(rewards)
    
    # Compute statistics
    stats = {}
    for key, values in aggregated.items():
        if len(values) > 0:
            mean = np.mean(values)
            std = np.std(values)
            stats[key] = (mean, std, len(values))
    
    return stats


def format_value(mean: float, std: float, count: int) -> str:
    """Format value as mean±std."""
    if count == 0:
        return "—"
    return f"{mean:.0f}±{std:.0f}"


def generate_table(stats: Dict, output_file: Optional[str] = None) -> str:
    """Generate LaTeX/plain text table in the format requested."""
    lines = []
    
    # Find which methods actually have data
    methods_with_data = set()
    for (env, method, delay) in stats.keys():
        methods_with_data.add(method)
    
    # If no methods found, use default methods
    if not methods_with_data:
        methods_with_data = set(METHODS)
    
    # Create header row: Task Method Action Delay 0 4 8 16
    # Format: Task | Method | Action | Delay | 0 | 4 | 8 | 16
    header = f"{'Task':<20} {'Method':<25} {'Action':<15} {'Delay':<10}"
    for delay in DELAYS:
        header += f" {delay:>15}"
    lines.append(header)
    lines.append("=" * len(header))
    
    # Task rows - each task has multiple methods (but we only show methods with data)
    for task in TASKS:
        first_method = True
        # Only show methods that have data, or show all if user wants
        methods_to_show = sorted(methods_with_data) if methods_with_data else METHODS
        
        for method in methods_to_show:
            if first_method:
                task_col = task
                first_method = False
            else:
                task_col = ""
            
            # Action column: only show "0 (=SAC)" for first method (Vanilla SAC)
            action_col = "0 (=SAC)" if method == "Vanilla SAC" else ""
            
            row = f"{task_col:<20} {method:<25} {action_col:<15} {'':<10}"
            for delay in DELAYS:
                key = (task, method, delay)
                if key in stats:
                    mean, std, count = stats[key]
                    value = format_value(mean, std, count)
                else:
                    value = "—"
                row += f" {value:>15}"
            lines.append(row)
        # Add empty line between tasks
        if task != TASKS[-1]:
            lines.append("")
    
    # Task average row
    lines.append("-" * len(header))
    row = f"{'Task average':<20} {'':<25} {'':<15} {'':<10}"
    for delay in DELAYS:
        # Aggregate across all tasks for this delay
        values = []
        for task in TASKS:
            # Try to find any method that has data for this task and delay
            for method in methods_to_show:
                key = (task, method, delay)
                if key in stats:
                    mean, std, count = stats[key]
                    values.append(mean)
                    break  # Use first method that has data
        
        if len(values) > 0:
            avg_mean = np.mean(values)
            avg_std = np.std(values) if len(values) > 1 else 0
            value = format_value(avg_mean, avg_std, len(values))
        else:
            value = "—"
        row += f" {value:>15}"
    lines.append(row)
    
    table = "\n".join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(table)
        print(f"Table saved to {output_file}")
    
    return table


def main():
    parser = argparse.ArgumentParser(description="Generate results table from wandb or logs")
    parser.add_argument("--wandb-project", type=str, default="RL_Apr2", 
                       help="Wandb project name (default: RL_Apr2)")
    parser.add_argument("--wandb-group", type=str, help="Wandb group name (e.g., sac_rnn_grid_search_v1)")
    parser.add_argument("--log-dir", type=str, help="Directory containing log files (optional)")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--method", type=str, default="sac_rnn", 
                       help="Method name to filter (default: sac_rnn)")
    
    args = parser.parse_args()
    
    results = {}
    
    # Prioritize wandb if both are specified
    if args.wandb_project:
        print("Collecting data from wandb...")
        wandb_results = collect_from_wandb(args.wandb_project, args.wandb_group)
        results.update(wandb_results)
    
    if args.log_dir:
        print("\nCollecting data from log files...")
        log_results = collect_from_logs(args.log_dir)
        results.update(log_results)
    
    if not results:
        print("No results found! Please specify --wandb-project (and optionally --wandb-group)")
        print("\nExample usage:")
        print("  python utils/generate_results_table.py --wandb-project RL_Apr2 --wandb-group sac_rnn_grid_search_v1")
        return
    
    print(f"\nCollected {len(results)} result entries")
    
    # Aggregate
    stats = aggregate_results(results)
    print(f"Aggregated into {len(stats)} (env, method, delay) combinations")
    
    # Generate table
    table = generate_table(stats, args.output)
    print("\n" + table)


if __name__ == "__main__":
    main()
