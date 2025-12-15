import json
import subprocess
import shutil
import random
import time
from pathlib import Path
from itertools import product

CONFIG_PATH = Path("config.json")
BACKUP_PATH = Path("config.json.bak")

def run_pipeline(run_name):
    print(f"\n[Experiment] Starting run: {run_name}")
    try:
        subprocess.run(
            ["uv", "run", "python", "-m", "gamemaster.evaluation.run"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[Experiment] Run {run_name} failed: {e}")

def main():
    if not CONFIG_PATH.exists():
        print(f"Error: {CONFIG_PATH} not found.")
        return

    print(f"Backing up {CONFIG_PATH} to {BACKUP_PATH}...")
    shutil.copy(CONFIG_PATH, BACKUP_PATH)

    try:
        with open(CONFIG_PATH, "r") as f:
            base_config = json.load(f)

        param_grid = {
            "top_k_games": [2, 3],
            "top_k_reviews": [5, 10],
            "mix_games": [
                [0.8, 0.2],
                [0.5, 0.5]
            ],
            "mix_reviews": [
                [0.6, 0.3, 0.1],
                [0.4, 0.4, 0.2]
            ]
        }
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        all_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        print(f"Total possible combinations: {len(all_combinations)}")
        
        selected_runs = all_combinations
        
        print(f"Selected {len(selected_runs)} runs for execution.")

        for i, params in enumerate(selected_runs, 1):
            print(f"\n--- Run {i}/{len(selected_runs)} ---")
            print("Params:", json.dumps(params, indent=2))
            
            
            if "retrieval" not in base_config: base_config["retrieval"] = {}
            base_config["retrieval"]["top_k_games"] = params["top_k_games"]
            base_config["retrieval"]["top_k_reviews"] = params["top_k_reviews"]
            base_config["retrieval"]["mix_games"] = params["mix_games"]
            base_config["retrieval"]["mix_reviews"] = params["mix_reviews"]
            
            run_id = f"sweep_{i}_G{params['top_k_games']}_R{params['top_k_reviews']}"
            
            if "wandb" not in base_config: base_config["wandb"] = {}
            base_config["wandb"]["wandb_run_name"] = run_id
            
            with open(CONFIG_PATH, "w") as f:
                json.dump(base_config, f, indent=2)
            
            run_pipeline(run_id)
            
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nError during experiments: {e}")
    finally:
        if BACKUP_PATH.exists():
            print(f"\nRestoring original config from {BACKUP_PATH}...")
            shutil.copy(BACKUP_PATH, CONFIG_PATH)
        print("Done.")

if __name__ == "__main__":
    main()
