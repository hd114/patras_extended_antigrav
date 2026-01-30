# experiment_runner.py

import yaml
import os
import sys
import copy
import json
import pandas as pd
from datetime import datetime
import multiprocessing as mp

# Ensure we can import from src if running from scripts/ or root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.training.run_qat_brevitas_structured import run_qat_training_pipeline

EXPERIMENTS_CONFIG_PATH = "config/experiments.yaml"
BASE_CONFIG_PATH = "config.yaml"
NUM_REPETITIONS_PER_CONFIG = 5
MAX_PARALLEL_PROCESSES = 6

def deep_update(source_dict: dict, overrides: dict) -> dict:
    updated_dict = copy.deepcopy(source_dict)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in updated_dict and isinstance(updated_dict[key], dict):
            updated_dict[key] = deep_update(updated_dict[key], value)
        else:
            updated_dict[key] = value
    return updated_dict

def get_nested_val(data_dict: dict, path: list, default: any = None) -> any:
    current = data_dict
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def run_single_experiment_wrapper(args_tuple):
    config_set_name, i_rep, temp_config_path, num_total_reps, unique_run_output_dir = args_tuple
    print(f"Starting pipeline run for: {config_set_name}, Rep {i_rep}/{num_total_reps}")
    
    try:
        pipeline_result_dir = run_qat_training_pipeline(config_path=temp_config_path)
        
        # Enforce result directory consistency
        if not pipeline_result_dir or not os.path.samefile(pipeline_result_dir, unique_run_output_dir):
            pipeline_result_dir = unique_run_output_dir

        if pipeline_result_dir and os.path.isdir(pipeline_result_dir):
            found_json_log = None
            for fname in os.listdir(pipeline_result_dir):
                if fname.endswith("_full_run_log.json"):
                    found_json_log = os.path.join(pipeline_result_dir, fname)
                    break
            
            if found_json_log:
                return {
                    "config_set_name": config_set_name,
                    "repetition": i_rep,
                    "status": "success",
                    "json_log_path": found_json_log,
                    "results_directory": pipeline_result_dir
                }
            else:
                 return {
                    "config_set_name": config_set_name,
                    "repetition": i_rep,
                    "status": "error",
                    "message": "No JSON log found",
                    "results_directory": pipeline_result_dir
                }
        else:
             return {
                "config_set_name": config_set_name,
                "repetition": i_rep,
                "status": "error",
                "message": "Result directory not valid"
            }
            
    except Exception as e_run:
        print(f"ERROR during execution of '{config_set_name}', Rep {i_rep}: {e_run}")
        import traceback
        error_log_path = os.path.splitext(temp_config_path)[0] + "_error.txt"
        with open(error_log_path, "w") as f_err:
            f_err.write(f"Error in config: {config_set_name}, Rep: {i_rep}\n")
            f_err.write(f"Config path: {temp_config_path}\n")
            f_err.write(f"Target dir: {unique_run_output_dir}\n")
            f_err.write(str(e_run) + "\n")
            f_err.write(traceback.format_exc())
            
        return {
            "config_set_name": config_set_name,
            "repetition": i_rep,
            "status": "error",
            "message": str(e_run)
        }

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Load Base Config
    if not os.path.exists(BASE_CONFIG_PATH):
        print(f"ERROR: Base config '{BASE_CONFIG_PATH}' not found.")
        exit(1)
        
    with open(BASE_CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)

    # Load Experiments Config
    if not os.path.exists(EXPERIMENTS_CONFIG_PATH):
        print(f"ERROR: Experiments config '{EXPERIMENTS_CONFIG_PATH}' not found.")
        exit(1)
        
    with open(EXPERIMENTS_CONFIG_PATH, 'r') as f:
        experiments_data = yaml.safe_load(f)
        parameter_grid = experiments_data.get("experiments", [])

    if not parameter_grid:
        print("No experiments found in configuration.")
        exit(0)

    # Setup Meta Run Directory
    meta_run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_run_parent_dir = base_config.get("run_settings", {}).get("results_base_dir", "results/qat_runs")
    main_output_dir_for_meta_run = os.path.join(meta_run_parent_dir, f"meta_run_{meta_run_timestamp}_Refactored")
    os.makedirs(main_output_dir_for_meta_run, exist_ok=True)
    
    temp_configs_storage_dir = os.path.join(main_output_dir_for_meta_run, "run_configs")
    os.makedirs(temp_configs_storage_dir, exist_ok=True)

    print(f"Results will be saved to: {main_output_dir_for_meta_run}")

    # Prepare Tasks
    tasks_to_run = []
    for i_config_set, param_variation_overrides in enumerate(parameter_grid):
        config_set_name = param_variation_overrides.get('name_suffix', f"configset_{i_config_set+1}")
        
        for i_rep in range(1, NUM_REPETITIONS_PER_CONFIG + 1):
            current_run_config = copy.deepcopy(base_config)
            
            # Apply overrides from experiments.yaml
            current_overrides = {
                "run_settings": param_variation_overrides.get("run_settings", {}),
                "data_params": param_variation_overrides.get("data_params", {}),
                "model_params": param_variation_overrides.get("model_params", {}),
                "training_params": param_variation_overrides.get("training_params", {}),
                "evaluation_params": param_variation_overrides.get("evaluation_params", {})
            }
            current_run_config = deep_update(current_run_config, current_overrides)
            
            # Meta run info
            current_run_config.setdefault("meta_run_info", {})
            current_run_config["meta_run_info"]["config_set_name"] = config_set_name
            current_run_config["meta_run_info"]["repetition_number"] = i_rep
            
            # Output directory
            unique_run_output_dir = os.path.join(main_output_dir_for_meta_run, config_set_name, f"rep_{i_rep}")
            os.makedirs(unique_run_output_dir, exist_ok=True)
            current_run_config.setdefault("run_settings", {})["results_base_dir"] = unique_run_output_dir

            # Save temp config
            temp_config_filename = f"config_{config_set_name}_rep{i_rep}.yaml"
            temp_config_path = os.path.join(temp_configs_storage_dir, temp_config_filename)
            with open(temp_config_path, 'w') as f_temp_cfg:
                yaml.dump(current_run_config, f_temp_cfg, sort_keys=False, indent=2)
            
            tasks_to_run.append((config_set_name, i_rep, temp_config_path, NUM_REPETITIONS_PER_CONFIG, unique_run_output_dir))

    print(f"Prepared {len(tasks_to_run)} tasks.")

    # Execute
    if tasks_to_run:
        print(f"Starting execution with {MAX_PARALLEL_PROCESSES} processes...")
        with mp.Pool(processes=MAX_PARALLEL_PROCESSES) as pool:
            results = pool.map(run_single_experiment_wrapper, tasks_to_run)
        
        # Aggregate Results (Simplified)
        all_runs_summary_data = []
        for result in results:
            if result and result.get("status") == "success":
                try:
                    with open(result["json_log_path"], 'r') as f_res:
                        run_log_data = json.load(f_res)
                    
                    run_config_used = run_log_data.get("input_configuration_from_yaml", {})
                    
                    run_summary = {
                        "config_set_name": result["config_set_name"],
                        "repetition": result["repetition"],
                        "dataset": get_nested_val(run_config_used, ["run_settings", "dataset_name"]),
                        "bits": get_nested_val(run_config_used, ["model_params", "quantization_bits"]),
                        "best_val_f1": get_nested_val(run_log_data,["best_model_metrics_achieved_val", "best_f1_weighted_val"]),
                    }
                    all_runs_summary_data.append(run_summary)
                except Exception as e:
                    print(f"Error processing result for {result.get('config_set_name')}: {e}")

        # Save Summary
        if all_runs_summary_data:
            summary_df = pd.DataFrame(all_runs_summary_data)
            summary_csv_path = os.path.join(main_output_dir_for_meta_run, f"summary_{meta_run_timestamp}.csv")
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Summary saved to {summary_csv_path}")