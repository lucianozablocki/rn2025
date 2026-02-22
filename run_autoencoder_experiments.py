"""
Run all autoencoder experiments from autoencoder_configs folder.

Usage:
    python run_autoencoder_experiments.py
    python run_autoencoder_experiments.py --configs_dir autoencoder_configs/exp3_channels  # run specific experiment set
    python run_autoencoder_experiments.py --dry_run  # just print what would be run
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_configs(configs_dir):
    """Find all JSON config files in the autoencoder_configs directory."""
    configs_path = Path(configs_dir)
    if not configs_path.exists():
        print(f"Error: configs directory '{configs_dir}' does not exist")
        sys.exit(1)
    
    configs = sorted(configs_path.rglob("*.json"))
    return configs


def get_output_path(config_path, configs_dir, results_dir):
    """Generate output path based on config path structure."""
    # e.g., configs/exp1_extra_layer/no_extra_32ch.json -> results/exp1_extra_layer/no_extra_32ch
    relative = config_path.relative_to(configs_dir)
    output_path = Path(results_dir) / relative.with_suffix('')
    return output_path


def run_experiment(config_path, output_path, dry_run=False):
    """Run a single experiment."""
    cmd = [
        sys.executable,
        "train_autoencoder.py",
        "--config", str(config_path),
        "--out_path", str(output_path)
    ]
    
    print(f"\n{'=' * 60}")
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print(f"Command: {' '.join(cmd)}")
    print('=' * 60)
    
    if dry_run:
        print("[DRY RUN] Skipping execution")
        return True
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return False
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run all autoencoder experiments")
    parser.add_argument("--configs_dir", type=str, default="autoencoder_configs", 
                        help="Directory containing config files (default: autoencoder_configs)")
    parser.add_argument("--results_dir", type=str, default="results/autoencoder",
                        help="Directory to store results (default: results/autoencoder)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip experiments that already have results")
    args = parser.parse_args()
    
    # Find all configs
    configs = find_configs(args.configs_dir)
    
    if not configs:
        print(f"No config files found in '{args.configs_dir}'")
        sys.exit(1)
    
    print(f"Found {len(configs)} config files:")
    for config in configs:
        print(f"  - {config}")
    
    # Run experiments
    successful = 0
    failed = 0
    skipped = 0
    
    for i, config_path in enumerate(configs, 1):
        output_path = get_output_path(config_path, args.configs_dir, args.results_dir)
        
        print(f"\n[{i}/{len(configs)}] Running experiment...")
        
        # Check if already exists
        if args.skip_existing and (output_path / "test_results.json").exists():
            print(f"Skipping (results exist): {output_path}")
            skipped += 1
            continue
        
        success = run_experiment(config_path, output_path, args.dry_run)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {len(configs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
