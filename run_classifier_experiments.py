"""
Run classifier hyperparameter experiments from classifier_configs folder.

Uses variant 1 (encoder pretrained + frozen) to tune classifier hyperparameters.
Once best hyperparameters are found, run train_classifier.py with other variants.

Usage:
    python run_classifier_experiments.py --encoder_path results/autoencoder/best/encoder.pth --encoder_config results/autoencoder/best/encoder_config.json
    python run_classifier_experiments.py --configs_dir classifier_configs/exp1_channels --dry_run
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_configs(configs_dir):
    """Find all JSON config files in the classifier_configs directory."""
    configs_path = Path(configs_dir)
    if not configs_path.exists():
        print(f"Error: configs directory '{configs_dir}' does not exist")
        sys.exit(1)
    
    configs = sorted(configs_path.rglob("*.json"))
    return configs


def get_output_path(config_path, configs_dir, results_dir):
    """Generate output path based on config path structure."""
    # e.g., classifier_configs/exp1_channels/baseline_32_64.json
    #    -> results/classifier/exp1_channels/baseline_32_64
    relative = config_path.relative_to(configs_dir)
    output_path = Path(results_dir) / relative.with_suffix('')
    return output_path


def run_experiment(config_path, output_path, encoder_path, encoder_config, dry_run=False):
    """Run a single experiment with variant 1 (pretrained + frozen encoder)."""
    cmd = [
        sys.executable,
        "train_classifier.py",
        "--config", str(config_path),
        "--out_path", str(output_path),
        "--variant", "1",
        "--encoder_path", str(encoder_path),
        "--encoder_config", str(encoder_config)
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
    parser = argparse.ArgumentParser(description="Run classifier hyperparameter experiments (variant 1)")
    parser.add_argument("--configs_dir", type=str, default="classifier_configs", 
                        help="Directory containing config files (default: classifier_configs)")
    parser.add_argument("--results_dir", type=str, default="results/classifier",
                        help="Directory to store results (default: results/classifier)")
    parser.add_argument("--encoder_path", type=str, required=True,
                        help="Path to pretrained encoder weights (.pth)")
    parser.add_argument("--encoder_config", type=str, required=True,
                        help="Path to encoder config (.json)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip experiments that already have results")
    args = parser.parse_args()
    
    # Validate encoder paths
    if not Path(args.encoder_config).exists():
        print(f"Error: encoder config not found: {args.encoder_config}")
        sys.exit(1)
    
    if not Path(args.encoder_path).exists():
        print(f"Error: encoder weights not found: {args.encoder_path}")
        sys.exit(1)
    
    # Find all configs
    configs = find_configs(args.configs_dir)
    
    if not configs:
        print(f"No config files found in '{args.configs_dir}'")
        sys.exit(1)
    
    print(f"Found {len(configs)} config files")
    print(f"Using variant 1: pretrained + frozen encoder")
    print(f"\nConfigs:")
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
        
        success = run_experiment(
            config_path, output_path,
            args.encoder_path, args.encoder_config, args.dry_run
        )
        
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
