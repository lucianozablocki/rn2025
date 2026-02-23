"""
Run all 4 training variants with the best classifier config.

Variants:
1) Encoder PRETRAINED + FROZEN
2) Encoder NOT PRETRAINED + NOT FROZEN (train from scratch)
3) Encoder PRETRAINED + NOT FROZEN (fine-tune)
4) Encoder NOT PRETRAINED + FROZEN

Usage:
    python run_variant_comparison.py \
        --config classifier_configs/best.json \
        --encoder_path results/autoencoder/best/encoder.pth \
        --encoder_config results/autoencoder/best/encoder_config.json
"""

import argparse
import subprocess
import sys
from pathlib import Path


VARIANT_NAMES = {
    1: "v1_pretrained_frozen",
    2: "v2_not_pretrained_not_frozen",
    3: "v3_pretrained_not_frozen",
    4: "v4_not_pretrained_frozen"
}


def run_variant(config_path, output_path, variant, encoder_path, encoder_config, dry_run=False):
    """Run a single variant."""
    cmd = [
        sys.executable,
        "train_classifier.py",
        "--config", str(config_path),
        "--out_path", str(output_path),
        "--variant", str(variant),
        "--encoder_config", str(encoder_config)
    ]
    
    # Variants 1 and 3 need pretrained encoder weights
    if variant in [1, 3]:
        cmd.extend(["--encoder_path", str(encoder_path)])
    
    print(f"\n{'=' * 60}")
    print(f"VARIANT {variant}: {VARIANT_NAMES[variant]}")
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
        print(f"Error running variant {variant}: {e}")
        return False
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run all 4 training variants")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to classifier config JSON")
    parser.add_argument("--results_dir", type=str, default="results/classifier/variant_comparison",
                        help="Directory to store results")
    parser.add_argument("--encoder_path", type=str, required=True,
                        help="Path to pretrained encoder weights (.pth)")
    parser.add_argument("--encoder_config", type=str, required=True,
                        help="Path to encoder config (.json)")
    parser.add_argument("--variants", type=int, nargs="+", default=[1, 2, 3, 4],
                        choices=[1, 2, 3, 4],
                        help="Which variants to run (default: all)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip variants that already have results")
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.config).exists():
        print(f"Error: config not found: {args.config}")
        sys.exit(1)
    
    if not Path(args.encoder_config).exists():
        print(f"Error: encoder config not found: {args.encoder_config}")
        sys.exit(1)
    
    if (1 in args.variants or 3 in args.variants) and not Path(args.encoder_path).exists():
        print(f"Error: encoder weights not found: {args.encoder_path}")
        sys.exit(1)
    
    print(f"Config: {args.config}")
    print(f"Running variants: {args.variants}")
    print(f"Results dir: {args.results_dir}")
    
    # Run variants
    successful = 0
    failed = 0
    skipped = 0
    
    for variant in args.variants:
        output_path = Path(args.results_dir) / VARIANT_NAMES[variant]
        
        # Check if already exists
        if args.skip_existing and (output_path / "test_results.json").exists():
            print(f"\nSkipping variant {variant} (results exist): {output_path}")
            skipped += 1
            continue
        
        success = run_variant(
            args.config, output_path, variant,
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
    print(f"Variants run: {len(args.variants)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
