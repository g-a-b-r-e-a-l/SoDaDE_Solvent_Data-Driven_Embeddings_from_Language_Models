#!/usr/bin/env python3
"""
Batch inference script to process multiple CSV files through the inference pipeline.
"""

import os
import glob
import subprocess
import argparse
from pathlib import Path


def run_batch_inference(
    input_folder: str,
    output_folder: str,
    model_path: str,
    inference_positions: list,
    batch_size: int = 16,
    num_samples: int = None,
    masking_probability: float = 0.0,
    dropout_rate: float = 0.3,
    csv_pattern: str = "*.csv"
):
    """
    Run inference on all CSV files in a folder.
    
    Args:
        input_folder: Path to folder containing CSV files
        output_folder: Path to folder where outputs will be saved
        model_path: Path to trained model
        inference_positions: List of positions to predict
        batch_size: Batch size for inference
        num_samples: Number of samples to process per file (None for all)
        masking_probability: Masking probability
        dropout_rate: Dropout rate used during training
        csv_pattern: Pattern to match CSV files (default: "*.csv")
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(input_folder, csv_pattern))
    
    if not csv_files:
        print(f"No CSV files found in {input_folder} matching pattern {csv_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Output folder: {output_folder}")
    print(f"Inference positions: {inference_positions}")
    
    successful_files = []
    failed_files = []
    
    for i, csv_file in enumerate(csv_files, 1):
        csv_filename = Path(csv_file).stem
        print(f"\n[{i}/{len(csv_files)}] Processing: {csv_filename}")
        
        # Create output path for this file
        output_file = os.path.join(output_folder, f"{csv_filename}_predictions.pt")
        
        # Build command
        cmd = [
            "python", "inference.py",
            "-mp", model_path,
            "-dp", csv_file,
            "-ip", str(inference_positions),
            "-bs", str(batch_size),
            "-mp_prob", str(masking_probability),
            "-dr", str(dropout_rate),
            "-op", output_file
        ]
        
        if num_samples is not None:
            cmd.extend(["-ns", str(num_samples)])
        
        try:
            # Run the inference command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Successfully processed {csv_filename}")
            successful_files.append(csv_filename)
            
            # Print any output from the inference script
            if result.stdout:
                print("Output:", result.stdout.strip())
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Error processing {csv_filename}")
            print(f"Error output: {e.stderr}")
            failed_files.append(csv_filename)
        except Exception as e:
            print(f"✗ Unexpected error processing {csv_filename}: {e}")
            failed_files.append(csv_filename)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Successful: {len(successful_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if successful_files:
        print(f"\nSuccessfully processed:")
        for file in successful_files:
            print(f"  ✓ {file}")
    
    if failed_files:
        print(f"\nFailed to process:")
        for file in failed_files:
            print(f"  ✗ {file}")
    
    # List output files
    print(f"\nOutput files saved to: {output_folder}")
    txt_files = glob.glob(os.path.join(output_folder, "*.txt"))
    pt_files = glob.glob(os.path.join(output_folder, "*.pt"))
    
    print(f"Generated {len(pt_files)} .pt files and {len(txt_files)} .txt files")


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference on multiple CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python batch_inference.py -if "./csv_files" -of "./predictions" -mp "model.pth" -ip "[2,5,8]"
    
    python batch_inference.py \\
        -if "./data/test_files" \\
        -of "./results/batch_predictions" \\
        -mp "trained_model.pth" \\
        -ip "[0,1,3,7]" \\
        -bs 32 \\
        -ns 1000
        """
    )
    
    # Required arguments
    parser.add_argument("-if", "--input_folder", type=str, required=True,
                       help="Folder containing CSV files to process")
    parser.add_argument("-of", "--output_folder", type=str, required=True,
                       help="Folder where predictions will be saved")
    parser.add_argument("-mp", "--model_path", type=str, required=True,
                       help="Path to trained model file (.pth)")
    parser.add_argument("-ip", "--inference_positions", type=str, required=True,
                       help="List of positions to predict (e.g., '[2,5,8]')")
    
    # Optional arguments
    parser.add_argument("-bs", "--batch_size", type=int, default=16,
                       help="Batch size for inference (default: 16)")
    parser.add_argument("-ns", "--num_samples", type=int, default=None,
                       help="Number of samples to process per file (default: all)")
    parser.add_argument("-mp_prob", "--masking_probability", type=float, default=0.0,
                       help="Masking probability (default: 0.0)")
    parser.add_argument("-dr", "--dropout_rate", type=float, default=0.3,
                       help="Dropout rate used during training (default: 0.3)")
    parser.add_argument("-cp", "--csv_pattern", type=str, default="*.csv",
                       help="Pattern to match CSV files (default: '*.csv')")
    
    args = parser.parse_args()
    
    # Parse inference positions
    try:
        inference_positions = eval(args.inference_positions)
        if not isinstance(inference_positions, list):
            raise ValueError("Inference positions must be a list")
        if not all(isinstance(pos, int) for pos in inference_positions):
            raise ValueError("All positions must be integers")
    except Exception as e:
        raise ValueError(f"Invalid format for inference positions: {e}")
    
    # Run batch inference
    run_batch_inference(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_path=args.model_path,
        inference_positions=inference_positions,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        masking_probability=args.masking_probability,
        dropout_rate=args.dropout_rate,
        csv_pattern=args.csv_pattern
    )


if __name__ == "__main__":
    main()