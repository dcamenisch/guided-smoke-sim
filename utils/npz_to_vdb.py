"""Convert NPZ simulation state files to OpenVDB format for rendering."""

import sys
import argparse
import numpy as np
import pyopenvdb as vdb
from pathlib import Path


def convert_npz_to_vdb(npz_path, vdb_path, field="density", verbose=True):
    """Convert a single NPZ file to VDB format

    Args:
        npz_path: Path to input NPZ file
        vdb_path: Path to output VDB file
        field: Field to export (default: "density")
        verbose: Print progress messages

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load NPZ file
        if verbose:
            print(f"Loading {npz_path}...")
        data = np.load(npz_path)

        # Check if field exists
        if field not in data:
            print(f"Error: Field '{field}' not found in {npz_path}")
            print(f"Available fields: {list(data.keys())}")
            return False

        # Get density data and grid dimensions
        density = data[field]

        # Determine if 2D or 3D
        if len(density.shape) == 2:
            # 2D simulation - add a third dimension
            if verbose:
                print("2D simulation detected, extending to 3D...")
            density = density[:, :, np.newaxis]

        # Get grid spacing (dx)
        dx = float(data.get("dx", 1.0))

        if verbose:
            print(f"Grid dimensions: {density.shape}")
            print(f"Grid spacing (dx): {dx}")
            print(f"Density range: [{density.min():.6f}, {density.max():.6f}]")

        # Create OpenVDB grid and copy data directly from array
        if verbose:
            print("Converting to VDB format...")

        grid = vdb.FloatGrid()
        grid.copyFromArray(density)
        grid.name = field

        # Write VDB file
        if verbose:
            print(f"Writing to {vdb_path}...")
        vdb.write(str(vdb_path), grids=[grid])

        if verbose:
            print(f"✓ Successfully converted {npz_path} to {vdb_path}")

        return True

    except Exception as e:
        print(f"Error converting {npz_path}: {e}")
        import traceback

        traceback.print_exc()
        return False


def convert_directory(
    input_dir, output_dir, field="density", pattern="*.npz", verbose=True
):
    """Convert all NPZ files in a directory to VDB format

    Args:
        input_dir: Input directory containing NPZ files
        output_dir: Output directory for VDB files
        field: Field to export (default: "density")
        pattern: File pattern to match (default: "*.npz")
        verbose: Print progress messages

    Returns:
        Number of files successfully converted
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all NPZ files
    npz_files = sorted(input_path.glob(pattern))

    if not npz_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return 0

    print(f"Found {len(npz_files)} NPZ files")
    print(f"Converting to VDB format...\n")

    success_count = 0
    for i, npz_file in enumerate(npz_files, 1):
        # Generate output filename
        vdb_file = output_path / (npz_file.stem + ".vdb")

        print(f"[{i}/{len(npz_files)}] {npz_file.name} -> {vdb_file.name}")

        if convert_npz_to_vdb(npz_file, vdb_file, field=field, verbose=False):
            success_count += 1
            print(f"  ✓ Success")
        else:
            print(f"  ✗ Failed")
        print()

    print(
        f"\nConversion complete: {success_count}/{len(npz_files)} files converted successfully"
    )
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert NPZ simulation state files to OpenVDB format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python npz_to_vdb.py state_0000.npz output.vdb
  python npz_to_vdb.py results/experiment-name
  python npz_to_vdb.py input/ output/ --field vorticity
        """,
    )

    parser.add_argument("input", type=str, help="Input NPZ file or directory")
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        default=None,
        help="Output VDB file or directory (optional: auto-generates for directories)",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="density",
        help="Field to export from NPZ file (default: density)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npz",
        help="File pattern for batch conversion (default: *.npz)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress verbose output"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    verbose = not args.quiet

    # Determine output path
    if args.output is not None:
        output_path = Path(args.output)
    else:
        # Auto-generate output path
        if input_path.is_file():
            print("Error: Output path is required for single file conversion")
            sys.exit(1)
        elif input_path.is_dir():
            # Add -vdb suffix to directory name
            output_path = input_path.parent / (input_path.name + "-vdb")
            if verbose:
                print(f"Auto-generated output directory: {output_path}")
        else:
            print(f"Error: Input path '{input_path}' does not exist")
            sys.exit(1)

    # Check if input is a file or directory
    if input_path.is_file():
        # Single file conversion
        if output_path.is_dir():
            # If output is directory, use same filename
            output_file = output_path / (input_path.stem + ".vdb")
        else:
            output_file = output_path

        success = convert_npz_to_vdb(
            input_path, output_file, field=args.field, verbose=verbose
        )

        sys.exit(0 if success else 1)

    elif input_path.is_dir():
        # Directory conversion
        success_count = convert_directory(
            input_path,
            output_path,
            field=args.field,
            pattern=args.pattern,
            verbose=verbose,
        )

        sys.exit(0 if success_count > 0 else 1)

    else:
        print(f"Error: Input path '{input_path}' does not exist")
        sys.exit(1)


if __name__ == "__main__":
    main()
