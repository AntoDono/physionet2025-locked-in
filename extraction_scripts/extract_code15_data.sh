#!/bin/bash

# Script to extract CODE-15 data using prepare_code15_data.py
# Usage: ./extract_code15_data.sh <input_directory> <output_directory>

# Check if correct number of arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    echo "Example: $0 ~/PhysionetData/MoodyChallenge/code-15 ~/PhysionetData/code15_wfdb"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Check for required CSV files
EXAMS_CSV="$INPUT_DIR/exams.csv"
LABELS_CSV="$INPUT_DIR/code15_chagas_labels.csv"

if [ ! -f "$EXAMS_CSV" ]; then
    echo "Error: exams.csv not found in $INPUT_DIR"
    exit 1
fi

if [ ! -f "$LABELS_CSV" ]; then
    echo "Error: code15_chagas_labels.csv not found in $INPUT_DIR"
    exit 1
fi

# Find all HDF5 files
echo "Finding HDF5 files in $INPUT_DIR..."
HDF5_FILES=($(find "$INPUT_DIR" -name "exams_part*.hdf5" -type f | sort))

if [ ${#HDF5_FILES[@]} -eq 0 ]; then
    echo "Error: No HDF5 files found in $INPUT_DIR"
    exit 1
fi

echo "Found ${#HDF5_FILES[@]} HDF5 files:"
for file in "${HDF5_FILES[@]}"; do
    echo "  - $(basename $file)"
done

# Change to the script directory (assuming prepare_code15_data.py is in the parent directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Process each HDF5 file individually
echo ""
echo "Starting extraction process..."
echo ""

PROCESSED=0
FAILED=0

for i in "${!HDF5_FILES[@]}"; do
    hdf5_file="${HDF5_FILES[$i]}"
    
    echo "[$((i+1))/${#HDF5_FILES[@]}] Processing $(basename $hdf5_file)..."
    
    # Run the Python script for this single file
    python prepare_code15_data.py \
        -i "$hdf5_file" \
        -d "$EXAMS_CSV" \
        -l "$LABELS_CSV" \
        -o "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully processed $(basename $hdf5_file)"
        PROCESSED=$((PROCESSED + 1))
    else
        echo "  ✗ Failed to process $(basename $hdf5_file)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

# Summary
echo "========================================"
echo "Extraction Summary:"
echo "  Total files: ${#HDF5_FILES[@]}"
echo "  Successfully processed: $PROCESSED"
echo "  Failed: $FAILED"
echo "========================================"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "All files extracted successfully!"
    echo "Output files are in: $OUTPUT_DIR"
    exit 0
else
    echo ""
    echo "Warning: Some files failed to process"
    exit 1
fi 