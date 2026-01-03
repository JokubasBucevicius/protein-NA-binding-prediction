#!/bin/bash

# Change directory to the script's location
cd "$(dirname $0)"

# Path to the file containing PDB IDs
INPUT_LIST="./lists/cas9.txt"

# Path where the PDB files are already downloaded
PDB_FILES_DIR="../strukturos/cas9/"

# Directory to store processed receptor data
OUTPUT_DIR="./graphs/"

# Read the list and process each line
cat "$INPUT_LIST" | while read -r PDBID; do
    # Construct the path to the PDB file
    COMPLEXFILE="${PDB_FILES_DIR}/${PDBID}.pdb"

    # Check if the PDB file exists
    if [ -f "$COMPLEXFILE" ]; then
        echo "Processing $COMPLEXFILE"

        # Process the PDB file to extract and describe the receptor protein
        ./tools/extract-and-describe-receptor-protein \
            --input-complex "$COMPLEXFILE" \
            --output-dir "$OUTPUT_DIR" \
            --no-faspr \
            --all-chains-no-bsite \
		
    else
        echo "PDB file not found: $COMPLEXFILE"
    fi
done
