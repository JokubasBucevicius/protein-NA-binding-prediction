# =============================================================================
# Binding Site Visualization Script for PyMOL
# =============================================================================
#
# This script loads predicted PDB files and colors residues by binding probability.
# Residues below the threshold stay white; residues above threshold show a
# white-to-red gradient based on their binding probability.
#
# Usage:
#   pymol visualize_binding.pml
#
# To adjust the threshold:
#   Edit the THRESHOLD variable below, then run:
#   @visualize_binding.pml
#
# =============================================================================

# ===== CONFIGURATION =====
# Residues with probability below this value stay white
# Residues above this value are colored on a white->red gradient

python
from pymol import cmd

# ----- ADJUST THIS VALUE -----
THRESHOLD = 0.7
# -----------------------------

print("=" * 60)
print("Binding Site Visualization")
print(f"Threshold: {THRESHOLD}")
print("=" * 60)

# Clear any existing objects
cmd.delete("all")

# Define protein files to load
dna_proteins = [
    "dna-proteins/1b72_A_predicted.pdb",
    "dna-proteins/1bl0_A_predicted.pdb",
    "dna-proteins/1c8c_A_predicted.pdb",
    "dna-proteins/1le8_A_predicted.pdb",
    "dna-proteins/1llm_D_predicted.pdb",
]

rna_proteins = [
    "rna-proteins/1qtq_A_predicted.pdb",
    "rna-proteins/1qtq_A_ground_truth.pdb",
]

# Load all predicted PDB files
print("\nLoading DNA-binding proteins...")
for pdb_file in dna_proteins:
    try:
        cmd.load(pdb_file)
        print(f"  Loaded: {pdb_file}")
    except Exception as e:
        print(f"  Warning: Could not load {pdb_file}: {e}")

print("\nLoading RNA-binding proteins...")
for pdb_file in rna_proteins:
    try:
        cmd.load(pdb_file)
        print(f"  Loaded: {pdb_file}")
    except Exception as e:
        print(f"  Warning: Could not load {pdb_file}: {e}")

# Apply visualization settings to all loaded objects
print("\nApplying visualization settings...")

for obj in cmd.get_object_list():
    # Remove nucleic acid atoms from visualization
    cmd.remove(f"{obj} and (resn DA+DT+DC+DG+DU+A+T+C+G+U+RA+RU+RC+RG or nucleic)")

    # Show as cartoon representation
    cmd.show("cartoon", obj)
    cmd.hide("lines", obj)
    cmd.hide("sticks", obj)

    # Base color: white for all residues
    cmd.color("white", obj)

    # Apply white->red spectrum for residues above threshold
    # B-factor contains the binding probability (0-1)
    selection = f"{obj} and b > {THRESHOLD}"

    # Check if there are any residues above threshold
    count = cmd.count_atoms(selection)
    if count > 0:
        cmd.spectrum("b", "white_red", selection,
                     minimum=THRESHOLD, maximum=1.0)
        print(f"  {obj}: {count} atoms above threshold")
    else:
        print(f"  {obj}: No atoms above threshold")

# Global visualization settings
cmd.bg_color("black")
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_smooth_loops", 1)
cmd.set("cartoon_flat_sheets", 1)
cmd.set("ray_shadows", 0)
cmd.set("antialias", 2)

# Reset view
cmd.reset()
cmd.zoom("all", buffer=5)

print("\n" + "=" * 60)
print("Visualization complete!")
print(f"Threshold: {THRESHOLD}")
print("Color scheme: White (low prob) -> Red (high prob)")
print("=" * 60)

python end
