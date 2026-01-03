# =============================================================================
# Ground Truth Binding Site Visualization Script for PyMOL
# =============================================================================
#
# This script loads ground truth PDB files and colors residues by binding site.
# Non-binding residues (0) stay white; binding residues (1) are colored red.
#
# Usage:
#   pymol visualize_ground_truth.pml
#
# =============================================================================

python
from pymol import cmd

print("=" * 60)
print("Ground Truth Binding Site Visualization")
print("=" * 60)

# Clear any existing objects
cmd.delete("all")

# Define protein files to load
dna_proteins = [
    "dna-proteins/1b72_A_ground_truth.pdb",
    "dna-proteins/1bl0_A_ground_truth.pdb",
    "dna-proteins/1c8c_A_ground_truth.pdb",
    "dna-proteins/1le8_A_ground_truth.pdb",
    "dna-proteins/1llm_D_ground_truth.pdb",
]

rna_proteins = [
    "rna-proteins/1qtq_A_ground_truth.pdb",
    "rna-proteins/2db3_A_ground_truth.pdb",
    "rna-proteins/2i82_A_ground_truth.pdb",
    "rna-proteins/2jlv_A_ground_truth.pdb",
    "rna-proteins/3bso_B_ground_truth.pdb",
]

# Load all ground truth PDB files
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

    # Base color: white for all residues (non-binding)
    cmd.color("white", obj)

    # Color binding residues (b-factor = 1) in red
    # B-factor contains the binding site label (0 or 1)
    selection = f"{obj} and b > 0.5"

    # Check if there are any binding residues
    count = cmd.count_atoms(selection)
    if count > 0:
        cmd.color("red", selection)
        print(f"  {obj}: {count} binding atoms")
    else:
        print(f"  {obj}: No binding atoms")

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
print("Color scheme: White (non-binding) | Red (binding)")
print("=" * 60)

python end
