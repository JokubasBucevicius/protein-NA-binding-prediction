# =============================================================================
# Multiclass Binding Site Visualization Script for PyMOL
# =============================================================================
#
# This script visualizes multiclass binding predictions AND ground truth where:
# - B-factor contains DNA binding probability/label
# - Occupancy contains RNA binding probability/label
#
# Color scheme:
# - DNA only (>threshold): White -> Red gradient
# - RNA only (>threshold): White -> Blue gradient
# - Both DNA+RNA (>threshold): White -> Purple gradient
# - Neither: White
#
# Usage:
#   pymol visualize_multiclass.pml
#
# =============================================================================

python
from pymol import cmd, stored

# ----- CONFIGURATION -----
DNA_THRESHOLD = 0.5
RNA_THRESHOLD = 0.001
DNA_MAX = 1.0      # Max DNA probability for gradient
RNA_MAX = 0.005    # Max RNA probability for gradient (adjust based on predictions)
# -------------------------

print("=" * 60)
print("Multiclass Binding Site Visualization")
print(f"DNA Threshold: {DNA_THRESHOLD}")
print(f"RNA Threshold: {RNA_THRESHOLD}")
print("=" * 60)

# Clear any existing objects
cmd.delete("all")

# Define protein files to load
proteins = [
    ("multiclass/9lgi_A_multiclass.pdb", "prediction"),
    ("multiclass/9lgi_A_ground_truth.pdb", "ground_truth"),
]

# Load PDB files
print("\nLoading proteins...")
for pdb_file, label in proteins:
    try:
        # Extract base name for object naming
        obj_name = pdb_file.split("/")[-1].replace(".pdb", "")
        cmd.load(pdb_file, obj_name)
        print(f"  Loaded: {pdb_file} as {obj_name}")
    except Exception as e:
        print(f"  Warning: Could not load {pdb_file}: {e}")

print("\nApplying multiclass coloring...")

def interpolate_color(base_color, prob, threshold, max_val):
    """Interpolate from white to base_color based on probability."""
    if prob <= threshold:
        return [1.0, 1.0, 1.0]  # White

    # Scale probability: threshold->max_val maps to 0.0->1.0
    scaled = (prob - threshold) / (max_val - threshold)
    scaled = min(1.0, max(0.0, scaled))

    # Interpolate from white [1,1,1] to base_color
    r = 1.0 - scaled * (1.0 - base_color[0])
    g = 1.0 - scaled * (1.0 - base_color[1])
    b = 1.0 - scaled * (1.0 - base_color[2])

    return [r, g, b]


def get_multiclass_color(dna_prob, rna_prob):
    """Determine color based on DNA and RNA probabilities."""
    dna_above = dna_prob > DNA_THRESHOLD
    rna_above = rna_prob > RNA_THRESHOLD

    if dna_above and rna_above:
        # Both: Purple gradient based on scaled probabilities
        # Use the higher relative intensity
        dna_scaled = (dna_prob - DNA_THRESHOLD) / (DNA_MAX - DNA_THRESHOLD)
        rna_scaled = (rna_prob - RNA_THRESHOLD) / (RNA_MAX - RNA_THRESHOLD)
        max_scaled = max(dna_scaled, rna_scaled)
        max_scaled = min(1.0, max(0.0, max_scaled))
        # Interpolate to purple
        r = 1.0 - max_scaled * (1.0 - 0.6)
        g = 1.0 - max_scaled * 1.0
        b = 1.0 - max_scaled * (1.0 - 0.8)
        return [r, g, b]
    elif dna_above:
        # DNA only: Red gradient
        return interpolate_color([1.0, 0.0, 0.0], dna_prob, DNA_THRESHOLD, DNA_MAX)
    elif rna_above:
        # RNA only: Blue gradient
        return interpolate_color([0.0, 0.4, 1.0], rna_prob, RNA_THRESHOLD, RNA_MAX)
    else:
        # Neither: White
        return [1.0, 1.0, 1.0]


# Process each loaded object
for obj in cmd.get_object_list():
    print(f"\n  Processing: {obj}")

    # Remove nucleic acid atoms
    cmd.remove(f"{obj} and (resn DA+DT+DC+DG+DU+A+T+C+G+U+RA+RU+RC+RG or nucleic)")

    # Show as cartoon
    cmd.show("cartoon", obj)
    cmd.hide("lines", obj)
    cmd.hide("sticks", obj)

    # Base color: white
    cmd.color("white", obj)

    # Get unique residues
    stored.residues = []
    cmd.iterate(f"{obj} and name CA", "stored.residues.append((resi, b, q))")

    # Track statistics
    dna_only_count = 0
    rna_only_count = 0
    both_count = 0
    neither_count = 0

    # Color each residue based on its probabilities
    color_index = 0
    for resi, b_factor, occupancy in stored.residues:
        dna_prob = b_factor  # B-factor = DNA probability
        rna_prob = occupancy  # Occupancy = RNA probability

        # Get color for this residue
        rgb = get_multiclass_color(dna_prob, rna_prob)

        # Create unique color name and set it
        color_name = f"mc_color_{obj}_{color_index}"
        cmd.set_color(color_name, rgb)

        # Color the residue
        cmd.color(color_name, f"{obj} and resi {resi}")

        # Update statistics
        dna_above = dna_prob > DNA_THRESHOLD
        rna_above = rna_prob > RNA_THRESHOLD
        if dna_above and rna_above:
            both_count += 1
        elif dna_above:
            dna_only_count += 1
        elif rna_above:
            rna_only_count += 1
        else:
            neither_count += 1

        color_index += 1

    print(f"    DNA only (red): {dna_only_count} residues")
    print(f"    RNA only (blue): {rna_only_count} residues")
    print(f"    Both DNA+RNA (purple): {both_count} residues")
    print(f"    Neither (white): {neither_count} residues")

# Arrange objects side by side
objects = cmd.get_object_list()
if len(objects) >= 2:
    # Move ground truth to the right
    for obj in objects:
        if "ground_truth" in obj:
            cmd.translate([80, 0, 0], obj)

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
print(f"DNA Threshold: {DNA_THRESHOLD} | RNA Threshold: {RNA_THRESHOLD}")
print("Color scheme:")
print("  White: No binding")
print("  Red gradient: DNA binding only")
print("  Blue gradient: RNA binding only")
print("  Purple gradient: Both DNA + RNA binding")
print("")
print("Layout: Prediction (left) | Ground Truth (right)")
print("=" * 60)

python end
