# STLander

A small Python package (SciPy + PyVista + **PySide6**) to:

1. Load an STL
2. Translate its **surface** center of mass (COM) to the origin
3. Rotate so **principal axes** align with global **XYZ**
4. Visualize **original vs aligned** side-by-side in a **single window**
5. Switch between **Perspective/Parallel**, **Surface/Wireframe/Points**, etc.
6. Save the aligned STL

## Install (editable)

```bash
python -m pip install -U pip
pip install -e .
```

## Run the GUI

```bash
stlander
# or:
stlander gui
```

## Run via CLI

```bash
stlander align input.stl output_aligned.stl
```

## Notes on COM / principal axes

STL files are surface meshes. This package computes an **area-weighted** COM and principal axes,
treating the surface as a thin shell with uniform areal density.

If you need true **volume** mass properties for watertight solids, extend `compute_surface_com_and_axes`
to use a volumetric inertia tensor (tetra-based). The current implementation is a robust default for
alignment/visualization.


## GUI extras

- Quick camera views along **±X / ±Y / ±Z**
- Option to map **PA2** to **Y** (default) or **Z** (swap Y/Z)
