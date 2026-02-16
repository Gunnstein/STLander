"""STLander — Principal Axis Alignment.

Load an STL, translate its (surface) center of mass to the origin, rotate so the
principal axes align with global XYZ, and visualize original vs aligned side-by-side.
"""

from .core import load_mesh, align_to_principal_axes, compute_surface_com_and_axes

__all__ = [
    "load_mesh",
    "align_to_principal_axes",
    "compute_surface_com_and_axes",
]

__version__ = "1.0.1"
