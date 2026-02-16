from __future__ import annotations

import argparse

from .core import load_mesh, align_to_principal_axes, save_mesh
from .gui_qt import main as gui_main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stlander",
        description="Align an STL to its principal axes (COM to origin, axes to XYZ).",
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    sub.add_parser("gui", help="Launch the PySide GUI.")

    c = sub.add_parser("align", help="Align an STL and write the result.")
    c.add_argument("input", help="Input STL path")
    c.add_argument("output", help="Output STL path")
    c.add_argument("--pa2-target", choices=["Y", "Z"], default="Y", help="Where to map the 2nd principal axis: Y (default) or Z (swap Y/Z).")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd in (None, "gui"):
        gui_main()
        return 0

    if args.cmd == "align":
        mesh = load_mesh(args.input)
        res = align_to_principal_axes(mesh, pa2_target=getattr(args, 'pa2_target', 'Y'))
        save_mesh(res.aligned, args.output)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
