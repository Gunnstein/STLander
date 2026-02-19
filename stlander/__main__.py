try:
    from .cli import main
except Exception:
    # When executed as a top-level script (for example by PyInstaller),
    # relative imports may fail with "no known parent package".
    # Fall back to absolute import which works both when installed
    # as a package and when bundled into a single-file executable.
    from stlander.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
