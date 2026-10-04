# Version policy

Meldr uses semantic versioning.
The version has the form `MAJOR.MINOR.PATCH`.

- A patch release fixes a bug without a change to the public contract.
- A minor release adds a backward-compatible feature.
- A major release can change or remove a public contract.

Before version 1.0, a minor release can change a contract.
The changelog lists each such change.
The public contract is the export list in `meldr/__init__.py` and the documented config keys.

The version lives in two files:

- `pyproject.toml` holds the package version.
- `src/meldr/_version.py` holds `meldr.__version__`.

Keep these values equal.
Make a release from a clean tag in the form `vMAJOR.MINOR.PATCH`.
The publish workflow builds the wheel from that tag.
PyPI trusts the GitHub Actions identity.
The workflow has no PyPI token.

Meldr supports Python 3.10 and newer supported Python releases.
Remove a Python version in the first Meldr release made six months after that version reaches end of life.
The changelog records the removal and the replacement versions.

For a release:

1. Update the two version values.
2. Add a changelog entry.
3. Run the format, lint, type, STE, and test gates.
4. Push the version tag.
5. Check the package on PyPI.
