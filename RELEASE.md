# Release Process

## 1) Prepare Release Branch

- Ensure CI is green.
- Confirm tests pass locally:

```bash
python -m pytest -q
```

## 2) Update Version and Changelog

- Bump version in `pyproject.toml`.
- Add release notes in `CHANGELOG.md` with date and sections (`Added`, `Changed`, `Fixed`).

## 3) Validate Documentation

- Verify `README.md` commands still work.
- Ensure new features or breaking changes are documented.

## 4) Tag and Publish

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

- Create a GitHub Release from the tag.
- Paste release notes from `CHANGELOG.md`.

## 5) Post-release

- Optionally export and publish model artifacts to Hugging Face.
- Announce important changes and migration notes.
