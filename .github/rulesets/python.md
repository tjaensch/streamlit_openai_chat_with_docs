# Python Ruleset — Python-Specific Best Practices

Activate by adding `python` to the `rulesets` list in `.csi.yml`.

## Rules

### PY-001: Type hints on public function signatures
All public functions and methods should have type annotations for parameters and return values.

### PY-002: Use `pyproject.toml` or `requirements.txt` with pinned versions
Dependencies should be pinned to specific versions (e.g., `requests==2.31.0`) rather than unpinned (`requests`). Prefer `pyproject.toml` for modern projects.

### PY-003: No bare `except:` clauses
Bare `except:` or `except Exception:` blocks that silently swallow errors should be narrowed to specific exception types. At minimum, log the error.

### PY-004: `__init__.py` files should not be empty placeholders
If `__init__.py` exists, it should either export the package's public API or be documented as intentionally empty. Remove unused `__init__.py` files in non-package directories.

### PY-005: Use `pathlib.Path` over `os.path` for new code
New path manipulation code should prefer `pathlib.Path` over `os.path.join` and friends.

### PY-006: No `print()` for logging in libraries/services
Applications and libraries should use the `logging` module instead of `print()` for diagnostic output. `print()` is acceptable in CLI scripts and `__main__` blocks.

### PY-007: Test files should follow naming conventions
Test files should be named `test_*.py` or `*_test.py`. Test functions should start with `test_`.

### PY-008: No `import *` in production code
Wildcard imports (`from module import *`) should not be used in production code — they pollute the namespace and hide dependencies.

### PY-009: Virtual environment should be gitignored
Directories named `.venv/`, `venv/`, `env/` should appear in `.gitignore`.

### PY-010: Python version should be specified
A `.python-version` file or `python_requires` in `pyproject.toml` should declare the minimum supported Python version.
