# Generic Ruleset — Language-Agnostic Software Engineering Best Practices

This ruleset is **always active** and applies to any repository regardless of language or framework.

## Rules

### G-001: README must exist and be non-empty
Every repository must have a `README.md` at the root with at minimum: project name, description, and basic usage instructions.

### G-002: LICENSE file should exist
Open-source and internal projects should have a LICENSE file at the root.

### G-003: .gitignore should cover common patterns
The `.gitignore` must include patterns for: OS files (`.DS_Store`, `Thumbs.db`), IDE files (`.idea/`, `.vscode/`), and language-specific build artifacts.

### G-004: No committed secrets or credentials
Source files, config files, and YAML must not contain hardcoded API keys, tokens, passwords, or private keys. Use environment variables or secret managers.

### G-005: CI workflows should exist
Projects with source code should have at least one CI workflow under `.github/workflows/` that runs on pull requests.

### G-006: Pinned dependency versions
Dependency manifests should pin versions (exact or range) rather than using unpinned `*` or `latest`.

### G-007: No binary files in source tree
Binary files (`.exe`, `.dll`, `.so`, `.jar`, compiled assets) should not be committed to the repo unless necessary (e.g., test fixtures with documentation).

### G-008: Consistent line endings
The repository should have a `.gitattributes` file or consistent line ending usage. Mixed `CRLF`/`LF` causes spurious diffs.

### G-009: Meaningful commit-able state
The default branch should not contain `WIP`, `temp`, or `test123` files that indicate incomplete work.

### G-010: GitHub Actions should pin major versions
Workflow `uses:` references should pin to a major version tag (e.g., `actions/checkout@v4`) rather than `@main` or `@master`.
