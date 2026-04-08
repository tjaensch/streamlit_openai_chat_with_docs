#!/usr/bin/env python3
"""
openai-scan.py — OpenAI-backed CSI scanner (scan-only, no file editing).

Reads the same agent prompt used by the Copilot CLI backend and sends it to
the OpenAI API. Outputs a structured CSI report in the same format.

NOTE: This backend is scan-only. It cannot apply fixes because it lacks the
Copilot CLI's tool-use capability for file editing. Use the Copilot backend
for full scan-and-fix functionality.

Usage:
    python openai-scan.py --prompt-file <path> --output <path> [--model <model>] [--timeout <seconds>]
    python openai-scan.py --output <path> --fallback-report <error_message>

    When --fallback-report is given, --prompt-file is not required. The script
    writes a CSI-contract-compliant failure report and exits with code 1.

Requires:
    - OPENAI_API_KEY environment variable (not needed with --fallback-report)
    - openai>=1.0.0 pip package (not needed with --fallback-report)
"""

import argparse
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path


# Source file extensions the scanner should read
SOURCE_EXTENSIONS = {
    ".go", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".kt", ".kts", ".cs", ".rs", ".rb",
    ".swift", ".php", ".c", ".cpp", ".h", ".hpp",
    ".sh", ".bash", ".ps1", ".nf", ".groovy",
    ".tf", ".hcl", ".yaml", ".yml", ".toml", ".json",
}

# Directories to always exclude
EXCLUDE_DIRS = {
    "node_modules", ".git", "vendor", "__pycache__",
    "dist", ".venv", "venv", ".next", "build", "target",
}


def get_repo_context(max_files: int = 100, max_source_chars: int = 80_000) -> str:
    """Gather repository context: file tree, config files, and source code."""
    lines = ["## Repository Structure\n"]

    # File tree (excluding common noise)
    try:
        result = subprocess.run(
            [
                "find", ".", "-maxdepth", "4", "-type", "f",
                "-not", "-path", "*/node_modules/*",
                "-not", "-path", "*/.git/*",
                "-not", "-path", "*/vendor/*",
                "-not", "-path", "*/__pycache__/*",
                "-not", "-path", "*/dist/*",
                "-not", "-path", "*/.venv/*",
            ],
            capture_output=True, text=True, timeout=30
        )
        files = sorted(result.stdout.strip().splitlines())[:max_files]
        lines.append("```")
        lines.extend(files)
        lines.append("```\n")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        lines.append("(Could not list files)\n")

    # Read key config/meta files
    key_files = [
        "README.md", "CONTRIBUTING.md", ".gitignore", ".csi.yml",
        "package.json", "requirements.txt", "pyproject.toml",
        "go.mod", "go.sum", "Cargo.toml",
        "Makefile", "Dockerfile", "docker-compose.yml",
    ]

    for fname in key_files:
        path = Path(fname)
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
                lines.append(f"### {fname}\n```\n{content}\n```\n")
            except OSError:
                pass

    # Read workflow files
    wf_dir = Path(".github/workflows")
    if wf_dir.is_dir():
        for wf in sorted(wf_dir.glob("*.yml"))[:10]:
            try:
                content = wf.read_text(encoding="utf-8", errors="replace")
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
                lines.append(f"### {wf}\n```yaml\n{content}\n```\n")
            except OSError:
                pass

    # Read source files so the model can analyse actual code
    lines.append("## Source Files\n")
    source_chars = 0
    source_files = []

    for root, dirs, filenames in os.walk("."):
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fname in sorted(filenames):
            if Path(fname).suffix.lower() in SOURCE_EXTENSIONS:
                source_files.append(os.path.join(root, fname))

    for fpath in sorted(source_files):
        if source_chars >= max_source_chars:
            lines.append(f"\n... (source budget exhausted at {max_source_chars} chars, {len(source_files)} files total)\n")
            break
        try:
            content = Path(fpath).read_text(encoding="utf-8", errors="replace")
            # Per-file cap at 5 KB to avoid one huge file eating the budget
            if len(content) > 5000:
                content = content[:5000] + "\n... (truncated)"
            source_chars += len(content)
            ext = Path(fpath).suffix.lstrip(".")
            lines.append(f"### {fpath}\n```{ext}\n{content}\n```\n")
        except OSError:
            pass

    return "\n".join(lines)


def build_scan_failure_report(error_message: str) -> str:
    """Return a CSI-formatted fallback report for backend failures."""
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    # Sanitize error: escape backticks and truncate to prevent markdown breakage
    sanitized = error_message.replace("`", "'")
    normalized_error = (" ".join(sanitized.split()) or "Unknown error")[:200]

    return textwrap.dedent(
        f"""\
        ## Scan Results

        **Issue ID**: CSI-CONFIG-001
        **Category**: CONFIG
        **Severity**: 🔴 HIGH
        **Description**: OpenAI scan backend could not complete.

        ### What Changed
        No files were modified. OpenAI runs in scan-only mode, and this invocation failed before it could produce findings.

        ### Evidence
        `.github/scripts/openai-scan.py` fallback report: `{normalized_error}`

        ### Verification
        Re-run the scan after resolving the backend error shown above.

        ---

        ## Remaining Issues

        1. **[CONFIG] 🔴 HIGH**: OpenAI scan backend could not complete — `.github/scripts/openai-scan.py:92-135`

        ---

        ## Scan Summary

        | Category | Issues Found |
        |----------|-------------|
        | DRY Violations | 0 |
        | Documentation Drift | 0 |
        | Tooling Currency | 0 |
        | Dead Code | 0 |
        | Code Quality | 0 |
        | Security Hygiene | 0 |
        | Dependency Health | 0 |
        | Config Consistency | 1 |
        | **Total** | **1** |

        *Scan completed: {timestamp}*
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-backed CSI scanner")
    parser.add_argument("--prompt-file", required=False, help="Path to the agent prompt file")
    parser.add_argument("--output", required=True, help="Path to write the report")
    parser.add_argument("--model", default="o3", help="OpenAI model to use")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout in seconds")
    parser.add_argument("--fallback-report", metavar="MSG", help="Write a CSI-formatted failure report with the given error message and exit")
    args = parser.parse_args()
    output_path = Path(args.output)

    def write_report_and_exit(report: str, exit_code: int = 0) -> None:
        """Write report to output path (creating parent dirs) and exit."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        line_count = len(report.splitlines())
        print(f"CSI report written: {output_path} ({line_count} lines)")
        sys.exit(exit_code)

    # Fallback report mode: write a CSI-formatted failure report and exit
    if args.fallback_report:
        write_report_and_exit(build_scan_failure_report(args.fallback_report), exit_code=1)

    if not args.prompt_file:
        print("::error::--prompt-file is required when not using --fallback-report", file=sys.stderr)
        write_report_and_exit(build_scan_failure_report("--prompt-file is required"), exit_code=1)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("::error::OPENAI_API_KEY environment variable is not set", file=sys.stderr)
        write_report_and_exit(build_scan_failure_report("OPENAI_API_KEY environment variable is not set"), exit_code=1)

    try:
        from openai import OpenAI
    except ImportError:
        print("::error::openai package not installed. Run: pip install openai>=1.0.0", file=sys.stderr)
        write_report_and_exit(build_scan_failure_report("openai package not installed. Run: pip install openai>=1.0.0"), exit_code=1)

    # Read the prompt
    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_file():
        print(f"::error::Prompt file not found: {args.prompt_file}", file=sys.stderr)
        write_report_and_exit(build_scan_failure_report(f"Prompt file not found: {args.prompt_file}"), exit_code=1)

    agent_prompt = prompt_path.read_text(encoding="utf-8")

    # Gather repo context (since OpenAI can't browse files)
    repo_context = get_repo_context()

    scan_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    system_message = textwrap.dedent(f"""\
        You are a repository maintenance scanner. You analyze codebases for
        maintenance issues across these categories: DRY violations, documentation
        drift, tooling currency, dead code, code quality, security hygiene,
        dependency health, and config consistency.

        You are running in SCAN-ONLY mode. You cannot edit files. Your job is to
        produce a structured report of your findings.

        IMPORTANT RULES:
        - Always include ALL three sections (Scan Results, Remaining Issues, Scan Summary).
        - If no issues are found, set all counts to 0 and write "✅ No maintenance issues detected. Repository is in good health." under Remaining Issues.
        - The first section MUST be titled "## Scan Results" (NOT "Applied Fix").
        - In Remaining Issues, each item MUST use square brackets around the category name, e.g. **[CODE_QUALITY]** not **Code Quality**.
        - Use the exact timestamp provided below — do NOT generate your own.
        - The *Scan completed:* line must always be the very last line of your output.

        Output your response in this EXACT format (do not deviate):

        ## Scan Results

        **Issue ID**: <ID or None>
        **Category**: <category or None>
        **Severity**: <🔴 HIGH, 🟡 MEDIUM, 🟢 LOW, or None>
        **Description**: <one-line summary of the highest-priority finding, or "No issues found.">

        ### What Changed
        No files were modified. This is a scan-only report from the OpenAI backend.

        ### Evidence
        <Summarize the evidence for the highest-priority finding, or state that no issues were found.>

        ### Verification
        <How to verify the finding, or state no changes were required.>

        ---

        ## Remaining Issues

        <numbered list of ALL findings, ordered by severity HIGH → MEDIUM → LOW>
        <EACH item MUST follow this EXACT format — note the square brackets:>
        1. **[CATEGORY_NAME] 🔴 HIGH**: Description — `file:line` evidence
        2. **[CATEGORY_NAME] 🟡 MEDIUM**: Description — `file:line` evidence
        3. **[CATEGORY_NAME] 🟢 LOW**: Description — `file:line` evidence

        Valid CATEGORY_NAME values (use these exactly, in square brackets):
        DRY_VIOLATIONS, DOCUMENTATION_DRIFT, TOOLING_CURRENCY, DEAD_CODE,
        CODE_QUALITY, SECURITY_HYGIENE, DEPENDENCY_HEALTH, CONFIG_CONSISTENCY

        ---

        ## Scan Summary

        | Category | Issues Found |
        |----------|-------------|
        | DRY Violations | X |
        | Documentation Drift | X |
        | Tooling Currency | X |
        | Dead Code | X |
        | Code Quality | X |
        | Security Hygiene | X |
        | Dependency Health | X |
        | Config Consistency | X |
        | **Total** | **X** |

        *Scan completed: {scan_timestamp}*
    """)

    user_message = f"{agent_prompt}\n\n---\n\n{repo_context}"

    client = OpenAI(api_key=api_key)

    try:
        # Build API kwargs — reasoning models (o1, o3, etc.) don't support
        # 'temperature' or 'max_tokens'; use 'max_completion_tokens' instead.
        api_kwargs: dict = dict(
            model=args.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            timeout=args.timeout,
        )
        is_reasoning = args.model.startswith(("o1", "o3"))
        if is_reasoning:
            api_kwargs["max_completion_tokens"] = 16384
        else:
            api_kwargs["temperature"] = 0.2
            api_kwargs["max_tokens"] = 4096

        response = client.chat.completions.create(**api_kwargs)

        report = response.choices[0].message.content or ""
        if not report.strip():
            raise ValueError("OpenAI API returned an empty report")

    except Exception as exc:
        report = build_scan_failure_report(str(exc))
        print(f"::error::OpenAI API error: {exc}", file=sys.stderr)
        write_report_and_exit(report, exit_code=1)

    # Write report
    write_report_and_exit(report)


if __name__ == "__main__":
    main()
