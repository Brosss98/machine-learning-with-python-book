#!/usr/bin/env python3
"""Build tool for "Machine Learning with Python: From Linear Algebra to Deep Learning".

Copyright (c) 2026 Mario Medone - MIT License (code).

The book is written as notebooks in the "percent" text format (``# %%`` cells).
The workflow keeps all computation on YOUR computer:

    execute      .py (percent) -> executed .ipynb, with all outputs and figures stored inside
    build        executed notebooks -> static HTML in _build/html (no code is run by this step);
                 it also regenerates _build/html/sitemap.xml and robots.txt (see ``sitemap``)
    sitemap      (re)generate _build/html/sitemap.xml and robots.txt from the pages that were actually built
    publish      push _build/html to the gh-pages branch (GitHub only serves the files)
    preview      live preview server while you write (Jupyter Book)
    serve        serve a ``build --local`` result at http://localhost:8000
    setup-kernel register this Python environment as a Jupyter kernel (once, before ``execute``)
    set-version  update the version number in all the places where it appears

Typical release of a new chapter:

    python tools/book.py execute
    python tools/book.py build --strict
    python tools/book.py publish

Run ``python tools/book.py --help`` for the options. Use ``--dry-run`` to see the
commands without running them.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ["chapters", "appendices"]          # folders that contain percent notebooks (.py)
BUILD_DIR = REPO_ROOT / "_build"
HTML_DIR = BUILD_DIR / "html"
BUILD_MODE_FILE = BUILD_DIR / "build_mode.txt"      # "github-pages" or "local"
REPO_NAME = "machine-learning-with-python-book"
KERNEL_NAME = REPO_NAME                              # name of the Jupyter kernel for this environment
KERNEL_DISPLAY_NAME = "Python (ML book)"
DEFAULT_BASE_URL = f"/{REPO_NAME}"                  # GitHub Pages serves a project at /<repo-name>/
VERSION_FILE = REPO_ROOT / "VERSION"
FILES_WITH_VERSION = ["front_matter/index.md", "front_matter/copyright.md", "CITATION.cff"]
SITE_URL = os.environ.get("BOOK_SITE_URL", "https://brosss98.github.io/" + REPO_NAME).rstrip("/")   # public address of the book
SITEMAP_STATE = REPO_ROOT / "tools" / "sitemap_state.json"   # remembers when each page last changed (commit it)


# ----------------------------------------------------------------------------- helpers
def run(cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None, hint: str = "") -> None:
    """Print and (unless --dry-run) run a command from the repository root."""
    shown = " ".join(cmd)
    if env and "BASE_URL" in env:
        shown = f"BASE_URL={env['BASE_URL']!r} {shown}"
    print(f"$ {shown}")
    if dry_run:
        return
    full_env = {**os.environ, **env} if env else None
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=full_env)
    if result.returncode != 0:
        message = f"\nCommand failed with exit code {result.returncode}: {shown}"
        if hint:
            message += f"\n\n{hint}"
        sys.exit(message)


def need(program: str, hint: str, *, dry_run: bool) -> str:
    """Return the path of an executable, or stop with a helpful message."""
    path = shutil.which(program)
    if path is None:
        if dry_run:
            return program
        sys.exit(f"'{program}' was not found. {hint}\n"
                 "Did you activate the environment and run 'pip install -r requirements.txt'?")
    return path


def percent_notebooks() -> list[Path]:
    files: list[Path] = []
    for folder in SOURCE_DIRS:
        files += sorted((REPO_ROOT / folder).glob("*.py"))
    return files


def is_up_to_date(py_file: Path) -> bool:
    """True when the executed notebook exists and is newer than its source."""
    ipynb = py_file.with_suffix(".ipynb")
    return ipynb.exists() and ipynb.stat().st_mtime >= py_file.stat().st_mtime


# ----------------------------------------------------------------------------- commands
def tidy_notebook(py_file: Path) -> None:
    """Make sure the notebook starts with its title (a Markdown cell).

    Jupyter Book takes the page title from the first heading, but only if nothing comes before it.
    A stray first cell (empty, raw, or made only of comments) would stay visible as an empty box
    and make the title appear twice. Such cells are removed from the .ipynb; the .py is untouched.
    """
    ipynb = py_file.with_suffix(".ipynb")
    if not ipynb.exists():
        return
    nb = json.loads(ipynb.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    removed = 0
    while cells and cells[0]["cell_type"] != "markdown":
        first = cells[0]
        source = "".join(first.get("source", []))
        only_comments = all(not line.strip() or line.lstrip().startswith("#") for line in source.splitlines())
        if first["cell_type"] == "raw" or (first["cell_type"] == "code" and only_comments and not first.get("outputs")):
            cells.pop(0)
            removed += 1
        else:
            break
    if removed:
        ipynb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    head = cells[0] if cells else {"cell_type": "-", "source": ""}
    text = "".join(head.get("source", [])).strip().splitlines()[:1]
    print(f"first cell of {ipynb.name}: {head['cell_type']} {text[0][:60]!r}" if text else
          f"first cell of {ipynb.name}: {head['cell_type']} (empty)")
    if removed:
        print(f"  removed {removed} stray cell(s) placed before the title")


def cmd_execute(args: argparse.Namespace) -> None:
    jupytext = need("jupytext", "It is installed with the book requirements.", dry_run=args.dry_run)
    if args.files:
        targets = [Path(f).resolve() for f in args.files]
    else:
        targets = percent_notebooks()
    if not targets:
        sys.exit("No percent notebooks found.")
    done = skipped = 0
    for py_file in targets:
        rel = py_file.relative_to(REPO_ROOT)
        if not args.force and not args.files and is_up_to_date(py_file):
            print(f"skip   {rel}   (up to date; use --force to run it again)")
            skipped += 1
            continue
        print(f"\nexecuting {rel} ...")
        # "--set-kernel -" makes jupytext use the kernel of the ACTIVE environment, so that the
        # notebook is executed with the same Python (and packages) that you are using now.
        run([jupytext, "--set-kernel", "-", "--to", "ipynb", "--execute", str(rel)], dry_run=args.dry_run,
            hint=("If the error says 'No kernel found that matches the current python executable', "
                  "register this environment once with:\n    python tools/book.py setup-kernel\n"
                  "and run the command again."))
        if not args.dry_run:
            tidy_notebook(py_file)
        done += 1
    print(f"\nExecuted {done} notebook(s), skipped {skipped}.")


# ----------------------------------------------------------------------------- sitemap
def html_pages() -> list[tuple[str, Path]]:
    """Public pages found in _build/html, as (absolute URL, file). Assets and 404 pages are skipped."""
    pages = []
    for f in sorted(HTML_DIR.rglob("*.html")):
        rel = f.relative_to(HTML_DIR)
        if any(part.startswith(("_", ".")) or part == "build" for part in rel.parts[:-1]) or rel.name == "404.html":
            continue
        head = f.read_text(encoding="utf-8", errors="ignore")[:20000].lower()
        if re.search(r'<meta[^>]+name=["\']robots["\'][^>]+noindex', head):
            continue                                    # the page asks not to be indexed
        if rel.name == "index.html":                    # pagename/index.html is served at /pagename/
            folder = "/".join(rel.parts[:-1])
            url = f"{SITE_URL}/{folder}/" if folder else f"{SITE_URL}/"
        else:                                           # older layout: pagename.html is served at /pagename
            url = f"{SITE_URL}/{rel.with_suffix('').as_posix()}"
        pages.append((url, f))
    return sorted(pages)                                 # the home page comes first


def write_sitemap(*, dry_run: bool) -> None:
    """Write _build/html/sitemap.xml and robots.txt. <lastmod> changes only for pages whose HTML changed since the last run."""
    if dry_run:
        print(f"[dry-run] would write {HTML_DIR / 'sitemap.xml'} and robots.txt (addresses under {SITE_URL}/)")
        return
    pages = html_pages()
    if not pages:
        sys.exit(f"No HTML pages found in {HTML_DIR}. Run 'python tools/book.py build' first.")
    try:
        previous = json.loads(SITEMAP_STATE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        previous = {}
    today = datetime.date.today().isoformat()
    state, entries, changed = {}, [], 0
    for url, f in pages:
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        old = previous.get(url)
        lastmod = old["lastmod"] if old and old["sha256"] == digest else today
        changed += lastmod == today
        state[url] = {"sha256": digest, "lastmod": lastmod}
        entries.append(f"  <url>\n    <loc>{escape(url)}</loc>\n    <lastmod>{lastmod}</lastmod>\n  </url>")
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + "\n".join(entries) + "\n</urlset>\n")
    (HTML_DIR / "sitemap.xml").write_text(xml, encoding="utf-8")
    robots = ("# https://www.robotstxt.org/robotstxt.html\n"
              "User-agent: *\n"
              "Allow: /\n"
              f"Sitemap: {SITE_URL}/sitemap.xml\n")
    (HTML_DIR / "robots.txt").write_text(robots, encoding="utf-8")   # replaces the one made by Jupyter Book
    SITEMAP_STATE.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"sitemap.xml and robots.txt written: {len(pages)} page(s), {changed} new or changed since the last run (<lastmod> {today}); "
          f"remember to commit {SITEMAP_STATE.relative_to(REPO_ROOT)}")


def cmd_sitemap(args: argparse.Namespace) -> None:
    if BUILD_MODE_FILE.exists() and BUILD_MODE_FILE.read_text().strip() == "local":
        sys.exit("The last build was a local one (no public addresses). Run 'python tools/book.py build' first.")
    write_sitemap(dry_run=args.dry_run)


def cmd_build(args: argparse.Namespace) -> None:
    missing = [p.relative_to(REPO_ROOT) for p in percent_notebooks() if not p.with_suffix(".ipynb").exists()]
    if missing:
        sys.exit("These notebooks have no executed .ipynb yet. Run 'python tools/book.py execute' first:\n  "
                 + "\n  ".join(str(m) for m in missing))
    stale = [p.relative_to(REPO_ROOT) for p in percent_notebooks() if not is_up_to_date(p)]
    if stale:
        print("WARNING: the source is newer than the executed notebook for:\n  "
              + "\n  ".join(str(s) for s in stale)
              + "\n  (the book will show the OLD outputs; run 'python tools/book.py execute')\n")

    book = need("jupyter-book", "It is installed with the book requirements (jupyter-book>=2).", dry_run=args.dry_run)
    base_url = "" if args.local else os.environ.get("BOOK_BASE_URL", DEFAULT_BASE_URL)
    cmd = [book, "build", "--html"]
    if args.strict:
        cmd.append("--strict")                       # stop on broken links and similar problems
    # NOTE: no --execute flag: by default Jupyter Book does not run notebooks, it shows the stored outputs.
    run(cmd, dry_run=args.dry_run, env={"BASE_URL": base_url})
    if not args.dry_run:
        BUILD_DIR.mkdir(exist_ok=True)
        BUILD_MODE_FILE.write_text("local" if args.local else "github-pages")
    if not args.local:
        write_sitemap(dry_run=args.dry_run)           # always regenerated, so it is never out of date
    if args.local:
        print("\nLocal build ready. View it with: python tools/book.py serve   (do NOT publish this build)")
    else:
        print(f"\nBuild ready for GitHub Pages (BASE_URL={base_url!r}). Publish it with: python tools/book.py publish")


def cmd_preview(args: argparse.Namespace) -> None:
    book = need("jupyter-book", "It is installed with the book requirements (jupyter-book>=2).", dry_run=args.dry_run)
    run([book, "start"], dry_run=args.dry_run)


def cmd_serve(args: argparse.Namespace) -> None:
    if not (HTML_DIR / "index.html").exists() and not args.dry_run:
        sys.exit("Nothing to serve: run 'python tools/book.py build --local' first.")
    if BUILD_MODE_FILE.exists() and BUILD_MODE_FILE.read_text().strip() != "local":
        print("NOTE: the current build was made for GitHub Pages (links start with "
              f"'{DEFAULT_BASE_URL}'): styles may be missing here. Use 'build --local' to test locally.")
    print(f"Serving {HTML_DIR.relative_to(REPO_ROOT)} at http://localhost:{args.port}  (Ctrl+C to stop)")
    run([sys.executable, "-m", "http.server", str(args.port), "--directory", str(HTML_DIR)], dry_run=args.dry_run)


def cmd_publish(args: argparse.Namespace) -> None:
    if not args.dry_run:
        if not (HTML_DIR / "index.html").exists():
            sys.exit("Nothing to publish: run 'python tools/book.py build' first.")
        mode = BUILD_MODE_FILE.read_text().strip() if BUILD_MODE_FILE.exists() else "unknown"
        if mode != "github-pages":
            sys.exit(f"The current build is '{mode}', not a GitHub Pages build.\n"
                     "Run 'python tools/book.py build' (without --local) and then publish.")
    ghp = need("ghp-import", "It is installed with the book requirements.", dry_run=args.dry_run)
    # -n: add .nojekyll (GitHub must not run a site generator), -p: push, -f: force-push the gh-pages branch
    cmd = [ghp, "-n", "-p", "-f", str(HTML_DIR.relative_to(REPO_ROOT))]
    if args.remote:
        cmd += ["-r", args.remote]
    print("This pushes the contents of _build/html to the 'gh-pages' branch. No code runs on GitHub.")
    run(cmd, dry_run=args.dry_run)


def cmd_setup_kernel(args: argparse.Namespace) -> None:
    """Register the active Python environment as a Jupyter kernel (needed by jupytext --execute)."""
    print("Registering the active Python environment as a Jupyter kernel (user level).")
    run([sys.executable, "-m", "ipykernel", "install", "--user", "--name", KERNEL_NAME,
         "--display-name", KERNEL_DISPLAY_NAME], dry_run=args.dry_run,
        hint="Is 'ipykernel' installed in this environment? Run: pip install -r requirements.txt")
    print(f"\nDone. You can check it with: jupyter kernelspec list   (look for '{KERNEL_NAME}')")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_execute(args)
    cmd_build(args)


def cmd_set_version(args: argparse.Namespace) -> None:
    new = args.version
    if not re.fullmatch(r"\d+\.\d+\.\d+", new):
        sys.exit("The version must look like 0.2.0 (MAJOR.MINOR.PATCH).")
    old = VERSION_FILE.read_text().strip()
    if old == new:
        sys.exit(f"The version is already {new}.")
    for rel in FILES_WITH_VERSION:
        path = REPO_ROOT / rel
        text = path.read_text()
        count = text.count(old)
        print(f"{rel}: {count} occurrence(s) of {old} -> {new}")
        if not args.dry_run:
            path.write_text(text.replace(old, new))
    if not args.dry_run:
        VERSION_FILE.write_text(new + "\n")
    print(f"\nVersion set to {new}. Then, e.g.:  git commit -am 'Release v{new}' && git tag v{new}")


# ----------------------------------------------------------------------------- command line
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
        p.set_defaults(func=func)
        return p

    p = add("execute", cmd_execute, "run the percent notebooks and store the outputs in .ipynb files")
    p.add_argument("files", nargs="*", help="notebooks to run (default: all the ones that are out of date)")
    p.add_argument("--force", action="store_true", help="run all notebooks, even if up to date")

    p = add("build", cmd_build, "build the static HTML site (no code is executed)")
    p.add_argument("--local", action="store_true", help="build for local viewing (no GitHub Pages base URL)")
    p.add_argument("--strict", action="store_true", help="fail on warnings such as broken links")

    add("sitemap", cmd_sitemap, "regenerate _build/html/sitemap.xml and robots.txt (build does it already)")
    add("preview", cmd_preview, "live preview server for writing")
    p = add("serve", cmd_serve, "serve the built site on http://localhost:PORT")
    p.add_argument("--port", type=int, default=8000)
    p = add("publish", cmd_publish, "push the built site to the gh-pages branch")
    p.add_argument("--remote", help="git remote to push to (default: origin)")

    p = add("all", cmd_all, "execute, then build (for GitHub Pages)")
    p.add_argument("files", nargs="*", help=argparse.SUPPRESS)
    p.add_argument("--force", action="store_true")
    p.add_argument("--local", action="store_true")
    p.add_argument("--strict", action="store_true")

    add("setup-kernel", cmd_setup_kernel, "register this environment as a Jupyter kernel (once)")

    p = add("set-version", cmd_set_version, "set the version number everywhere it appears")
    p.add_argument("version", help="new version, for example 0.2.0")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
