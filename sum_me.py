#!/usr/bin/env python3

import os
from pathlib import Path

# ============================================
# Configuration
# ============================================

PROJECT_DIR = "/home/mrtenkorang/open-source-projects/ps-gnn/ps-gnn"
OUTPUT_FILE = f"{PROJECT_DIR.split('/')[-1]}.txt"

IGNORE_DIRS = {
   ".git",
   ".github",
   ".idea",
   ".vscode",
   "__pycache__",
   "node_modules",
   "venv",
   ".venv",
   "dist",
   "build",
   ".next",
   "coverage",
   ".gstack",
   ".pytest_cache",
   "target",
   "ipynb_checkpoints",
   "dist",
    "conda-recipe",
    "books",
    
}

IGNORE_EXTENSIONS = {
   ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico",
   ".pdf", ".zip", ".tar", ".gz", ".7z",
   ".exe", ".dll", ".so", ".class", ".jar",
   ".pyc", ".db", ".sqlite", ".json", ".log",
   ".mp3", ".mp4", ".mov", ".avi", ".gitignore",
   ".woff", ".woff2", ".ttf", ".md", ".pyc", ".env", ".sql", ".ipynb", ".csv", ".tsv", ".xlsx", ".xls", ".pptx", ".ppt", ".docx", ".doc",".env"
}

# ============================================
# Helpers
# ============================================

def is_binary(path: Path):
   try:
       with open(path, "rb") as f:
           return b"\0" in f.read(4096)
   except Exception:
       return True


def should_skip(path: Path):
   if any(part in IGNORE_DIRS for part in path.parts):
       return True

   if path.suffix.lower() in IGNORE_EXTENSIONS:
       return True

   if path.is_file() and is_binary(path):
       return True

   return False


# ============================================
# Directory Tree
# ============================================

def build_tree(root: Path):
   lines = []

   def walk(directory: Path, prefix=""):
       entries = sorted(
           [
               p for p in directory.iterdir()
               if not should_skip(p)
           ],
           key=lambda x: (x.is_file(), x.name.lower())
       )

       for index, entry in enumerate(entries):
           connector = "└── " if index == len(entries) - 1 else "├── "
           lines.append(prefix + connector + entry.name)

           if entry.is_dir():
               extension = "    " if index == len(entries) - 1 else "│   "
               walk(entry, prefix + extension)

   lines.append(root.name)
   walk(root)

   return "\n".join(lines)


# ============================================
# Merge Files
# ============================================

def collect_files(root: Path):
   files = []

   for file in root.rglob("*"):
       if not file.is_file():
           continue

       if should_skip(file):
           continue

       files.append(file)

   return sorted(files)


def merge_project(root_dir, output_file):

   root = Path(root_dir).resolve()

   files = collect_files(root)
   tree = build_tree(root)

   with open(output_file, "w", encoding="utf-8") as out:

       out.write("# PROJECT OVERVIEW\n\n")
       out.write(f"Root: {root}\n")
       out.write(f"Files: {len(files)}\n\n")

       out.write("## Directory Tree\n\n")
       out.write("```\n")
       out.write(tree)
       out.write("\n```\n\n")

       out.write("=" * 80 + "\n")
       out.write("PROJECT FILES\n")
       out.write("=" * 80 + "\n\n")

       for file in files:

           relative = file.relative_to(root)

           out.write("=" * 80 + "\n")
           out.write(f"BEGIN FILE: {relative}\n")
           out.write("=" * 80 + "\n\n")

           try:
               with open(file, "r", encoding="utf-8", errors="replace") as f:
                   out.write(f.read())
           except Exception as e:
               out.write(f"<<ERROR: {e}>>")

           out.write("\n\n")
           out.write("=" * 80 + "\n")
           out.write(f"END FILE: {relative}\n")
           out.write("=" * 80 + "\n\n")

   print(f"Done! Merged {len(files)} files into {output_file}")


if __name__ == "__main__":
   merge_project(PROJECT_DIR, OUTPUT_FILE)
