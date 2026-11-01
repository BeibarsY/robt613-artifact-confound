"""Assemble notebooks/final.ipynb from config.py, src/ and scripts/ (no repo dependency).

The notebook re-creates config.py and src/*.py with %%writefile, then runs the stage scripts
in order. Hypotheses and decision rules are copied verbatim from PLAN.md section 1.
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
STAGES = [  # (script, title, figure prefix)
    ("01_audit.py", "Stage 1 - Data audit", "fig_1"),
    ("02_signal_inspection.py", "Stage 2 - Signal inspection and filtering", "fig_2"),
    ("03_artifacts.py", "Stage 3 - Artifact handling and QC", "fig_3"),
    ("04_neurophys_check.py", "Stage 4 - Neurophysiological check", "fig_4"),
    ("05_decode_all_conditions.py", "Stages 5-6 - Features, classifiers and honest evaluation", "fig_6"),
    ("06_stats.py", "Stage 7 - Hypothesis tests", "fig_7"),
]
MODULES = ["config.py", "src/__init__.py", "src/timing.py", "src/plots.py", "src/data.py",
           "src/preprocess.py", "src/features.py", "src/neurophys.py", "src/decode.py", "src/stats.py"]


def plan_section_1() -> str:
    """Research question, hypotheses and decision rules, verbatim from PLAN.md."""
    text = (ROOT / "PLAN.md").read_text()
    m = re.search(r"## 1\. Research question.*?(?=\n---)", text, flags=re.S)
    return m.group(0)


def script_body(name: str) -> tuple[str, str]:
    """(docstring, code) of a stage script without the repo sys.path bootstrap."""
    src = (ROOT / "scripts" / name).read_text()
    doc = re.match(r'"""(.*?)"""', src, flags=re.S).group(1).strip()
    code = src[src.index('"""', 3) + 3:]
    code = re.sub(r"^import sys\n", "", code, flags=re.M)
    code = re.sub(r"^from pathlib import Path\n", "", code, flags=re.M)
    code = re.sub(r"^sys\.path\.insert\(.*\)\n", "", code, flags=re.M)
    return doc, code.strip() + "\n"


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    c = [md("# ROBT613 midterm - Artifacts as confounds in motor-imagery decoding\n\n"
            "**Authors:** Chidiadi Bethel Mba, Sungkar Bolat, Beibars Ybraiakhyn, Ramazan Zypayev\n\n"
            "Dataset: BNCI2014-001 (BCI Competition IV 2a) via MOABB. Run top to bottom on a fresh Colab CPU runtime.\n"
            "Set `ROBT613_MODE` below to `SMOKE` (1 subject, 1 session) for a quick check or `FULL` (9 subjects, 2 sessions).\n\n"
            "Every number in the report is read from `results/tables/`, produced by this notebook."),
         md(plan_section_1()),
         code("# Environment. Pins match the ROBT613 course notebooks. Other packages = Colab defaults (printed at the end).\n"
              "import sys, subprocess, os, time\n"
              "T_START = time.time()\n"
              "if 'google.colab' in sys.modules:\n"
              "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'mne==1.13.2', 'moabb==1.7.2'])\n"
              "os.environ.setdefault('ROBT613_MODE', 'FULL')\n"
              "os.makedirs('src', exist_ok=True)\n"
              "print('MODE', os.environ['ROBT613_MODE'])"),
         md("## Shared code\nThe project modules are written to disk so that each stage below imports the same functions.")]
    for mod in MODULES:
        c.append(code(f"%%writefile {mod}\n" + (ROOT / mod).read_text()))
    c.append(code("import importlib, warnings\nwarnings.filterwarnings('ignore')\nimport config\n"
                  "print('subjects', config.CFG['subjects'], 'sessions', config.CFG['sessions'], 'seed', config.SEED)"))
    for name, title, prefix in STAGES:
        doc, body = script_body(name)
        c.append(md(f"## {title}\n\n{doc}"))
        c.append(code(body))
        c.append(code("from IPython.display import Image, display\nimport glob\n"
                      f"for f in sorted(glob.glob('results/figures/{prefix}*.png')):\n"
                      "    print(f); display(Image(f, width=900))"))
    c.append(md("## Reproducibility record"))
    c.append(code("import importlib.metadata as md_, platform\n"
                  "print('python', platform.python_version())\n"
                  "print({p: md_.version(p) for p in ['mne', 'moabb', 'scikit-learn', 'numpy', 'scipy', 'pandas', 'matplotlib']})\n"
                  "print('subjects', config.CFG['subjects'], 'sessions', config.CFG['sessions'], 'seed', config.SEED)\n"
                  "import pandas as pd\nprint(pd.read_csv('results/timing.csv').tail(10).to_string(index=False))\n"
                  "print(f'total wall time {(time.time() - T_START) / 60:.1f} min')"))
    c.append(code("import shutil\nshutil.make_archive('robt613_results', 'zip', 'results')\nprint('results zipped -> robt613_results.zip')"))
    nb["cells"] = c
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    out = ROOT / "notebooks" / "final.ipynb"
    nbf.write(build(), out)
    print("wrote", out)
