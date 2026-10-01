# ENV: skip
# (a module imported by the figure scripts, not a generator of its own)
"""Data access for the component-separation figures, from the public HuggingFace dataset
ASKabalan/furax-cs-results.

Every r likelihood of the section comes from the raw run folders (results.npz, best_params.npz,
mask.npy) passed through furax-cs's own `r_analysis snap`, with the same arguments as the paper's
runners (COMSEP_DATA/FINAL/runners/paper/section_4*.sh), so the numbers are the thesis's. The
processed rows of the dataset (data/c1d1s1/*.parquet) are not used: they do not reproduce the
thesis values (e.g. r + sigma(r) = 3.9e-3 for BD100_TD500_BS500_GAL020, where the raw run gives
1.5e-3, as in the thesis gridding figure).

`snapshot(name, runs, results_dirs, ...)` snaps a few runs and keeps the full rows. The selection
figures (09_selection) are plotted by furax-cs `r_analysis plot` itself; AGENTS.md holds the
commands.

Everything is cached in ../.cache, so each download or computation happens once.
"""

import os
import subprocess
import sys
from pathlib import Path

CACHE = Path(__file__).resolve().parent / ".cache"
REPO = "ASKabalan/furax-cs-results"
RAW_FILES = ("results.npz", "best_params.npz", "mask.npy")


def raw_root(runs):
    """Download raw run folders and return the local raw/RESULTS.

    `runs` are globs of run folders under raw/RESULTS/, e.g.
    "KMEANS_C1D1S1/BD4000_TD10_BSXXX/kmeans_c1d1s1_BD4000_TD10_BS50_GAL*"; only the three files a
    snapshot reads are fetched.
    """
    from huggingface_hub import snapshot_download

    patterns = [f"raw/RESULTS/{r.strip('/')}/{name}" for r in runs for name in RAW_FILES]
    root = snapshot_download(REPO, repo_type="dataset", allow_patterns=patterns,
                             local_dir=CACHE / "hf_raw")
    return Path(root) / "raw" / "RESULTS"


def _snap(name, runs, results_dirs, fetch, combine=False, names=None, noise_selection=None,
          sky="c1d1s1", env=None):
    """Run `r_analysis snap` once (cached) and return its parquet files."""
    out = CACHE / "snapshots" / f"{name}.parquet"
    done = sorted(out.parent.glob(f"{name}_0*.parquet"))
    if not done:
        root = raw_root(fetch)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(Path(sys.executable).parent / "r_analysis"), "snap", "-r", *runs,
               "-ird", *[str(root / d) + "/" for d in results_dirs],
               "--sky", sky, "--no-images", "--max-size", "10", "-o", str(out)]
        if combine:
            cmd.append("--combine")
        if names:
            cmd += ["--name", *names]
        if noise_selection:
            cmd += ["--noise-selection", noise_selection]
        print("running", " ".join(cmd[1:3]), name)
        subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})
        done = sorted(out.parent.glob(f"{name}_0*.parquet"))
    return done


def snapshot(name, runs, results_dirs, fetch, **kw):
    """Rows of an `r_analysis snap` over the given raw result folders, as a list of dicts.

    `results_dirs` are the -ird folders (under raw/RESULTS/), `fetch` the run-folder globs to
    download into them; `kw` are passed on (combine, names, noise_selection, sky, env).
    """
    import pyarrow.parquet as pq

    rows = []
    for f in _snap(name, runs, results_dirs, fetch, **kw):
        rows += pq.read_table(f).to_pylist()
    return rows
