# ENV: skip
# (a module imported by the figure scripts, not a generator of its own)
"""Data behind the patch-selection slides: the raw runs of the HuggingFace dataset
ASKabalan/furax-cs-results (KMEANS_C1D1S1, r = 0) through furax-cs `r_analysis snap`
(_data.streamed_snapshot), as runners/paper/section_42.sh does.

chain()  the monotonic-complexity chain of the thesis figure (Fig. 7 of the paper): K_bd raised at
         K_Td = K_bs = 500, then K_Td at K_bd = 10000, K_bs = 500, then K_bs at K_bd = 10000,
         K_Td = 3500, in the low-latitude region GAL060 (the region of the thesis figure; the
         paper runner names GAL040), 54 runs. For each point: patch counts, r-hat, sigma(r), the
         likelihood on r, and the variance of the reconstructed CMB.
scans()  the one-parameter scans of the gridding figures, in each region: vary K_bd
         (K_Td = K_bs = 500), vary K_Td (K_bd = 10000, K_bs = 500), vary K_bs (K_bd = 10000,
         K_Td = 500); r + sigma(r) against the patch count, the smallest per count, as furax-cs
         plots it.
"""

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _data import CACHE, UNSEEN, run_folders, streamed_snapshot

PARAMS = ("beta_dust", "temp_dust", "beta_pl")
# one raw folder per scan, and the folder of the last leg of the chain
FOLDERS = {"beta_dust": "BDXXX_TD500_BS500", "temp_dust": "BD10000_TDXXX_BS500",
           "beta_pl": "BD10000_TD500_BSXXX", "chain": "BD10000_TD3500_BSXXX"}
TD_STEPS = "|".join(str(k) for k in range(1000, 3501, 500))
BS_STEPS = "|".join(str(k) for k in range(1000, 10001, 500))
# The chain of the thesis figure (chap5/section_42/variance_vs_residual_r_total) is the
# low-latitude region GAL060, as its caption says; runners/paper/section_42.sh names GAL040.
REGION = "GAL060"
CHAIN = re.compile(rf"^(BD\d+_TD500_BS500|BD10000_TD({TD_STEPS})_BS500|BD10000_TD3500_BS({BS_STEPS}))_{REGION}$")


def n_patches(p):
    p = np.asarray(p)
    return int(np.unique(p[p > UNSEEN / 2]).size)


def _rows(key):
    folder = f"KMEANS_C1D1S1/{FOLDERS[key]}"
    runs = run_folders(folder) if key != "chain" else run_folders(folder, rf"_BS({BS_STEPS})_{REGION}$")
    return streamed_snapshot(f"kmeans_{FOLDERS[key]}" + ("_" + REGION if key == "chain" else ""),
                             folder, runs)


def _phase(kw):
    bd, td, bs = (int(x) for x in re.findall(r"(?:BD|TD|BS)(\d+)", kw))
    if td == 500 and bs == 500:
        return 0
    if bs == 500:
        return 1
    return 2


def _objects(arrays):
    """The likelihood grids differ in length from run to run: keep them as an object array."""
    out = np.empty(len(arrays), dtype=object)
    out[:] = arrays
    return out


PERMASK = Path("/home/wassim/Projects/CMB/furax-cs-results/raw/PAPER_ANALYSIS/section_42/snapshots/permask_thesis_order")


def _permask_rows():
    """The chain runs from the per-mask r_analysis snapshots of the raw runs (kw BD_TD_BS_GAL)."""
    import pyarrow.parquet as pq

    cols = ["kw", "r_best", "sigma_r_pos", "sigma_r_neg", "r_grid", "L_vals",
            "patches_beta_dust", "patches_temp_dust", "patches_beta_pl"]
    rows = []
    for f in sorted(PERMASK.glob("*.parquet")):
        rows += [dict(r, variance=np.nan) for r in pq.read_table(f, columns=cols).to_pylist()
                 if CHAIN.match(r["kw"])]
    return rows


def chain():
    npz = CACHE / "selection_chain.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        return {k: d[k] for k in d.files}
    seen, pts = set(), []
    for key in ("permask",):
        for r in _permask_rows():
            if not CHAIN.match(r["kw"]) or r["kw"] in seen:
                continue
            seen.add(r["kw"])
            k = [n_patches(r[f"patches_{p}"]) for p in PARAMS]
            pts.append({"kw": r["kw"], "phase": _phase(r["kw"]), "k": k, "total": sum(k),
                        "var": r["variance"], "r": r["r_best"], "sp": r["sigma_r_pos"],
                        "sn": r["sigma_r_neg"], "r_grid": np.asarray(r["r_grid"]),
                        "L": np.asarray(r["L_vals"]),
                        "patches": [np.asarray(r[f"patches_{p}"]) for p in PARAMS]})
    pts.sort(key=lambda p: (p["phase"], p["total"]))
    out = {key: np.array([p[key] for p in pts]) for key in ("kw", "phase", "k", "total", "var", "r", "sp", "sn")}
    out["r_grid"] = _objects([p["r_grid"] for p in pts])
    out["L"] = _objects([p["L"] for p in pts])
    out["patches"] = np.stack([np.stack(p["patches"]) for p in pts])
    np.savez(npz, **out)
    return out


def scans():
    npz = CACHE / "selection_scans.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        return d["scans"].item()
    out = {}
    for p in PARAMS:
        rows = _rows(p)
        for region in ("GAL020", "GAL040", "GAL060"):
            best = {}
            for r in rows:
                if r["kw"].endswith(region):
                    k, y = n_patches(r[f"patches_{p}"]), r["r_best"] + r["sigma_r_pos"]
                    best[k] = min(y, best.get(k, np.inf))
            k = np.array(sorted(best))
            out[(p, region)] = (k, np.array([best[x] for x in k]))
    np.savez(npz, scans=np.array(out, dtype=object))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:                   # fetch one folder only (to run folders in parallel)
        for key in sys.argv[1:]:
            _rows(key)
        sys.exit()
    s = scans()
    for key, (k, rs) in s.items():
        print(key, len(k), "points, K", k.min(), "-", k.max(), "best", k[np.argmin(rs)])
    c = chain()
    print(len(c["kw"]), "chain points; totals", c["total"].min(), "-", c["total"].max())
    i = int(np.argmin(c["r"] + c["sp"]))
    print("optimum", c["kw"][i], c["total"][i], (c["r"][i] + c["sp"][i]) * 1e3)
