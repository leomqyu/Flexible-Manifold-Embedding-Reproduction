"""Collect measured runs, emit reproduction tables and accuracy-vs-dimension plots.

Paper values are transcribed from Tables I and II of Nie et al., IEEE TIP 2010.
No value in the generated tables is estimated from a plot.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["umist", "cmu_pie", "yale_b", "coil20", "20news"]
DISPLAY = {"umist": "UMIST", "cmu_pie": "CMU PIE", "yale_b": "YALE-B",
           "coil20": "COIL-20", "20news": "20-NEWS"}
LABELS = {"umist": [1, 2, 3], "cmu_pie": [1, 2, 3], "yale_b": [1, 2, 3],
          "coil20": [1, 2, 3], "20news": [10, 20, 30]}
SEMI_METHODS = ["FME", "LapRLS/L", "LGC", "GFHF"]
UNSUP_METHODS = ["FME/U", "PCA", "LPP", "LPP-SR"]

# Table I: (dataset, labels, method, unlabeled mean, unlabeled sd, test mean, test sd)
PAPER_I_ROWS = [
    ("UMIST",1,"MFA",None,None,None,None),("UMIST",1,"GFHF",63.6,6.2,None,None),
    ("UMIST",1,"LGC",64.5,5.9,None,None),("UMIST",1,"TCA",63.2,5.2,62.9,5.8),
    ("UMIST",1,"SDA",56.2,5.4,55.6,5.1),("UMIST",1,"LapRLS/L",58.1,5.9,57.9,6.1),
    ("UMIST",1,"FME",63.5,5.5,63.1,5.4),
    ("UMIST",2,"MFA",70.5,4.2,70.8,3.9),("UMIST",2,"GFHF",79.1,3.9,None,None),
    ("UMIST",2,"LGC",79.3,3.6,None,None),("UMIST",2,"TCA",78.0,4.3,77.9,4.2),
    ("UMIST",2,"SDA",76.8,4.2,76.2,4.3),("UMIST",2,"LapRLS/L",74.9,4.6,74.3,4.5),
    ("UMIST",2,"FME",79.7,4.5,79.1,4.2),
    ("UMIST",3,"MFA",81.1,3.9,80.6,4.2),("UMIST",3,"GFHF",85.8,3.5,None,None),
    ("UMIST",3,"LGC",83.8,3.9,None,None),("UMIST",3,"TCA",83.9,4.1,83.6,3.8),
    ("UMIST",3,"SDA",83.7,3.8,83.2,3.9),("UMIST",3,"LapRLS/L",82.1,3.9,81.7,3.8),
    ("UMIST",3,"FME",86.9,3.2,86.1,3.1),
    ("YALE-B",1,"MFA",None,None,None,None),("YALE-B",1,"GFHF",22.5,2.9,None,None),
    ("YALE-B",1,"LGC",29.2,3.1,None,None),("YALE-B",1,"TCA",38.5,3.0,39.6,3.2),
    ("YALE-B",1,"SDA",38.2,3.0,39.0,3.1),("YALE-B",1,"LapRLS/L",52.9,3.6,53.2,3.1),
    ("YALE-B",1,"FME",53.9,3.3,54.2,2.9),
    ("YALE-B",2,"MFA",49.6,4.6,49.1,4.9),("YALE-B",2,"GFHF",35.9,3.3,None,None),
    ("YALE-B",2,"LGC",42.1,3.1,None,None),("YALE-B",2,"TCA",71.6,3.3,71.4,3.1),
    ("YALE-B",2,"SDA",72.1,3.6,71.9,3.2),("YALE-B",2,"LapRLS/L",73.9,3.2,73.6,2.9),
    ("YALE-B",2,"FME",75.1,2.6,75.0,2.7),
    ("YALE-B",3,"MFA",68.1,2.9,68.6,2.8),("YALE-B",3,"GFHF",45.2,3.9,None,None),
    ("YALE-B",3,"LGC",49.6,3.5,None,None),("YALE-B",3,"TCA",81.5,2.9,81.2,2.3),
    ("YALE-B",3,"SDA",83.8,2.1,83.0,2.1),("YALE-B",3,"LapRLS/L",84.2,2.6,83.8,2.5),
    ("YALE-B",3,"FME",85.9,2.1,85.6,2.0),
    ("CMU PIE",1,"MFA",None,None,None,None),("CMU PIE",1,"GFHF",33.9,3.3,None,None),
    ("CMU PIE",1,"LGC",36.2,3.1,None,None),("CMU PIE",1,"TCA",61.3,3.6,60.8,3.7),
    ("CMU PIE",1,"SDA",59.4,3.2,58.7,2.8),("CMU PIE",1,"LapRLS/L",57.9,3.1,57.5,2.6),
    ("CMU PIE",1,"FME",63.2,2.8,62.7,2.6),
    ("CMU PIE",2,"MFA",72.1,2.6,71.8,2.3),("CMU PIE",2,"GFHF",47.8,2.6,None,None),
    ("CMU PIE",2,"LGC",47.9,2.3,None,None),("CMU PIE",2,"TCA",78.6,2.4,78.4,2.3),
    ("CMU PIE",2,"SDA",81.5,2.1,81.2,2.0),("CMU PIE",2,"LapRLS/L",79.1,2.2,79.0,1.8),
    ("CMU PIE",2,"FME",81.8,2.0,81.5,1.9),
    ("CMU PIE",3,"MFA",83.0,1.9,83.1,1.8),("CMU PIE",3,"GFHF",55.8,2.1,None,None),
    ("CMU PIE",3,"LGC",55.9,1.9,None,None),("CMU PIE",3,"TCA",86.9,1.2,86.6,1.1),
    ("CMU PIE",3,"SDA",88.6,1.2,88.8,1.1),("CMU PIE",3,"LapRLS/L",87.8,1.1,87.7,1.1),
    ("CMU PIE",3,"FME",89.1,1.2,88.9,1.0),
    ("COIL-20",1,"MFA",None,None,None,None),("COIL-20",1,"GFHF",78.6,2.1,None,None),
    ("COIL-20",1,"LGC",78.5,2.6,None,None),("COIL-20",1,"TCA",70.6,2.9,70.5,2.8),
    ("COIL-20",1,"SDA",59.9,2.5,59.8,3.2),("COIL-20",1,"LapRLS/L",60.5,3.2,60.6,3.5),
    ("COIL-20",1,"FME",75.1,3.2,75.5,3.1),
    ("COIL-20",2,"MFA",70.2,2.6,70.1,3.2),("COIL-20",2,"GFHF",83.2,2.2,None,None),
    ("COIL-20",2,"LGC",82.9,2.1,None,None),("COIL-20",2,"TCA",78.1,2.5,77.9,2.1),
    ("COIL-20",2,"SDA",73.2,2.7,73.3,2.5),("COIL-20",2,"LapRLS/L",73.5,2.9,73.1,2.5),
    ("COIL-20",2,"FME",82.2,2.9,81.9,3.1),
    ("COIL-20",3,"MFA",76.5,2.5,76.2,2.3),("COIL-20",3,"GFHF",85.6,2.0,None,None),
    ("COIL-20",3,"LGC",85.9,2.1,None,None),("COIL-20",3,"TCA",81.7,2.2,81.5,2.6),
    ("COIL-20",3,"SDA",78.3,2.2,78.1,2.5),("COIL-20",3,"LapRLS/L",78.6,2.6,78.8,2.5),
    ("COIL-20",3,"FME",86.1,2.3,85.6,2.6),
    ("20-NEWS",10,"MFA",46.5,7.2,46.2,7.6),("20-NEWS",10,"GFHF",72.5,7.5,None,None),
    ("20-NEWS",10,"LGC",80.9,2.3,None,None),("20-NEWS",10,"TCA",55.2,5.3,56.6,5.2),
    ("20-NEWS",10,"SDA",57.5,3.9,58.1,5.8),("20-NEWS",10,"LapRLS/L",61.9,4.5,62.2,4.6),
    ("20-NEWS",10,"FME",83.2,3.2,82.5,3.6),
    ("20-NEWS",20,"MFA",61.9,5.9,61.3,6.2),("20-NEWS",20,"GFHF",83.6,2.5,None,None),
    ("20-NEWS",20,"LGC",83.9,1.5,None,None),("20-NEWS",20,"TCA",68.6,3.3,67.5,3.2),
    ("20-NEWS",20,"SDA",72.9,3.8,73.6,3.6),("20-NEWS",20,"LapRLS/L",75.6,2.5,76.2,2.6),
    ("20-NEWS",20,"FME",88.2,1.9,87.6,2.0),
    ("20-NEWS",30,"MFA",70.9,4.5,71.5,4.1),("20-NEWS",30,"GFHF",86.1,1.1,None,None),
    ("20-NEWS",30,"LGC",85.5,1.0,None,None),("20-NEWS",30,"TCA",75.6,3.2,73.9,2.5),
    ("20-NEWS",30,"SDA",78.9,2.7,80.5,2.2),("20-NEWS",30,"LapRLS/L",80.9,1.7,81.2,1.9),
    ("20-NEWS",30,"FME",90.1,1.3,89.6,1.5),
]

# Table II: (dataset, method, mean, sd, optimal dimension, parameter note)
PAPER_II_ROWS = [
    ("UMIST","PCA",82.6,3.2,43,""),("UMIST","LPP",80.4,3.5,31,""),
    ("UMIST","LPP-SR",81.5,3.2,35,"lambda=1e-9"),
    ("UMIST","FME/U",86.3,2.8,35,"mu=1, gamma=1"),
    ("YALE-B","PCA",43.1,1.2,60,""),("YALE-B","LPP",53.3,3.1,60,""),
    ("YALE-B","LPP-SR",60.5,3.0,60,"lambda=1e-3"),
    ("YALE-B","FME/U",68.2,2.5,60,"mu=1e-9, gamma=1e3"),
    ("CMU PIE","PCA",53.4,1.5,55,""),("CMU PIE","LPP",85.0,1.2,55,""),
    ("CMU PIE","LPP-SR",81.7,2.1,55,"lambda=1e-5"),
    ("CMU PIE","FME/U",89.1,1.0,55,"mu=1e-9, gamma=1e3"),
]

PAPER_I = pd.DataFrame(PAPER_I_ROWS, columns=[
    "dataset","labels","method","paper_unlabeled","paper_unlabeled_sd",
    "paper_test","paper_test_sd"])
PAPER_II = pd.DataFrame(PAPER_II_ROWS, columns=[
    "dataset","method","paper_mean","paper_sd","paper_dim","paper_params"])


def copy_run(src, dst):
    if not src.is_dir():
        print("missing run:", src)
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("summary.csv", "summary.md", "runs.csv", "progress.json", "merge_metadata.json"):
        file = src / name
        if file.exists():
            shutil.copy2(file, dst / name)
    return True


def load_summary(folder):
    file = folder / "summary.csv"
    return pd.read_csv(file) if file.exists() else None


def metric(summary, method, phase, regime, column="mean_pct"):
    if summary is None:
        return None
    row = summary[(summary.method == method) & (summary.phase == phase) &
                  (summary.regime == regime)]
    if row.empty:
        return None
    return float(row.iloc[0][column])


def fmt(value, sd=None):
    if value is None or not np.isfinite(value):
        return "N/A"
    if sd is None or not np.isfinite(sd):
        return f"{value:.2f}"
    return f"{value:.2f} ± {sd:.2f}"


def delta(ours, paper):
    if ours is None or paper is None or not np.isfinite(ours) or not np.isfinite(paper):
        return "N/A"
    return f"{ours - paper:+.2f}"


def build_semi_table(results):
    rows = []
    paper = PAPER_I.set_index(["dataset","labels","method"])
    for ds in DATASETS:
        primary = load_summary(results / "semi_supervised" / ds)
        paperpca = load_summary(results / "semi_supervised" / f"{ds}_paper_pca")
        for labels in LABELS[ds]:
            for method in SEMI_METHODS:
                key = (DISPLAY[ds], labels, method)
                p = paper.loc[key] if key in paper.index else None
                pu = None if p is None or pd.isna(p.paper_unlabeled) else float(p.paper_unlabeled)
                pt = None if p is None or pd.isna(p.paper_test) else float(p.paper_test)
                ou = metric(primary, method, "unlabeled", labels)
                ot = metric(primary, method, "test", labels)
                ou_sd = metric(primary, method, "unlabeled", labels, "std_pct")
                ot_sd = metric(primary, method, "test", labels, "std_pct")
                au = metric(paperpca, method, "unlabeled", labels)
                at = metric(paperpca, method, "test", labels)
                au_sd = metric(paperpca, method, "unlabeled", labels, "std_pct")
                at_sd = metric(paperpca, method, "test", labels, "std_pct")
                rows.append({
                    "Dataset": DISPLAY[ds], "Labels/class": labels, "Method": method,
                    "Our unlab (train-PCA)": fmt(ou, ou_sd),
                    "Paper unlab": fmt(pu, None if p is None or pd.isna(p.paper_unlabeled_sd) else float(p.paper_unlabeled_sd)),
                    "Δ unlab": delta(ou, pu),
                    "Our test (train-PCA)": fmt(ot, ot_sd),
                    "Paper test": fmt(pt, None if p is None or pd.isna(p.paper_test_sd) else float(p.paper_test_sd)),
                    "Δ test": delta(ot, pt),
                    "Our unlab (paper-PCA)": fmt(au, au_sd),
                    "Our test (paper-PCA)": fmt(at, at_sd),
                })
    return pd.DataFrame(rows)


def best_row(summary, method):
    if summary is None:
        return None
    rows = summary[summary.method == method]
    if rows.empty:
        return None
    row = rows.loc[rows.mean_pct.idxmax()]
    return row


def build_unsup_table(results):
    rows = []
    for ds in ["umist", "cmu_pie", "yale_b", "coil20"]:
        primary = load_summary(results / "unsupervised" / ds)
        paperpca = load_summary(results / "unsupervised" / f"{ds}_paper_pca")
        # merge targeted dimension runs for the paper-style protocol
        extra = []
        for pattern in ["*_unsup_pcaall_d35", "*_unsup_pcaall_d60", "*_unsup_pcaall_d33"]:
            for folder in (results / "unsupervised").glob(pattern):
                if folder.name.startswith(ds + "_"):
                    table = load_summary(folder)
                    if table is not None:
                        extra.append(table)
        if extra and paperpca is not None:
            paperpca = pd.concat([paperpca] + extra, ignore_index=True).drop_duplicates(
                subset=["regime","method","phase","dimension"], keep="first")
        paper = PAPER_II[PAPER_II.dataset == DISPLAY[ds]].set_index("method")
        for method in UNSUP_METHODS:
            b1 = best_row(primary, method)
            b2 = best_row(paperpca, method)
            p = paper.loc[method] if method in paper.index else None
            pm = None if p is None else float(p.paper_mean)
            pdim = None if p is None else int(p.paper_dim)
            if b1 is None:
                our1 = "N/A"
            else:
                our1 = f"{b1.mean_pct:.2f} ± {b1.std_pct:.2f} (d={int(b1.dimension)}, n={int(b1.n)})"
            if b2 is None:
                our2 = "N/A"
            else:
                our2 = f"{b2.mean_pct:.2f} ± {b2.std_pct:.2f} (d={int(b2.dimension)}, n={int(b2.n)})"
            rows.append({
                "Dataset": DISPLAY[ds], "Method": method,
                "Our best (train-PCA)": our1,
                "Our best (paper-PCA)": our2,
                "Paper best": fmt(pm, None if p is None else float(p.paper_sd)) + (f" (d={pdim})" if pdim else ""),
                "Δ (paper-PCA - paper)": delta(None if b2 is None else float(b2.mean_pct), pm),
            })
    return pd.DataFrame(rows)


def plot_dataset(results, ds, output):
    primary = load_summary(results / "unsupervised" / ds)
    paperpca = load_summary(results / "unsupervised" / f"{ds}_paper_pca")
    if primary is None and paperpca is None:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, table, style in (("train-PCA", primary, "--"), ("paper-PCA", paperpca, "-")):
        if table is None:
            continue
        for method, rows in table.groupby("method"):
            rows = rows.sort_values("dimension")
            ax.errorbar(rows.dimension, rows.mean_pct, yerr=rows.std_pct.fillna(0),
                        marker="o", ms=3.5, lw=1.4, ls=style, capsize=2,
                        label=f"{method} ({label})")
    paper = PAPER_II[PAPER_II.dataset == DISPLAY[ds]]
    if not paper.empty:
        row = paper[paper.method == "FME/U"].iloc[0]
        ax.errorbar([row.paper_dim], [row.paper_mean], yerr=[row.paper_sd],
                    fmt="k*", ms=12, capsize=3, label="FME/U (paper, best)")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"{DISPLAY[ds]} unsupervised recognition")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True,
                        help="root containing the raw run directories")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    outputs, results = args.outputs, args.results
    results.mkdir(parents=True, exist_ok=True)

    mapping = {}
    for ds in DATASETS:
        mapping[outputs / f"{ds}_semi"] = results / "semi_supervised" / ds
        mapping[outputs / f"{ds}_unsup"] = results / "unsupervised" / ds
    for ds in ["umist", "cmu_pie", "yale_b", "coil20"]:
        mapping[outputs / f"{ds}_semi_pcaall"] = results / "semi_supervised" / f"{ds}_paper_pca"
        mapping[outputs / f"{ds}_unsup_pcaall"] = results / "unsupervised" / f"{ds}_paper_pca"
        for dim in (35, 60, 33):
            src = outputs / f"{ds}_unsup_pcaall_d{dim}"
            if src.is_dir():
                mapping[src] = results / "unsupervised" / src.name
    for src, dst in mapping.items():
        copy_run(src, dst)

    PAPER_I.to_csv(results / "paper_table_I.csv", index=False)
    PAPER_II.to_csv(results / "paper_table_II.csv", index=False)
    semi = build_semi_table(results)
    unsup = build_unsup_table(results)
    semi.to_csv(results / "table_a_semi.csv", index=False)
    unsup.to_csv(results / "table_b_unsup.csv", index=False)

    figures = results / "figures"
    for ds in ["umist", "cmu_pie", "yale_b", "coil20"]:
        plot_dataset(results, ds, figures / f"{ds}_unsup_accuracy_vs_dimension.png")

    lines = ["# Reproduction tables", "",
             "Locally reproduced values are mean ± sample standard deviation over 20 random splits.",
             "`train-PCA` fits PCA on the training split only (no test-feature leakage).",
             "`paper-PCA` fits PCA on training+test features, matching the transductive preprocessing",
             "implied by the paper's reported 35/55/60-dimensional FME/U results; it is optimistic.",
             "`paper_oracle` selection means the best hyperparameters were chosen using the reported",
             "evaluation labels, as the paper's best-of-grid protocol does; this is not leakage-free validation.", "",
             "## Table A: semi-supervised FME", "", semi.to_markdown(index=False), "",
             "## Paper-only methods (not implemented locally)", ""]
    paper_only = PAPER_I[PAPER_I.method.isin(["MFA", "TCA", "SDA"])].copy()
    paper_only = paper_only.rename(columns={"dataset": "Dataset", "labels": "Labels/class",
                                            "method": "Method", "paper_unlabeled": "Paper unlab",
                                            "paper_unlabeled_sd": "sd", "paper_test": "Paper test",
                                            "paper_test_sd": "sd_test"})
    paper_only = paper_only.where(pd.notna(paper_only), "N/A")
    lines.append(paper_only.to_markdown(index=False))
    lines += ["", "## Table B: unsupervised FME/U", "", unsup.to_markdown(index=False), ""]
    (results / "tables.md").write_text("\n".join(lines) + "\n")
    print((results / "tables.md").read_text())


if __name__ == "__main__":
    main()
