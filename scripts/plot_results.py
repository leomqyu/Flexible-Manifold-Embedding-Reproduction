"""Plot measured experiment summaries, without inventing missing results."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary",type=Path,help="summary.csv from run_experiments.py")
    p.add_argument("--phase",default="test",choices=("test","unlabeled"))
    p.add_argument("--regime",type=int)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    table=pd.read_csv(args.summary)
    table=table[table.phase==args.phase]
    if args.regime is not None:
        table=table[table.regime==args.regime]
    if len(table)==0:
        raise ValueError("No matching data to plot")
    fig,ax=plt.subplots(figsize=(7,4))
    for method,records in table.groupby("method"):
        rows=records.sort_values("dimension")
        x=rows.dimension.to_numpy()
        y=rows.mean_pct.to_numpy()
        e=rows.std_pct.fillna(0).to_numpy()
        ax.errorbar(x,y,yerr=e,label=method,marker="o",capsize=2)
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"{args.phase.capitalize()} recognition ({args.summary.parent.name})")
    ax.legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    print(args.output)


if __name__=="__main__":main()
