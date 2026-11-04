"""Combine disjoint split ranges produced by run_experiments.py."""

import argparse
import json
import sys
from pathlib import Path
from runpy import run_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs",type=Path,nargs="+",help="completed run output folders")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        p.error("output already exists")
    states=[json.loads((folder/"progress.json").read_text()) for folder in args.inputs]
    template=states[0]["config"].copy()
    keys_ignored={"split_start","splits"}
    rows=[]
    existing=set()
    for state in states:
        config=state["config"]
        for key,value in template.items():
            if key not in keys_ignored and config.get(key)!=value:
                raise ValueError(f"incompatible run configuration: {key}")
        for r in state["records"]:
            key=(r["split"],r["regime"],r["method"],r["phase"],r["dimension"])
            if key in existing:
                raise ValueError("Overlapping records: "+str(key))
            existing.add(key)
            rows.append(r)
    script=Path(__file__).resolve().parents[1]/"run_experiments.py"
    sys.path.insert(0,str(script.parent))
    writer=run_path(str(script))["write_results"]
    args.output.mkdir(parents=True)
    n_splits=len({r["split"] for r in rows})
    writer(args.output,rows,n_splits)
    (args.output/"merge_metadata.json").write_text(json.dumps({
        "sources":[str(s) for s in args.inputs],"common_config":{
            k:v for k,v in template.items() if k not in keys_ignored}},indent=2))
    print((args.output/"summary.md").read_text())

if __name__=="__main__":main()
