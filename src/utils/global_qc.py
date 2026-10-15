"""Compatibility entry point for the generalized saved-output QC."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.pipeline_qc import main as pipeline_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--result-dir", help="Legacy single-sample invocation")
    parser.add_argument("--xml")
    parser.add_argument("--sample-id", default="sample")
    args = parser.parse_args(argv)
    if args.config:
        if args.result_dir or args.xml:
            parser.error("--config cannot be combined with --result-dir/--xml")
        return pipeline_main(["--config",args.config]+(["--samples",*args.samples] if args.samples else []))
    if not args.result_dir or not args.xml or args.samples:
        parser.error("Provide --config [--samples ...], or --result-dir and --xml")
    config={"samples":[{"id":args.sample_id,"result_dir":str(Path(args.result_dir).resolve()),"xml":str(Path(args.xml).resolve())}]}
    with tempfile.TemporaryDirectory(prefix="brain_qc_config_") as tmp:
        path=Path(tmp)/"qc.json"
        path.write_text(json.dumps(config),encoding="utf-8")
        return pipeline_main(["--config",str(path)])


if __name__=="__main__":
    raise SystemExit(main())
