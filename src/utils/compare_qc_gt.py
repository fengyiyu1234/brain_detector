"""Compare exhaustive, independent ROI annotations with saved predictions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.loader import load_config
from src.utils.pipeline_qc import dump_json, finite, integer, mapped_path, sha, write_csv


def maximum_matching(edges, n_gt):
    """Maximum-cardinality one-to-one matching using augmenting paths.

    Edge order prefers closer/higher-IoU pairs but does not globally optimize
    their total distance. This is not a greedy nearest-neighbor assignment.
    """
    pred_to_gt, gt_to_pred = {}, {}
    for start in range(len(edges)):
        queue, previous, visited_gt = deque([start]), {}, set()
        end = None
        while queue and end is None:
            pred = queue.popleft()
            for gt in edges[pred]:
                if not 0 <= gt < n_gt:
                    raise ValueError("Invalid matching edge")
                if gt in visited_gt:
                    continue
                visited_gt.add(gt)
                if gt not in gt_to_pred:
                    end = (pred, gt)
                    break
                other = gt_to_pred[gt]
                if other not in previous and other != start:
                    previous[other] = (pred, gt)
                    queue.append(other)
        if end is None:
            continue
        pred, gt = end
        while True:
            pred_to_gt[pred], gt_to_pred[gt] = gt, pred
            if pred == start:
                break
            pred, gt = previous[pred]
    return sorted(pred_to_gt.items())


def metrics(tp, fp, fn):
    return dict(tp=tp, fp=fp, fn=fn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                count_error=fp-fn,
                relative_count_error=(fp-fn)/(tp+fn) if tp+fn else None)


def iou(a, b):
    inter = max(0, min(a["x2"],b["x2"])-max(a["x1"],b["x1"]))
    inter *= max(0, min(a["y2"],b["y2"])-max(a["y1"],b["y1"]))
    aa=(a["x2"]-a["x1"])*(a["y2"]-a["y1"])
    bb=(b["x2"]-b["x1"])*(b["y2"]-b["y1"])
    return inter/(aa+bb-inter)


def read_annotations(path, mode, roi_ids):
    records, unique = defaultdict(list), set()
    required = {"roi_id","id","channel","class"} | ({"x1","y1","x2","y2","z"} if mode=="boxes2d" else {"cx","cy","cz"})
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if required-set(reader.fieldnames or []):
            raise ValueError(f"{path}: missing columns {sorted(required-set(reader.fieldnames or []))}")
        for row in reader:
            roi = row["roi_id"]
            if roi not in roi_ids:
                raise ValueError(f"{path}: unknown ROI {roi}")
            key = (roi,row["channel"],row["id"])
            if not row["id"] or key in unique:
                raise ValueError(f"{path}: empty/duplicate ID {key}")
            unique.add(key)
            numeric = ("x1","y1","x2","y2","z") if mode=="boxes2d" else ("cx","cy","cz")
            for column in numeric:
                row[column] = finite(row[column])
            if mode=="boxes2d":
                integer(row["z"])
                if row["x1"]>=row["x2"] or row["y1"]>=row["y2"]:
                    raise ValueError(f"{path}: invalid box {key}")
            if row.get("markers"):
                markers = json.loads(row["markers"])
                if not isinstance(markers,list) or any(not isinstance(m,str) for m in markers) or len(markers)!=len(set(markers)):
                    raise ValueError("markers must be a JSON array of unique strings")
                row["markers"] = set(markers)
            else:
                row["markers"] = None
            records[roi].append(row)
    return records


def evaluate(config):
    mode = config.get("mode", "points3d")
    if mode not in ("boxes2d","points3d","landmarks"):
        raise ValueError("mode must be boxes2d, points3d, or landmarks")
    if not config.get("coordinate_frame"):
        raise ValueError("Declare the common coordinate_frame of both CSVs")
    rois = config["rois"]
    roi_ids = {r["id"] for r in rois}
    if len(roi_ids)!=len(rois) or not rois or any(not rid for rid in roi_ids):
        raise ValueError("ROI IDs must be unique and nonempty")
    for roi in rois:
        if not roi.get("specimen_id"):
            raise ValueError("Each ROI needs specimen_id for biological-unit summaries")
        if mode!="landmarks" and roi.get("exhaustive") is not True:
            raise ValueError("Detection metrics require exhaustive=true for every ROI, including empty ones")
    mappings = config.get("path_mappings", {})
    pred_path = mapped_path(config["predictions"], mappings)
    gt_path = mapped_path(config["ground_truth"], mappings)
    pred = read_annotations(pred_path, mode, roi_ids)
    gt = read_annotations(gt_path, mode, roi_ids)
    scale = config.get("voxel_size_um", [.65,.65,8])
    if len(scale)!=3 or any(finite(s)<=0 for s in scale):
        raise ValueError("voxel_size_um must have three positive finite values")
    radius = finite(config.get("distance_threshold_um", 10))
    threshold = finite(config.get("iou_threshold", .5))
    if radius<=0 or not 0<threshold<=1:
        raise ValueError("Invalid distance or IoU threshold")
    per_roi, decisions, specimen = [], [], defaultdict(Counter)
    confusion, marker_counts, errors = Counter(), defaultdict(Counter), []
    for roi in rois:
        rid, sid = roi["id"], roi["specimen_id"]
        pp, gg = pred[rid], gt[rid]
        if len(pp)*len(gg)>config.get("max_pairs_per_roi", 2000000):
            raise ValueError(f"ROI {rid} exceeds max_pairs_per_roi; subdivide it before evaluating")
        edges, quality = [], {}
        for pi, p in enumerate(pp):
            neighbors = []
            for gi, g in enumerate(gg):
                if p["channel"]!=g["channel"]:
                    continue
                if mode=="landmarks":
                    if p["id"]!=g["id"]:
                        continue
                    value = math.sqrt(sum(((p[k]-g[k])*s)**2 for k,s in zip(("cx","cy","cz"),scale)))
                    neighbors.append((value,gi))
                elif config.get("class_aware",True) and p["class"]!=g["class"]:
                    continue
                elif mode=="boxes2d":
                    if p["z"]!=g["z"]:
                        continue
                    value=iou(p,g)
                    if value>=threshold:
                        neighbors.append((-value,gi))
                else:
                    value=math.sqrt(sum(((p[k]-g[k])*s)**2 for k,s in zip(("cx","cy","cz"),scale)))
                    if value<=radius:
                        neighbors.append((value,gi))
                if neighbors and neighbors[-1][1]==gi:
                    quality[(pi,gi)]=value
            edges.append([gi for _,gi in sorted(neighbors)])
        pairs=maximum_matching(edges,len(gg))
        if mode=="landmarks" and (len(pairs)!=len(pp) or len(pairs)!=len(gg)):
            raise ValueError(f"ROI {rid}: landmark IDs/channels must correspond exactly")
        row={"roi_id":rid,"specimen_id":sid,**metrics(len(pairs),len(pp)-len(pairs),len(gg)-len(pairs))}
        per_roi.append(row)
        specimen[sid].update({k:row[k] for k in ("tp","fp","fn")})
        used_p, used_g = set(), set()
        for pi,gi in pairs:
            p,g=pp[pi],gg[gi]
            used_p.add(pi);used_g.add(gi)
            q=quality[(pi,gi)]
            if mode!="boxes2d":
                errors.append(q)
            decisions.append(dict(roi_id=rid,specimen_id=sid,decision="TP",prediction_id=p["id"],gt_id=g["id"],
                                  metric="iou" if mode=="boxes2d" else "distance_um",value=q))
            confusion[(g["class"],p["class"])]+=1
            if p["markers"] is not None and g["markers"] is not None:
                markers=set(config.get("markers",[]))|p["markers"]|g["markers"]
                for marker in markers:
                    key=("tp" if marker in p["markers"] else "fn") if marker in g["markers"] else ("fp" if marker in p["markers"] else "tn")
                    marker_counts[marker][key]+=1
        for label, records, used in (("FP",pp,used_p),("FN",gg,used_g)):
            for index,r in enumerate(records):
                if index not in used:
                    decisions.append(dict(roi_id=rid,specimen_id=sid,decision=label,
                                          prediction_id=r["id"] if label=="FP" else "",gt_id=r["id"] if label=="FN" else ""))
    totals=Counter()
    for row in per_roi:
        totals.update({k:row[k] for k in ("tp","fp","fn")})
    report=dict(mode=mode,config=config,matching="maximum-cardinality one-to-one; edge preferences do not minimize total distance",
                micro=metrics(**totals),per_specimen=[{"specimen_id":sid,**metrics(**counts)} for sid,counts in sorted(specimen.items())],
                class_confusion_matched_only=[{"gt":g,"prediction":p,"count":n} for (g,p),n in sorted(confusion.items())],
                marker_metrics_matched_only={m:{**metrics(c["tp"],c["fp"],c["fn"]),"tn":c["tn"]} for m,c in marker_counts.items()},
                inputs=[{"path":str(p),"sha256":sha(p)} for p in (pred_path,gt_path)],
                code_sha256=sha(Path(__file__)),
                limitations=["Fixed-threshold metrics, not AP or instance-mask Dice.",
                             "Matched-only class/marker metrics exclude missed cells; interpret alongside detection recall.",
                             "No confidence interval: use independent specimens as resampling units."])
    if errors:
        ordered=sorted(errors)
        report["matched_distance_um"]={"n":len(errors),"mean":sum(errors)/len(errors),"p50":ordered[round((len(ordered)-1)*.5)],
                                       "p95":ordered[round((len(ordered)-1)*.95)]}
    if mode=="landmarks":
        report.pop("micro")
        report.pop("per_specimen")
        report["limitations"]=["Paired landmark error requires independent corresponding landmarks; no detection metrics.",
                              "Distances describe the chosen maximum-cardinality pairing for detection modes only."]
    return report,per_roi,decisions


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    args=parser.parse_args(argv)
    config=load_config(args.config)
    report,per_roi,decisions=evaluate(config)
    out=mapped_path(config["output_dir"],config.get("path_mappings",{}))
    out.mkdir(parents=True,exist_ok=True)
    dump_json(out/"gt_report.json",report)
    write_csv(out/"per_roi.csv",per_roi,["roi_id","specimen_id","tp","fp","fn","precision","recall","f1","count_error","relative_count_error"])
    write_csv(out/"matches.csv",decisions,["roi_id","specimen_id","decision","prediction_id","gt_id","metric","value"])
    print(f"Wrote {out}; metrics describe the explicitly annotated ROIs only.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
