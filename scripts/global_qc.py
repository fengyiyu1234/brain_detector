#!/usr/bin/env python
"""Read-only provenance and numeric QC for a completed global inference run."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, tempfile
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
CH = ("GFP", "RFP", "Olig2", "Sox9")
S2 = {"x1","y1","x2","y2","score","mean","class","z","tile_name","slice_name"}
S3 = {"x1","y1","x2","y2","score","mean","class","z"}
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def write(path, value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,delete=False) as f:
  json.dump(value,f,ensure_ascii=False,indent=2); temp=Path(f.name)
 os.replace(temp,path)
def bounds(xml):
 root=ET.parse(xml).getroot(); stacks=list(root.find("STACKS")); dims=root.find("dimensions")
 xyz=[(int(s.get("ABS_H")),int(s.get("ABS_V")),int(s.get("ABS_D"))) for s in stacks]
 xmin,ymin,zmin=map(min,zip(*xyz)); xmax,ymax,zmax=map(max,zip(*xyz))
 return {"width":xmax-xmin+2048,"height":ymax-ymin+2048,"z":int(dims.get("stack_slices"))-zmax+zmin,"z_start":zmax}
def inspect(path, required, b):
 out={"path":str(path),"exists":path.is_file(),"rows":0,"schema_missing":[],"nan_or_inf":0,"invalid_box":0,"xy_out_of_bounds":0,"z_out_of_bounds":0}
 if not out["exists"] or path.stat().st_size==0:return out
 with path.open(newline="",encoding="utf-8-sig") as f:
  rows=csv.DictReader(f); out["schema_missing"]=sorted(required-set(rows.fieldnames or ()))
  for r in rows:
   out["rows"]+=1
   try:
    x1,y1,x2,y2,z=(float(r[k]) for k in ("x1","y1","x2","y2","z"))
    if not all(map(math.isfinite,(x1,y1,x2,y2,z))):out["nan_or_inf"]+=1;continue
    if not(x1<x2 and y1<y2):out["invalid_box"]+=1
    if x1<0 or y1<0 or x2>b["width"] or y2>b["height"]:out["xy_out_of_bounds"]+=1
    if z<1 or z>b["z"]:out["z_out_of_bounds"]+=1
   except (KeyError,TypeError,ValueError):out["nan_or_inf"]+=1
 return out
def offsets(d):
 bad=[]; files=sorted(d.glob("*_offsets.json"))
 for p in files:
  raw=json.loads(p.read_text(encoding="utf-8")); value=raw.get("Olig2",raw.get("offsets",{}).get("Olig2"))
  if value is None or any(float(v)!=0 for v in (value.values() if isinstance(value,dict) else value)):bad.append(p.name)
 return {"count":len(files),"olig2_nonzero_or_missing":bad}
def main():
 a=argparse.ArgumentParser();a.add_argument("--result-dir",required=True);a.add_argument("--xml",required=True);a.add_argument("--sample-id",default="T4");args=a.parse_args()
 result=Path(args.result_dir);xml=Path(args.xml);out=result/"5_analysis_report"/"global_qc";b=bounds(xml)
 s2={c:result/"2_global_2d_raw"/(c+"_2d_global.csv") for c in CH};s3={c:result/"3_channel_3d"/(c+"_3d_tracked.csv") for c in CH};coloc=result/"4_colocalization"/"coloc_result.csv"
 tracked=[xml,result/"runtime_config.json",result/"inference.log",coloc,result/"global_summary_statistics.csv",*s2.values(),*s3.values()]
 manifest={"sample_id":args.sample_id,"created_utc":datetime.now(timezone.utc).isoformat(),"alignment_reference":"GFP","global_coordinate_frame":"Olig2/488","files":[{"path":str(p.resolve()),"size_bytes":p.stat().st_size,"mtime_utc":datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),"sha256":sha(p)} for p in tracked if p.is_file()]}
 q={"canvas":b,"stage2":{c:inspect(p,S2,b) for c,p in s2.items()},"stage3":{c:inspect(p,S3,b) for c,p in s3.items()},"coloc":inspect(coloc,S2,b),"offsets":offsets(result/"0_channel_alignment")}
 groups=[*q["stage2"].values(),*q["stage3"].values(),q["coloc"]];q["failures"]=[x["path"] for x in groups if not x["exists"] or not x["rows"] or x["schema_missing"] or x["nan_or_inf"] or x["invalid_box"]]
 if q["offsets"]["count"]!=45 or q["offsets"]["olig2_nonzero_or_missing"]:q["failures"].append("offsets")
 q["pass"]=not q["failures"];write(out/"t4_qc_manifest.json",manifest);write(out/"numeric_qc.json",q)
 lines=["# T4 numeric QC","","Status: "+("PASS" if q["pass"] else "FAIL"),"","| file | rows | NaN/Inf | invalid | XY OOB | Z OOB |","|---|---:|---:|---:|---:|---:|"]
 for x in groups:lines.append("| {} | {:,} | {:,} | {:,} | {:,} | {:,} |".format(Path(x["path"]).name,x["rows"],x["nan_or_inf"],x["invalid_box"],x["xy_out_of_bounds"],x["z_out_of_bounds"]))
 (out/"numeric_qc.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print("Wrote {}: {}".format(out,"PASS" if q["pass"] else "FAIL"))
if __name__=="__main__":main()
