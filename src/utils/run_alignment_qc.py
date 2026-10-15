"""Resumable all-tile QC with signal-qualified windows and explicit sparse fallbacks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import multiprocessing as mp
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.utils import validate_align_shifts as validation



def cache_signature(tile, params):
    """Tie cached curves to their scan, rasterization, source data and code."""
    files = []
    for key in ('filtered_dir', 'align_dir', 'raw_dir'):
        if key not in params:
            continue
        folder = Path(params[key])
        if key == 'filtered_dir':
            files.extend(folder/f"{tile}_{ch['id']}_result.csv" for ch in params['routing'])
        elif key == 'raw_dir':
            files.extend(folder/f"{tile}_{ch['id']}_inputs.json" for ch in params['routing'])
        else:
            files.extend(folder/f'{tile}_{suffix}.json' for suffix in ('offsets','measured_offsets'))
    stamps = {str(p): {'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
              for p in files if p.is_file()}
    code = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(validation.__file__),
                      ROOT/'src/core/point_cloud_aligner.py', ROOT/'src/core/z_linker.py')}
    return dict(evaluation=params['eval'],settings=params.get('settings'),tile_size=params.get('tile_size'),inputs=stamps,code=code)


def save_checkpoint(checkpoint, signature, rows):
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    part = checkpoint.with_suffix('.csv.part')
    pd.DataFrame(rows).to_csv(part,index=False)
    os.replace(part,checkpoint)
    meta = checkpoint.with_suffix('.json')
    part = meta.with_suffix('.json.part')
    part.write_text(json.dumps(dict(input_signature=signature,
                     csv_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()),sort_keys=True),encoding='utf-8')
    os.replace(part,meta)

def quality_process_tile(tile, params):
    checkpoint=Path(params['qc_checkpoint_dir'])/f'{tile}_curves.csv'
    signature=cache_signature(tile,params)
    meta=checkpoint.with_suffix('.json')
    valid=False
    if checkpoint.is_file() and meta.is_file():
        try:
            saved=json.loads(meta.read_text(encoding='utf-8'))
            valid=(saved['input_signature']==signature and saved['csv_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        except (OSError,ValueError,KeyError):
            pass
    if valid:
        df=pd.read_csv(checkpoint)
        minimum=params['eval']['sampling']['min_cells']
        if (df.n_ref_cells.ge(minimum).all() and df.n_tgt_cells.ge(minimum).all()
                and df.region_id.nunique()==params['eval']['sampling']['regions_per_tile']):
            df['signal_qualified']=True
            return df.to_dict('records'),[]
    try:
        rows,warnings=validation._process_tile(tile,params)
        for row in rows:
            row['signal_qualified']=True
        save_checkpoint(checkpoint,signature,rows)
        return rows,warnings
    except ValueError as exc:
        if 'Cannot select' not in str(exc):
            raise
        fallback=copy.deepcopy(params)
        fallback['eval']['sampling']['min_cells']=0
        rows,warnings=validation._process_tile(tile,fallback)
        for row in rows:
            row['signal_qualified']=False
        save_checkpoint(checkpoint,signature,rows)
        return rows,warnings+[f'{tile}: no signal-qualified window; sparse/empty fallback is diagnostic only, registration accuracy unknown']


def process_tile(tile,params):
    # main's output directory is not part of the original worker parameter set.
    output=Path(params['align_dir']).parent/params['eval']['output']['dir']
    params=dict(params,qc_checkpoint_dir=str(output/'tile_curves'))
    try:
        return quality_process_tile(tile,params)
    except (ValueError,FileNotFoundError,KeyError,pd.errors.EmptyDataError) as exc:
        return [],[f'{tile}: {exc}']


if __name__=='__main__':
    mp.freeze_support()
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    validation.process_tile=process_tile
    validation.main()
