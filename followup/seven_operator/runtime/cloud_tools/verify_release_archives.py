"""Verify portable archive files; optionally restore local audit metadata."""
import argparse,hashlib,json
from pathlib import Path
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True);ap.add_argument('--archive-dir',type=Path,required=True);ap.add_argument('--study-root',type=Path);ap.add_argument('--filename');a=ap.parse_args();records=json.loads(a.index.read_text());rows=[]
    if a.filename:records=[r for r in records if r['filename']==a.filename];assert len(records)==1
    for r in records:
        assert Path(r['filename']).name==r['filename'];p=a.archive_dir/r['filename'];assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'];installed=False
        if a.study_root and r.get('case_raw_audit_sha256'):
            root=a.study_root/r['directory'];raw=root/'CASE_RAW_AUDIT.json';assert root.resolve().is_relative_to(a.study_root.resolve()) and digest(raw)==r['case_raw_audit_sha256'];v=root/'ARCHIVE_VERIFICATION.json'
            if v.exists():old=json.loads(v.read_text());assert old['bytes']==r['bytes'] and old['sha256']==r['sha256']
            else:v.write_text(json.dumps({'status':r['verification_status'],'bytes':r['bytes'],'sha256':r['sha256'],'weights_retained_losslessly_in_archive':r['weights_retained_losslessly_in_archive'],'verified_archive':str(p.resolve()),'restored_case_raw_audit_sha256':digest(raw),'scope':'Portable archive rehash; original case and frozen inputs unchanged.'},indent=2));installed=True
        rows.append({'filename':r['filename'],'bytes':r['bytes'],'sha256':r['sha256'],'portable_metadata_installed':installed});print(json.dumps(rows[-1]),flush=True)
    print(json.dumps({'verified_archives':len(rows),'new_model_queries':0}))
if __name__=='__main__':main()
