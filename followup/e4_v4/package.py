"""Package reviewed E4-v4 pilot; exclude all mutable run artifacts."""
import zipfile
from generate import HERE,CONFIG,sha,dump

def package():
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[HERE/'CONFIG.json',HERE/'requirements.txt',HERE/'cloud.sh',HERE/'LOCAL_AUDIT.json',HERE/'CPU_AUDIT_pilot.json',HERE/'OUTPUT_LAYOUT_AUDIT.json']+list((HERE/'vendor').glob('*.py'))
    files+=list((HERE/'data/pilot').rglob('*.json'))+list((HERE/'data/pilot').rglob('*.jsonl'))
    files=sorted({p for p in files if p.name!='TOKEN_AUDIT.json'})
    assert all(p.is_file() for p in files)
    names=[p.relative_to(HERE).as_posix() for p in files]
    assert not any(n.startswith(('runs/','analysis/','logs/','upload/')) or n.endswith('TOKEN_AUDIT.json') or n=='MODEL_IDENTITY.json' for n in names)
    dump(HERE/'PACKAGE_MANIFEST.json',dict(version=CONFIG['version'],files={n:sha(HERE/n) for n in names},meaning='Reviewed independent pilot; mutable tokenizer audits/model identity/runs/analysis/logs/upload excluded.'))
    out=HERE/'upload';out.mkdir(exist_ok=True);archive=out/'e4_v4_pilot.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[HERE/'PACKAGE_MANIFEST.json']:z.write(p,'e4_v4/'+p.relative_to(HERE).as_posix())
    dump(out/'ARCHIVE.json',dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive)))
    print(archive)
if __name__=='__main__':package()
