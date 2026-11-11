"""Seal completed pilot and freeze unchanged confirmation protocol."""
import argparse,json
from generate import HERE,CONFIG,sha,dump
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot-reviewed',action='store_true',help='Affirm manual feasibility review documented in PILOT_REVIEW.md');args=parser.parse_args()
    assert args.pilot_reviewed,'Manual pilot feasibility review required; see PROTOCOL.md'
    review=HERE/'PILOT_REVIEW.md';assert review.exists(),'Write actual pilot findings and go/no-go decision in PILOT_REVIEW.md'
    assert not (HERE/'CONFIRM_LOCK.json').exists(),'Lock is immutable'
    paths=[HERE/'runs'/'pilot'/f'world{w}_{arm}'/'COMPLETE.json' for w in CONFIG['pilot'] for arm in ['edge','joint']]
    assert all(p.exists() for p in paths),'All four pilot arms must complete'
    selection=HERE/'analysis'/'pilot'/'SELECTION.json'
    assert selection.exists(),'Run analyze.py --phase pilot first'
    feasibility=HERE/'analysis/pilot/FEASIBILITY.json'
    assert json.loads(feasibility.read_text())['passed'],'Pilot feasibility gate failed; no confirmation freeze'
    files=list(HERE.glob('*.py'))+list((HERE/'vendor').glob('*.py'))+[HERE/'PROTOCOL.md',HERE/'CONFIG.json',HERE/'cloud.sh',HERE/'requirements.txt',selection,review,feasibility]
    dump(HERE/'CONFIRM_LOCK.json',dict(version=CONFIG['version'],files={str(p.relative_to(HERE)):sha(p) for p in files},
         pilot_complete={str(p.relative_to(HERE)):sha(p) for p in paths},note='Any protocol change requires new version/directory and fresh confirmation worlds.'))
