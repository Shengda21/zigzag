"""Rebuild constituent selection scores from raw tokens and verify lineage."""
import argparse,hashlib,json,os
from collections import Counter,defaultdict
from pathlib import Path


def audit(root,standalone=False):
    from transformers import AutoTokenizer
    root=Path(root).resolve();tok=AutoTokenizer.from_pretrained(os.environ.get('SIMODS_BASE_MODEL','/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct'),local_files_only=True);hashes={};reports={};partial=[]
    def digest(path):
        path=Path(path)
        if path not in hashes:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for z in iter(lambda:f.read(8*1024*1024),b''):h.update(z)
            hashes[path]=h.hexdigest()
        return hashes[path]
    for folder in sorted(root.glob('component_*')):
        if not folder.is_dir():continue
        if not(folder/'analysis.json').exists():partial.append(folder.name);continue
        meta=json.loads((folder/'analysis.json').read_text());bits=folder.name.split('_');exchange=bits[1].isdigit()
        if exchange:phase,chunk=map(int,bits[1:]);bankfile=root/f'component_{phase}.jsonl';adapter=root/f'results_exchange_{phase}_{chunk}/adapter';field='new_strict';meanfield='mean_new_rule_strict';minfield='minimum_new_rule_type';floor=.99
        else:role=bits[1];chunk=int(bits[2]);bankfile=root/f'test_{role}.jsonl';adapter=root/f'results_{role}_{chunk}/adapter';field='strict';meanfield='mean_strict';minfield='minimum_type_strict';floor=.999 if role=='primitive' else .99
        bank=[json.loads(l) for l in bankfile.read_text().splitlines()];counts=defaultdict(Counter);seen=set();cache={};tokens=0
        with (folder/'predictions.jsonl').open() as source:
            for line in source:
                p=json.loads(line);i=p['source_index'];r=bank[i];assert i not in seen and p['id']==r['id'];seen.add(i);raw=tuple(p['raw_generated_ids'])
                if raw not in cache:
                    stop=next((j for j,v in enumerate(raw) if v in (151645,151643)),None);cache[raw]=(tok.decode(raw if stop is None else raw[:stop],skip_special_tokens=False,clean_up_tokenization_spaces=False),None if stop is None else raw[stop],len(raw) if stop is None else stop+1)
                text,eos,n=cache[raw];tokens+=n;assert text==p['prediction'] and eos==p['terminator_id'];c=counts[r['type']];c['n']+=1;c[field]+=int(text==r['answer'] and eos==151645)
                if exchange:c['old_strict']+=int(text==r['old_answer'] and eos==151645)
        assert len(seen)==len(bank) and {k:dict(v) for k,v in counts.items()}==meta['types'];mean=sum(c[field] for c in counts.values())/len(bank);minimum=min(c[field]/c['n'] for c in counts.values());assert mean==meta[meanfield] and minimum==meta[minfield];assert (mean>=.999 and minimum>=floor)==meta['component_gate_pass'];weight=adapter/'adapter_model.safetensors';assert digest(weight)==meta['weights_sha256'];runtime=json.loads((adapter/'runtime.json').read_text());assert runtime['weights_sha256']==digest(weight);args=runtime['arguments'];parent=args.get('source')
        if parent:
            p=Path(parent).resolve()
            if standalone:assert p.is_relative_to(root)
            assert digest(p/'adapter_model.safetensors')==runtime['source_sha256']
        else:
            assert not exchange and role in ('primitive','identity') and chunk==1 and runtime['source_sha256'] is None
        reports[folder.name]={'programs':len(bank),'generated_tokens_through_first_eos':tokens,'mean':mean,'minimum_type':minimum,'component_gate_pass':meta['component_gate_pass'],'weights_and_parent_verified':True}
    result={'status':'PASS_COMPLETED_COMPONENT_RAW_OUTPUTS_AND_LINEAGE','components':reports,'partial_component_folders':partial,'standalone_case_sources_confined_to_case':standalone,'scope':'Computational integrity of constituent selection; a passed audit is not a passed scientific case.'};(root/'COMPONENT_RAW_AUDIT.json').write_text(json.dumps(result,indent=2));return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--standalone-case',action='store_true');a=ap.parse_args();r=audit(a.study,a.standalone_case);print(json.dumps({'status':r['status'],'components':len(r['components']),'partial':r['partial_component_folders']}),flush=True)


if __name__=='__main__':main()
