"""Model-blind exact tokenizer audit of the paired continuation budgets."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;MODEL=Path('/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')


def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);result={}
    for world in ('r1603','r4004'):
        data={arm:[json.loads(l) for l in (HERE/world/f'train_{arm}.jsonl').read_text().splitlines()] for arm in ('two','three')};lengths={}
        for arm,rows in data.items():
            values=[]
            for row in rows:
                user={'role':'user','content':row['prompt']};assistant={'role':'assistant','content':row['answer']};prefix=tok.encode(tok.apply_chat_template([user],tokenize=False,add_generation_prompt=True),add_special_tokens=False);full=tok.encode(tok.apply_chat_template([user,assistant],tokenize=False),add_special_tokens=False);assert full[:len(prefix)]==prefix;values.append((len(prefix),len(full)-len(prefix)))
            lengths[arm]=values
        assert lengths['two']==lengths['three']
        result[world]={'matched_examples':4096,'prompt_tokens':sum(v[0] for v in lengths['two']),'answer_tokens':sum(v[1] for v in lengths['two'])}
    (HERE/'TOKEN_AUDIT.json').write_text(json.dumps({'status':'PASS_EXACT_PAIRED_TOKEN_LENGTHS','worlds':result},indent=2));print(json.dumps(result))


if __name__=='__main__':main()
