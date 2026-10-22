"""Cloud SSH helper v2.

Differences from zigzag_v3/execution/cloud_remote.py:
  * host / port / user come from CLOUD_ENDPOINT.json next to this file (AutoDL
    style instances get a new host:port after every power cycle);
  * the password is read from a file OUTSIDE the synced project tree, named by
    env CLOUD_PASSWORD_FILE (default ~/.zigzag_cloud_pw); it is never written
    to any artifact;
  * host keys are trusted on first use and pinned in known_hosts here; pass
    --reset-hostkey after renting a new instance.
"""
from __future__ import annotations
import argparse
import json
import os
import socket
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import paramiko

HERE = Path(__file__).resolve().parent
ENDPOINT = HERE / 'CLOUD_ENDPOINT.json'
KNOWN = HERE / 'known_hosts'


def endpoint() -> dict:
    e = json.loads(ENDPOINT.read_text(encoding='utf8'))
    assert e.get('host') and e.get('port'), 'CLOUD_ENDPOINT.json needs host and port'
    return e


def password() -> str:
    f = Path(os.environ.get('CLOUD_PASSWORD_FILE', Path.home() / '.zigzag_cloud_pw')).expanduser()
    if 'CLOUD_EXPERIMENT_PASSWORD' in os.environ:
        return os.environ['CLOUD_EXPERIMENT_PASSWORD']
    assert f.exists(), f'password file missing: {f} (write the SSH password there, one line)'
    return f.read_text(encoding='utf8').strip()


def resolve(host: str) -> str:
    try:
        ip = socket.gethostbyname(host)
        if not ip.startswith('198.18.'):
            return ip
    except OSError:
        pass
    q = urllib.parse.urlencode({'name': host, 'type': 'A'})
    last = None
    for base in ('https://dns.google/resolve?', 'https://1.1.1.1/dns-query?'):
        try:
            req = urllib.request.Request(base + q, headers={'Accept': 'application/dns-json'})
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.load(r)
            ips = [a['data'] for a in d.get('Answer', []) if a['type'] == 1 and not a['data'].startswith('198.18.')]
            if ips:
                return ips[0]
        except Exception as e:      # noqa: BLE001
            last = e
    raise RuntimeError(f'DoH resolution failed for {host}: {last}')


def connect(reset_hostkey: bool = False) -> paramiko.SSHClient:
    e = endpoint()
    ip = resolve(e['host'])
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(25)
    if e.get('bind_source_ip'):
        sock.bind((e['bind_source_ip'], 0))
    sock.connect((ip, int(e['port'])))
    c = paramiko.SSHClient()
    if reset_hostkey and KNOWN.exists():
        KNOWN.unlink()
    if KNOWN.exists():
        c.load_host_keys(str(KNOWN))
        c.set_missing_host_key_policy(paramiko.RejectPolicy())
    else:
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(e['host'], port=int(e['port']), username=e.get('user', 'root'), password=password(), sock=sock,
              allow_agent=False, look_for_keys=False, timeout=25, banner_timeout=25, auth_timeout=25)
    if not KNOWN.exists():
        c.save_host_keys(str(KNOWN))
    return c


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 600) -> tuple[int, str, str]:
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf8', errors='replace')
    err = e.read().decode('utf8', errors='replace')
    return o.channel.recv_exit_status(), out, err


CHECK = r'''set -e
echo "== host =="; hostname; uptime
echo "== gpu =="; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv || true
echo "== disk =="; df -h /root/autodl-tmp 2>/dev/null || df -h /root
echo "== python =="; /root/miniconda3/bin/python -c "import torch,transformers,peft,safetensors;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'transformers',transformers.__version__,'peft',peft.__version__)" 2>&1 || echo "PYTHON_ENV_MISSING"
echo "== models =="; ls -d /root/autodl-tmp/zigzag_v3/models/qwen15 /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct 2>/dev/null || echo "NO_MODEL_DIR"
echo "== old projects =="; ls -d /root/autodl-tmp/zigzag_* 2>/dev/null || echo "NONE"
echo "== running python =="; ps -eo pid,etimes,args | grep -E "python" | grep -v grep || echo "NONE"
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--command')
    p.add_argument('--script')
    p.add_argument('--put', nargs=2, metavar=('LOCAL', 'REMOTE'))
    p.add_argument('--get', nargs=2, metavar=('REMOTE', 'LOCAL'))
    p.add_argument('--check', action='store_true', help='print GPU/disk/python/model status')
    p.add_argument('--reset-hostkey', action='store_true')
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    c = connect(a.reset_hostkey)
    try:
        if a.put or a.get:
            with c.open_sftp() as s:
                if a.put:
                    s.put(*a.put)
                else:
                    s.get(*a.get)
        cmd = a.command or (Path(a.script).read_text(encoding='utf-8-sig') if a.script else None) or (CHECK if a.check else None)
        if cmd:
            code, out, err = run(c, cmd, a.timeout)
            print(out, end='')
            print(err, end='', file=sys.stderr)
            sys.exit(code)
    finally:
        c.close()


if __name__ == '__main__':
    main()
