"""SSH helper for the LAN RTX 3090 workstation. Socket bound to the physical NIC so traffic avoids the TUN adapter.
Password is read from ~/.zigzag_lan_pw (outside the project tree).

  python lan.py --command "..."            run a shell command
  python lan.py --script file.sh            run a local script's text remotely (bash -s)
  python lan.py --put LOCAL REMOTE          sftp upload
  python lan.py --get REMOTE LOCAL          sftp download
"""
from __future__ import annotations
import argparse
import os
import socket
import sys
from pathlib import Path

import paramiko

SOURCE_IP = "<local NIC address>"   # bind the socket to the physical interface (a TUN adapter was present)
TARGET_IP = "<workstation address>"
PORT = 22
USER = "<user>"
PW_FILE = Path.home() / ".zigzag_lan_pw"


def connect() -> paramiko.SSHClient:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(20)
    sock.bind((SOURCE_IP, 0))
    sock.connect((TARGET_IP, PORT))
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(TARGET_IP, port=PORT, username=USER, password=PW_FILE.read_text(encoding="utf8").strip(), sock=sock,
              timeout=20, banner_timeout=20, auth_timeout=20, allow_agent=False, look_for_keys=False)
    return c


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 600):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode("utf8", errors="replace")
    err = e.read().decode("utf8", errors="replace")
    return o.channel.recv_exit_status(), out, err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--command")
    p.add_argument("--script")
    p.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    p.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    p.add_argument("--timeout", type=int, default=600)
    a = p.parse_args()
    c = connect()
    try:
        if a.put or a.get:
            with c.open_sftp() as s:
                if a.put:
                    s.put(*a.put)
                    print("put ok", a.put[1])
                else:
                    s.get(*a.get)
                    print("get ok", a.get[1])
        cmd = a.command
        if a.script:
            cmd = "bash -s <<'ZZEOF'\n" + Path(a.script).read_text(encoding="utf-8-sig") + "\nZZEOF"
        if cmd:
            code, out, err = run(c, cmd, a.timeout)
            print(out, end="")
            print(err, end="", file=sys.stderr)
            sys.exit(code)
    finally:
        c.close()


if __name__ == "__main__":
    main()
