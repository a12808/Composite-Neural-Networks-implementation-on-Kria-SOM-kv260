#!/usr/bin/env python3
import pexpect

def scp_folder_to_kria(local_folder, kria_user, kria_ip, kria_dest, password):
    scp_cmd = f"scp -r {local_folder} {kria_user}@{kria_ip}:{kria_dest}"
    print(f"Executing: {scp_cmd}")
    
    child = pexpect.spawn(scp_cmd, timeout=300)
    # Esperar pelo prompt de password ou confirmação de host
    idx = child.expect(["password:", "(yes/no)"], timeout=30)
    if idx == 1:
        child.sendline("yes")
        child.expect("password:", timeout=30)

    child.sendline(password)
    child.expect(pexpect.EOF)
    print(child.before.decode(errors="ignore"))
    print("Transfer complete")

if __name__ == "__main__":
    LOCAL_FOLDER = "/workspace/claudino/projects/Test_2/Test_2"
    KRIA_USER = "root"
    KRIA_IP = "192.168.0.100"
    KRIA_DEST = "/home/root/"
    PASSWORD = "root"

    scp_folder_to_kria(LOCAL_FOLDER, KRIA_USER, KRIA_IP, KRIA_DEST, PASSWORD)