#!/usr/bin/env python3
import pexpect
import os

def scp_files_from_kria(kria_user, kria_ip, kria_source_dir, files, local_dest, password):
    
    # Criar o diretório de destino se não existir
    os.makedirs(local_dest, exist_ok=True)
    
    for file in files:
        remote_file = f"{kria_user}@{kria_ip}:{kria_source_dir}/{file}"
        scp_cmd = f"scp {remote_file} {local_dest}/"
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
        print(f"File {file} transfer complete")

if __name__ == "__main__":
    
    LOCAL_FOLDER = "/workspace/claudino/projects/Test_4"
    KRIA_USER = "root"
    KRIA_IP = "192.168.0.100"
    KRIA_SOURCE_DIR = "/home/root/Test_4"
    FILES = [
        "test_C3_kv260_power_results_cpp.csv",
        "test_C3_kv260_results_cpp.csv"
    ]
    LOCAL_DEST = "."  # Diretório atual
    PASSWORD = "root"

    scp_files_from_kria(KRIA_USER, KRIA_IP, KRIA_SOURCE_DIR, FILES, LOCAL_FOLDER, PASSWORD)
    print("All files transferred successfully")