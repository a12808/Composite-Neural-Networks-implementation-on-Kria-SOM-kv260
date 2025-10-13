#!/usr/bin/env python3
import pexpect
import os
import sys


def scp_folder_to_kria(local_folder, kria_user, kria_ip, kria_dest, password):
    
    scp_cmd = f"scp -r {local_folder} {kria_user}@{kria_ip}:{kria_dest}"
    print(f"[CMD]: {scp_cmd}")
    
    child = pexpect.spawn(scp_cmd, timeout=300)
    child.logfile = sys.stdout.buffer   # real time
    idx = child.expect(["password:", "(yes/no)"], timeout=30)
    
    if idx == 1:
        child.sendline("yes")
        child.expect("password:", timeout=30)

    child.sendline(password)
    child.expect(pexpect.EOF)
    
    # real time
    #print(child.before.decode(errors="ignore"))
    
    print("Transfer complete \n")
    
def compile_cpp_on_kria(kria_user, kria_ip, kria_dest, cpp_file, output_binary, password):
    
    compile_cmd = (
        f"g++ {os.path.join(kria_dest, cpp_file)} -o {os.path.join(kria_dest, output_binary)} "
        f"`pkg-config --cflags --libs opencv4` -lvart-runner -lxir"
    )
    print(f"Compiling on Kria: {compile_cmd}")
    
    child = pexpect.spawn(f"ssh {kria_user}@{kria_ip}", timeout=300)
    #child.logfile = sys.stdout.buffer  # real time
    idx = child.expect(["password:", "(yes/no)"], timeout=30)
    if idx == 1:
        child.sendline("yes")
        child.expect("password:", timeout=30)
        
    child.sendline(password)
    
    child.expect(r'#\s*$')  # espera prompt do root
    child.sendline(compile_cmd)
    child.expect(r'#\s*$') 
    print(child.before.decode(errors="ignore"))
    child.sendline("exit")
    child.expect(pexpect.EOF)
    
    print("Compilation complete \n")

def run_binary_on_kria(kria_user, kria_ip, kria_dest, binary, args, password):
    
    run_cmd = f"{os.path.join(kria_dest, binary)} {args}"
    print(f"[CMD] {run_cmd}")
    
    child = pexpect.spawn(f"ssh {kria_user}@{kria_ip}", timeout=300)
    child.logfile = sys.stdout.buffer  # Mostra saída em tempo real
    idx = child.expect(["password:", "(yes/no)"], timeout=30)
    
    if idx == 1:
        child.sendline("yes")
        child.expect("password:", timeout=30)
    
    child.sendline(password)
    
    child.expect(r'#\s*$')  # espera prompt do root
    child.sendline(run_cmd)
    child.expect(r'#\s*$')  # espera prompt do root
    print(child.before.decode(errors="ignore"))
    child.sendline("exit")
    child.expect(pexpect.EOF)
    
    print("Execution complete \n")
    
    
if __name__ == "__main__":
    
    LOCAL_FOLDER = "/workspace/claudino/projects/Test_3/Test_3"
    KRIA_USER = "root"
    KRIA_IP = "192.168.0.100"
    KRIA_DEST = "/home/root/"
    PASSWORD = "root"

    CPP_FILE = "Test_3/test_3_kv260_run.cc"
    OUTPUT_BINARY = "Test_3/test_3_kv260_run"
    #BINARY_ARGS = f"{KRIA_DEST}resnet_fusion.xmodel {KRIA_DEST}labels.txt {KRIA_DEST}val_labels.txt {KRIA_DEST}dataset {KRIA_DEST}power_log.csv"

    RUN_AFTER_COMPILE = False

    # Envia a pasta para o Kria
    scp_folder_to_kria(LOCAL_FOLDER, KRIA_USER, KRIA_IP, KRIA_DEST, PASSWORD)

    # Compila o C++ no Kria
    compile_cpp_on_kria(KRIA_USER, KRIA_IP, KRIA_DEST, CPP_FILE, OUTPUT_BINARY, PASSWORD)

    # Executa opcionalmente o binário
    #if RUN_AFTER_COMPILE:
        #run_binary_on_kria(KRIA_USER, KRIA_IP, KRIA_DEST, OUTPUT_BINARY, BINARY_ARGS, PASSWORD)