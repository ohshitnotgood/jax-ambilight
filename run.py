import threading
from subprocess import run
import os

def _run_backend():
    os.system("cd backend && python main.py -v")
    
    
def _run_frontend():
    os.system("export WEBKIT_DISABLE_DMABUF_RENDERER=1 && cd desktop && pnpm tauri dev")

def main():
    t1 = threading.Thread(target=_run_backend)
    t2 = threading.Thread(target=_run_frontend)
    
    t1.start()
    t2.start()

if __name__ == "__main__":
    main()