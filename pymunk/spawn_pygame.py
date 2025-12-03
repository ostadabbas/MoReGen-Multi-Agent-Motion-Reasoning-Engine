import threading
import subprocess
import sys
import time
import argparse

def run_script(python_exe, script_path, run_time, out_file, result):
    try_proc = subprocess.Popen([python_exe, script_path])
    time.sleep(2) 
    if try_proc.poll() is not None:
        stdout, stderr = try_proc.communicate()
        print("Process terminated early.")
        print("Error output:", stderr.decode())
        result.append(stderr.decode())
    else:
        print("Process is still running. Likely no immediate error.")
        try_proc.terminate()
        proc = subprocess.Popen([python_exe, script_path])
        print(f"Spawned process PID: {proc.pid}")
        capture_script = r".\screen_record.py"
        duration = run_time + 2
        subprocess.Popen([
            sys.executable,
            capture_script,
            "--pid", str(proc.pid),
            "--output", out_file,
            "--duration", str(duration)
        ])
        time.sleep(run_time)
        proc.terminate()

if __name__ == "__main__":
    # Example usage: specify your python.exe and script.py paths
    parser = argparse.ArgumentParser(description="Spawn a script and record its window.")
    parser.add_argument("--run_time", type=int, default=3, help="Time to run the script (seconds)")
    parser.add_argument("--script_path", type=str, default=r".\demo.py", help="Path to the script to run")
    parser.add_argument("--out_file", type=str, default="output.mp4", help="Output file for recording")
    
    args = parser.parse_args()
    python_exe = r"f:\Projects\agent_synthetic\env_agent\python.exe"
    script_path = args.script_path
    output_file = args.out_file
    result = []
    thread = threading.Thread(target=run_script, args=(python_exe, script_path, args.run_time, output_file, result))
    thread.start()
    thread.join()
    if len(result) > 0:
        print(f"Error: {result[0]}", file=sys.stderr)
        sys.exit(1)
