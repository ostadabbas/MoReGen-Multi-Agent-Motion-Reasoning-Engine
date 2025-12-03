import os
import subprocess
import time

import socket
import struct
import json

VIS_HOST = '127.0.0.1'  # Server address
VIS_PORT = 56789        # Server port
CODE_HOST = 'zzzura.duckdns.org'  # Server address
CODE_PORT = 10586        # Server port

MANIM_EXECUTABLE = r"F:/Projects/agent_synthetic/manim/manimations/.venv/Scripts/manim.exe"
BASE_DIR = r"F:/Projects/agent_synthetic/manim/generations/"+time.strftime("%Y%m%d-%H%M%S")

ITER_LOOP = 10
PROMPT_LOC = r"F:/Projects/agent_synthetic/manim/prompt.txt"
PROMPT_JSON = r"F:/Projects/agent_synthetic/manim/prompt_json.json"

def load_prompts():
    with open(PROMPT_LOC, 'r') as f:
        prompts = f.read()
    
    with open(PROMPT_JSON, 'r') as f:
        prompt_json = json.load(f)

    return prompts.strip(), prompt_json

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# init_code: 'gemini', 'qwen' or None
def send_code_request(prompt, prompt_json, code, fix_code, init_code):
    req = {
        'prompt_a': prompt,
        'prompt_b': prompt_json,
        'video_path': None,
        'code': code,
        'fix_code': fix_code,
        'init_code': init_code
    }
    data = json.dumps(req).encode('utf-8')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((CODE_HOST, CODE_PORT))
        # Send the length of the data first
        s.sendall(struct.pack('>I', len(data)))
        # Then send the actual data
        s.sendall(data)
        
        # Receive the length of the incoming message
        raw_msglen = recvall(s, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Receive the actual message
        response = recvall(s, msglen)
        return json.loads(response.decode('utf-8'))
    
# returns is_success, video_path (or error message), code_path
def render_videos(i):
    tmpdir = BASE_DIR  
    os.makedirs(tmpdir, exist_ok=True)  # Ensure the temp folder exists
    script_path = os.path.join(tmpdir, f"scene_{iter:04d}.py")
    try:
        # Execute Manim with the correct path
        result = subprocess.run(
            [MANIM_EXECUTABLE, script_path], #MANIM_PATH "/Users/[Your_username]/anaconda3/envs/manim2/Scripts/manim.exe"
            capture_output=True,
            text=True,
            cwd=tmpdir
        )
        if result.returncode == 0:
            out = result.stdout
            file_loc = out.split('File ready at')[-1].split('INFO')[0].split('[')[0].replace('\n','').replace(' ','').replace('\'', "").strip()
            # Extract file_loc from Manim output
            old_file_loc = file_loc
            prior = file_loc.split('\\')[:-1]
            file_loc = file_loc.split('\\')[-1].split(".")[0] + f"_{iter:04d}.mp4"
            prior.append(file_loc)
            file_loc = '\\'.join(prior)
            os.rename(old_file_loc, file_loc)
            print(f"Extracted file location: {file_loc}")
            return True, file_loc.replace('\\', '/'), script_path
        else:
            print(f"Execution failed: {result.stderr}")
            err_msg = result.stderr
            return False, err_msg, script_path
            # deal with failed code, ask gemini to fix it
    except Exception as e:
            print(f"Error during execution: {str(e)}")

# returns the path to the updated script
def update_scripts(new_code, i):
    new_code = new_code.split("```python")[-1].split("```")[0] if len(new_code.split("```")) > 1 else new_code
    tmpdir = BASE_DIR
    os.makedirs(tmpdir, exist_ok=True)  # Ensure the temp folder exists
    script_path = os.path.join(tmpdir, f"scene_{i:04d}.py")
    with open(script_path, 'w') as f:
        f.write(new_code)
    return script_path

def vis_feedback(video_path):
    prompt = "Please describe what is not aligned with physics rules in the video."
    # we can try to add reference prompt later
    req = json.dumps({'purpose':'feedback', 'prompt': prompt, 'video_path': video_path}).encode('utf-8')
    msg = struct.pack('>I', len(req)) + req
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((VIS_HOST, VIS_PORT))
        s.sendall(msg)
        # Receive response length
        raw_msglen = recvall(s, 4)
        if not raw_msglen:
            print('No response from server')
            return
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Receive the actual data
        data = recvall(s, msglen)
        if not data:
            print('No data received')
            return
        # Parse JSON
        resp = json.loads(data.decode('utf-8'))
        return resp.get('visual_feedback')


if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)
    prompts, prompt_json = load_prompts()
    prompt_json = json.dumps(prompt_json)
    prompt_json = "```json\n" + prompt_json + "\n```"
    # construct feedback loop
    code = send_code_request(prompts, None, None, fix_code=False, init_code='gemini')
    update_scripts(code['code'], 0)
    for iter in range(0, ITER_LOOP):
        print(f"--- Iteration {iter} ---")
        is_success, video_path, script_path = render_videos(iter)
        while not is_success:
            print("Rendering failed, requesting code fix...")
            code = send_code_request(prompts, video_path, code['code'], fix_code=True, init_code=None)
            update_scripts(code['code'], iter)
            is_success, video_path, script_path = render_videos(iter)
        print(f"Video rendered at: {video_path}")
        vis_result = vis_feedback(video_path)
        print(f"Visual feedback: {vis_result[0]}")
        # based on feedback, update code
        code = send_code_request(prompts, vis_result[0], code['code'], fix_code=False, init_code=None)
        update_scripts(code['code'], iter+1)