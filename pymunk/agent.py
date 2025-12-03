import os
import subprocess
import time

import socket
import struct
import json

from util import *
from trajectory import *

VIS_HOST = '127.0.0.1'  # Server address
VIS_PORT = 56789        # Server port
CODE_HOST = 'SERV_ADDR'  # Server address
CODE_PORT = 10586        # Server port

# BASE_DIR = r"F:/Projects/agent_synthetic/pymunk/generations/"+time.strftime("%Y%m%d-%H%M%S")
BASE_DIR = r"F:/Projects/agent_synthetic/pymunk/generations/abl-gpt"
PYTHON_DIR = r"f:/Projects/agent_synthetic/env_agent/python.exe"

ITER_LOOP = 2
PROMPT_LOC = r"F:/Projects/agent_synthetic/dataset/prompts"

def load_prompts(prompt_file):
    with open(prompt_file, 'r') as f:
        prompts = f.read()

    return prompts.strip()
    
# returns is_success, video_path (or error message), code_path
def render_videos(i, scene):
    tmpdir = os.path.join(BASE_DIR, scene)
    os.makedirs(tmpdir, exist_ok=True)  # Ensure the temp folder exists
    script_path = os.path.join(tmpdir, f"scene_{i:04d}.py")
    out_file = os.path.join(tmpdir, f"output_{i:04d}.mp4")
    try:
        # Execute Manim with the correct path
        result = subprocess.run(
            [PYTHON_DIR, "spawn_pygame.py", "--run_time", "10", "--script_path", script_path, "--out_file", out_file], 
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and os.path.exists(out_file):
            return True, out_file, script_path
        else:
            print(f"Execution failed: {result.stderr}, {result.stdout}")
            err_msg = result.stderr
            return False, err_msg, script_path
            # deal with failed code, ask gemini to fix it
    except Exception as e:
            print(f"Error during execution: {str(e)}")

# returns the path to the updated script
def update_scripts(tmpdir, new_code, i):
    new_code = new_code.split("```python")[-1].split("```")[0] if len(new_code.split("```")) > 1 else new_code
    os.makedirs(tmpdir, exist_ok=True)  # Ensure the temp folder exists
    script_path = os.path.join(tmpdir, f"scene_{i:04d}.py")
    with open(script_path, 'w') as f:
        f.write(new_code)
    return script_path

def vis_feedback(video_path, og_prompt):
    prompt = "Please describe what is not aligned with physics rules in the video."
    # we can try to add reference prompt later
    prompt2 = f"Please describe how the video does not align with the user's original intent: {og_prompt}."
    req = json.dumps({'purpose':'feedback', 
                      'feedback_prompt': prompt, 
                      'original_prompt': prompt2,
                      'video_path': video_path}).encode('utf-8')
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
        if resp.get('visual_feedback_physics') is not None and resp.get('visual_feedback_intent') is not None:
            text = f"Physics Alignment:{resp.get('visual_feedback_physics')[0]}\nUser Intent Alignment:{resp.get('visual_feedback_intent')[0]}"
            return text
        else:
            return None


if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)
    # prompts = load_prompts()
    for prompt_loc_txt in os.listdir(PROMPT_LOC):
        prompts = load_prompts(os.path.join(PROMPT_LOC, prompt_loc_txt))
        scene = prompt_loc_txt.split(".txt")[0]
        print(scene)
        work_dir = os.path.join(BASE_DIR, scene)
        if os.path.exists(work_dir) and os.path.exists(os.path.join(work_dir, "output_{:04d}.mp4".format(ITER_LOOP-1))):
            print(f"Skipping existing scene: {scene}")
            continue
        # construct feedback loop
        code = send_code_request(CODE_HOST, CODE_PORT, prompts, None, None, intent='init_code', llm='gpt', client='pymunk')
        update_scripts(work_dir, code['code'], 0)
        constructed_json = code['json']
        os.makedirs(work_dir, exist_ok=True)  # Ensure the temp folder exists
        with open(os.path.join(work_dir, "prompt_json.json"), 'w') as f:
            f.write(constructed_json)
        for iter in range(0, ITER_LOOP):
            print(f"--- Iteration {iter} ---")
            is_success, video_path, script_path = render_videos(iter, scene)
            vis_result = None
            if is_success:
                print(f"Video rendered at: {video_path}")
                vis_result = vis_feedback(video_path, prompts)
            while not is_success or vis_result is None:
                print("Rendering failed, requesting code fix...")
                if not is_success:
                    error_msg = video_path
                else:
                    error_msg = "Program exited unexpectedly with no error during rendering, no video generated."
                print(f"Error message: {error_msg}, error msg ended.")
                code = send_code_request(CODE_HOST, CODE_PORT, prompts, error_msg, code['code'], intent='fix_code', llm='gpt', client='pymunk')
                update_scripts(work_dir, code['code'], iter)
                is_success, video_path, script_path = render_videos(iter, scene)
                print(f"Video rendered at: {video_path}")
                if ITER_LOOP > 1:
                    vis_result = vis_feedback(video_path, prompts)
                else:
                    vis_result = "No feedback needed for single iteration."
            # based on feedback, update code
            if ITER_LOOP > 1 and iter < ITER_LOOP - 1:
                suggestions = send_code_request(CODE_HOST, CODE_PORT, None, vis_result, None, intent="summarize_feedback", llm='gpt', client='pymunk')
                print(f"Feedback summary: {suggestions['summary']}")
                code = send_code_request(CODE_HOST, CODE_PORT, prompts+", initial conditions as json: ```json{constructed_json}```", suggestions['summary'], code['code'], intent='update_code_feedback', llm='gpt', client='pymunk')
                update_scripts(work_dir, code['code'], iter+1)
        # img_location = work_dir + "/out_frames/"
        # os.makedirs(img_location, exist_ok=True)
        # extract_frames_from_video(video_path, img_location, fps=24)
        # obj_pts, w, h, obj_names = locate_objs_in_frame(img_location, start_frame=0, host=CODE_HOST, port=CODE_PORT, prompt_json=constructed_json, client='pymunk')
        # video_tensor = cvt_video_tensor(img_location, w, h)
        # tracks, visibility = track_objs_in_video(video_tensor, obj_pts, obj_names)
        # save_track_json(tracks, "pred", work_dir+"/pred_tracked.json")
        # show_in_video(work_dir, video_tensor, "tracked", tracks, visibility)

                