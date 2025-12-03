
# Socket server imports
import socket
import threading
import struct
import json
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from google import genai

import os
from math import ceil
import torch
import subprocess
import transformers
from openai import OpenAI


HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 10586      # Arbitrary non-privileged port

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

Gemini_key = "YOUR_GEMINI_KEY"
OpenAI_key = "YOUR_OPENAI_KEY"
os.environ["GOOGLE_API_KEY"] = Gemini_key
os.environ["OPENAI_API_KEY"] = OpenAI_key
os.environ["TOKENIZERS_PARALLELISM"] = "false"
code_client = genai.Client()
gpt_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

print(transformers.__version__)
# model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

client_dict = {
            "manim": "manim library (community edition v0.19.0 only! Please do not add any text or math equations!)",
            "pymunk": "pymunk library"
        }

def get_json_from_response(response):
    ret = response.split("```json")
    if len(ret) > 1:
        try:
            ret = ret[1].split("```")[0]
            return json.loads(ret.strip())
        except json.JSONDecodeError:
            return {}
    return {}

def get_frame_list(output_path, fraction=0.0125):
    # Get all jpg files in the directory
    all_frames = sorted([f for f in os.listdir(output_path) if f.endswith('.jpg')])

    total_frames = len(all_frames)
    frames_to_keep = ceil(total_frames * fraction)

    # Calculate the step size to evenly distribute the selected frames
    step = total_frames // frames_to_keep

    # Select the frames
    selected_frames = all_frames[::step][:frames_to_keep]

    # Create the full paths for the selected frames
    frame_paths = [f"{os.path.join(output_path, frame)}" for frame in selected_frames]

    return frame_paths

def query_video(prompt, use_frames=True, frames_path="/home/qwen2_vl/content/frames", video_path=None):
    if use_frames:
        # Get the frames
        selected_frames = get_frame_list(output_path)

        # Create messages structure for frames
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": selected_frames,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        # Create messages structure for the entire video
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{video_path}",
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    print(f"Using {'frames' if use_frames else 'entire video'} for inference.")

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    with torch.no_grad():  # Use no_grad to save memory during inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Trim the generated output to remove the input prompt
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    torch.cuda.empty_cache()
    return output_text

def query_vis_code(vis_feedback, objective, code, client, llm):
    prompt = "Given the following objective: " + objective + ", feedback and guidance: "+\
        vis_feedback + " and the following code: " + code + ", please refine the code so that the generated video \
            aligns more with the physics rules and user intent. Please provide the full revised code only."
    if llm == 'gemini':
        response = query_code_gemini(prompt)
    elif llm == 'gpt':
        response = query_code_gpt(prompt, "gpt-5", client)
    elif llm == 'qwen':
        response = query_code_qwen(prompt, client)
    else:
        raise ValueError(f"Unsupported llm: {llm}")
    return response

def query_error_code(issue, code, client, llm):
    prompt = "This code returns the following issue when executed: ```" + issue + "```\nPlease fix this code: ```python\n"+code+\
    "```\n Please provide the full revised code only."
    if llm == 'gemini':
        response = query_code_gemini(prompt)
    elif llm == 'gpt':
        response = query_code_gpt(prompt, "gpt-5", client)
    elif llm == 'qwen':
        response = query_code_qwen(prompt, client)
    else:
        raise ValueError(f"Unsupported llm: {llm}")
    return response

def convert_mkv_to_mp4(input_path, output_path):
    """
    Converts an MKV video file to MP4 format using ffmpeg.
    Args:
        input_path (str): Path to the input MKV file.
        output_path (str): Path to save the output MP4 file.
    """
    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "copy",
        "-c:a", "copy",
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Conversion successful: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr.decode()}")

def query_code_qwen(prompt, client):
    with open (f"{client}_modules.txt", "r") as f:
        modules = f.read()
    messages = [
        {"role": "system", "content": f"You are a helpful code assistant that generates python code to create physics simulations and animations using the {client_dict[client]}. \
        Remember that these are correct syntax: ```{modules}```. By generating code, you help users create physics-accurate video simulations from text prompt that follows real-world physics concepts and phenomena. \
        You must ensure that the code you generate strictly adheres to the {client_dict[client]}'s syntax and capabilities. "},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=65536
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content

def json_code_qwen(prompt):
    # "role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    messages = [
        {"role": "system", "content": "You are a helpful code assistant that parses user input prompt and extracts the relevant object and physics information (such as mass and velocity) in json format."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=65536
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content

def query_code_gemini(prompt):
    response = code_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
    )
    return response.text

def query_code_gpt(prompt, gpt_model, client):
    response = gpt_client.responses.create(
        model=gpt_model,
        instructions=f"You are a helpful code assistant that generates and fixes python code to create physics simulations and animations using the {client_dict[client]}. \
        By generating code, you help users create physics-accurate video simulations from text prompt that follows real-world physics concepts and phenomena. \
        You must ensure that the code you generate strictly adheres to the {client_dict[client]}'s syntax and capabilities. Don not try to save the video, just make the window show the animation.",
        input=prompt,
    )
    return response.output_text

def general_query_gpt(prompt, gpt_model, client):
    response = gpt_client.responses.create(
        model=gpt_model,
        instructions=f"You are a helpful assistant that summarizes feedback and suggestions from users, and create straightforward actions a code agent can perform to improve the code quality and adhereance to user's intent. \
        By doing so, you help users create physics-accurate video simulations from text prompt that follows real-world physics concepts and phenomena. \
        You must ensure that the code you generate strictly adheres to the {client_dict[client]}'s syntax and capabilities.",
        input=prompt,
    )
    return response.output_text

def exposed_gpt_api(prompt, gpt_model, client, sys_instr):
    response = gpt_client.responses.create(
        model=gpt_model,
        instructions=sys_instr,
        input=prompt,
    )
    return response.output_text

# --- Socket Server ---
def handle_client(conn, addr):
    print(f"Connected by {addr}")
    try:
        # Receive message length (4 bytes, network byte order)
        raw_msglen = recvall(conn, 4)
        if not raw_msglen:
            return
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Receive the actual data
        data = recvall(conn, msglen)
        if not data:
            return
        # Parse JSON
        req = json.loads(data.decode('utf-8'))
        prompt = req.get('prompt_a')
        video_path = req.get('video_path')
        code = req.get('code')
        prompt_b = req.get('prompt_b')
        intent = req.get('intent', 'init_code')  # can be 'init_code', 'fix_code', 'update_code_feedback' or 'inquire_json'
        llm = req.get('llm', 'qwen')
        client = req.get('client', 'manim')
        if intent == 'init_code':
            print(f"Generating initial code with {llm}...")
            if prompt_b is None or prompt_b.strip() == "":
                json_response = json_code_qwen(prompt)
                json_data = json.dumps(get_json_from_response(json_response))
                construct_prompt = "Based on this objective: "+prompt+\
                    ", and this json data with detailed assets: ```json\n"+json_data+\
                    "``` please generate the full python script to render this scene in "+ client_dict[client] +". Please provide the full code only."
            else:
                construct_prompt = "Based on this objective: "+prompt+\
                    ", and this json data with detailed assets: "+prompt_b+\
                    " please generate the full python script to render this scene in "+ client_dict[client] +". Please provide the full code only."
            if llm == 'gemini':
                result_code = query_code_gemini(construct_prompt)
            elif llm == 'gpt':
                result_code = query_code_gpt(construct_prompt, "gpt-5", client)
            elif llm == 'qwen':
                result_code = query_code_qwen(construct_prompt, client)
            else:
                raise ValueError(f"Unsupported llm: {llm}")
            result = {'code': result_code, 'json': json_data}
        elif intent == 'fix_code':
            # in this mode we are expecting prompt_b to be the error message and code to be non-empty
            if llm == 'random':
                xllm = random.SystemRandom().choice(['qwen', 'gpt', 'gemini'])
            else:
                xllm = llm
            print("Fixing code with", xllm)
            result_code = query_error_code(prompt_b, code, client, xllm)
            result = {'code': result_code}
        elif intent == 'summarize_feedback':
            print("Summarizing feedback with GPT-5")
            prompt = "Given the following user feedback and suggestions: ```"+prompt_b+"```, please summarize the main points and provide clear, actionable steps that a code agent can take to improve the code quality and better align with the user's intent. \
                Focus on specific changes or enhancements that can be made to the code. Provide the summary in a concise manner."
            summary = general_query_gpt(prompt, "gpt-5", client)
            result = {'summary': summary}
        elif intent == 'update_code_feedback':
            if llm == 'random':
                xllm = random.SystemRandom().choice(['gpt', 'gemini'])
            else:
                xllm = llm
            print("Updating code with", xllm)
            result_code = query_vis_code(prompt_b, prompt, code, client, xllm)
            result = {'code': result_code}
        elif intent == 'inquire_json':
            if prompt is None or prompt.strip() == "":
                construct_prompt = f"Given this json file ```json\n{prompt_b}```, please tell me the most prominent objects and their color in a json format. Only provide the json data."
            else:
                construct_prompt = f"Given this json file ```json\n{prompt_b}```, "+prompt
            if llm == 'qwen':
                json_response = json_code_qwen(construct_prompt)
                json_data = json.dumps(get_json_from_response(json_response))
            else:
                json_data = exposed_gpt_api(construct_prompt, "gpt-5", client,
                    "You are a helpful assistant that is expert in parsing user input prompt and json data and extracts the relevant object and physics information as demand.")
            result = {'json': json_data}
        # Send back the result as JSON
        resp = json.dumps(result)
        send_msg(conn, resp.encode('utf-8'))
    except Exception as e:
        print(f"Error: {e}")
        resp = json.dumps({'error': str(e)})
        send_msg(conn, resp.encode('utf-8'))
    finally:
        conn.close()

def recvall(sock, n):
    # Helper function to receive n bytes or return None if EOF
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def start_server():
    print(f"Server listening on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    start_server()