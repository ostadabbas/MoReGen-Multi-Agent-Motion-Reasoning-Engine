
# Socket server imports
import socket
import threading
import struct
import json
import shutil

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from google import genai
import os
from math import ceil
import torch
import subprocess
import transformers
import cv2
import imageio.v3 as iio
from glob import glob
import numpy as np
from util import *

HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 56789      # Arbitrary non-privileged port
IMG_WIDTH = 420
IMG_HEIGHT = 270


print(transformers.__version__)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to('cuda')


def get_frame_list(output_path, start_frames=36, frames_to_keep=48):
    # Get all jpg files in the directory
    all_frames = sorted([f for f in os.listdir(output_path) if f.endswith('.jpg')])

    total_frames = len(all_frames)
    if frames_to_keep <= 0:
        frames_to_keep = total_frames - start_frames
    # Calculate the step size to evenly distribute the selected frames
    step = (total_frames-start_frames) // frames_to_keep

    # Select the frames
    selected_frames = all_frames[start_frames::step][:frames_to_keep]

    # Create the full paths for the selected frames
    frame_paths = [f"{os.path.join(output_path, frame)}" for frame in selected_frames]

    return frame_paths

def query_batch_images(prompts, image_paths):
    selected_frames = get_frame_list(image_paths,start_frames=0)
    print(selected_frames)
    ret = []
    for img in selected_frames:
        str_json = query_images(prompts, img)
        str_json = str_json[0].split("```json")[-1].split("```")[0]
        ret.append({img:str_json})
    return ret

def query_batch_images_until_obj(prompts, image_paths):
    selected_frames = get_frame_list(image_paths, 0, -1)
    for img in selected_frames:
        str_json = query_images(prompts, img)
        str_json = str_json[0].split("```json")[-1].split("```")[0]
        bbox = json.loads(str_json)
        if len(bbox) > 0:
            ret = {img:str_json}
            return ret
    return {}

def query_images(prompt, image_paths):
    # Create messages structure for images
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_paths,
                    "max_pixels": 360 * 420,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

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
        generated_ids = model.generate(**inputs, max_new_tokens=512)

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

def query_video(prompt, use_frames=True, frames_path="/home/qwen2_vl/content/frames", video_path=None):
    if use_frames:
        # Get the frames
        selected_frames = get_frame_list(frames_path)

        # Create messages structure for frames
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": selected_frames,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        # Create messages structure for the entire video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        if duration < 1.0:
            return False, ["Video too short"]
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
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    # Trim the generated output to remove the input prompt
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    torch.cuda.empty_cache()
    return True, output_text


    prompt = "Please generate full python script to render this scene in manim: " + prompt + " Please do not include any text or math equations. Please be physics accurate and respect physics rules. Please provide the full code only. Do not add text or equations."
    response = code_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
    )
    return response.text

def cotrack_pixel(frames_folder, pixels, start_frame, width, height):
    # points_xy : np.array[[x,y], ...]
    # resize???
    query_points = torch.tensor(pixels, dtype=torch.float32)[None, None].to('cuda')
    points_xy = query_points.squeeze(1)
    frame_pattern = os.path.join(frames_folder, "frame_*.jpg")
    frame_paths = sorted(glob(frame_pattern))
    frames = [cv2.resize(iio.imread(fp), (width, height)) for fp in frame_paths]
    frames_tensor = torch.from_numpy(np.stack(frames))
    video = frames_tensor.permute(0, 3, 1, 2)[None].float().to('cuda')
    # Resize points to match new frame size (assuming original points are in original frame size)
    # If points are already normalized [0,1], multiply by new size
    if points_xy.max() <= 1.0:
        points_xy = points_xy * torch.tensor([width, height], dtype=torch.float32).to(points_xy.device)
        points_xy = points_xy[None]
    else:
        points_xy = points_xy[None]
    with torch.amp.autocast('cuda', enabled=True):
        tracks, _ = cotracker(
                video[:, start_frame:],
                queries=points_xy.squeeze(0)
            )
    tracks = tracks[0].cpu().numpy()  # (T, N)
    return tracks

def get_bbox_center(x1, x2, y1, y2, width, height):
    # res 420 360
    if x1 < 1.0:
        x1 = int(x1 * width)
        x2 = int(x2 * width)
        y1 = int(y1 * height)
        y2 = int(y2 * height)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

def refine_bbox(obj_json, start_frame):
    print(obj_json)
    obj_json = obj_json if isinstance(obj_json, dict) else json.loads(obj_json)
    if 'bbox_2d' in obj_json.keys():
        obj_box = obj_json['bbox_2d']
    elif 'x1' in obj_json.keys():
        obj_box = (obj_json['x1'], obj_json['y1'], obj_json['x2'], obj_json['y2'])
    else:
        assert False, "No bbox coordinates found"
    cx, cy = get_bbox_center(obj_box[0], obj_box[2], obj_box[1], obj_box[3], IMG_WIDTH, IMG_HEIGHT)
    return [start_frame, cx, cy]

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
        purpose = req.get('purpose') # 'bbox' 'feedback'
        if purpose == 'feedback':
            feedback_prompt = req.get('feedback_prompt')
            original_prompt = req.get('original_prompt')
            video_path = req.get('video_path')
            # at the moment we only evaluate visual feedback
            res, phys_result = query_video(feedback_prompt, use_frames=False, video_path=video_path)
            res, intent_result = query_video(original_prompt, use_frames=False, video_path=video_path)
            result = {"visual_feedback_physics": phys_result,
                      "visual_feedback_intent": intent_result}
        elif purpose == 'bbox':
            objects = json.dumps(req.get('objects')) # a list of object name
            prompt = "Please provide the bounding box coordinates for each of the following objects in the format of a json list of dictionary, \
                where each dictionary contains 'object', 'x1', 'y1', 'x2', 'y2'. The coordinates should be normalized between 0 and 1. \
                    The objects are: ```json" + objects + "```. If an object is not present, please return an empty list."
            video_path = req.get('video_path')
            frame_pth = "./data/temp_frames/"
            if os.path.exists(frame_pth):
                shutil.rmtree(frame_pth)
            os.makedirs(frame_pth, exist_ok=True)
            extract_frames_from_video(video_path, frame_pth, fps=24)
            result = query_batch_images(prompt, frame_pth)
        elif purpose == 'bbox_v2':
            objects = json.dumps(req.get('objects')) # a list of object name
            prompt = "Please provide the bounding box coordinates for each of the following objects in the format of a json list of dictionary, \
                where each dictionary contains 'object', 'x1', 'y1', 'x2', 'y2'. The coordinates should be normalized between 0 and 1. \
                    The objects are: ```json" + objects + "```. If an object is not present, please return an empty list."
            video_path = req.get('video_path')
            frame_pth = "./data/temp_frames/"
            if os.path.exists(frame_pth):
                shutil.rmtree(frame_pth)
            os.makedirs(frame_pth, exist_ok=True)
            extract_frames_from_video(video_path, frame_pth, fps=24)
            result = query_batch_images_until_obj(prompt, frame_pth)
            start_frame = int(next(iter(result)).split("frame_")[-1].split(".jpg")[0])
            bboxes = next(iter(result.values()))
            pixels = []
            for item in json.loads(bboxes):
                pixels.append(refine_bbox(item, start_frame))
            print(pixels, start_frame)
            tracks = cotrack_pixel(frame_pth, np.array(pixels), start_frame, IMG_WIDTH, IMG_HEIGHT)
            result = {"tracks": tracks.tolist()}
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
        try:
            while True:
                conn, addr = s.accept()
                t = threading.Thread(target=handle_client, args=(conn, addr))
                t.daemon = True  # Make threads non-daemon so they finish before exit
                t.start()
        except KeyboardInterrupt:
            print("\nServer shutting down gracefully.")
    

if __name__ == "__main__":
    start_server()