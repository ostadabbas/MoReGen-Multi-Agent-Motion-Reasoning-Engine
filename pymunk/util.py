import os
import subprocess

import socket
import struct
import json
import shutil
import numpy as np

from dtw import dtw
from numpy.linalg import norm
from scipy.spatial import procrustes


def compute_dtw_distance(traj1, traj2):
    try:
        alignment = dtw(traj1, traj2, keep_internals=True)
        return alignment
    except Exception as e:
        print(traj1, traj2)
        print(e)

def procrustes_distance(traj1, traj2):
    mtx1, mtx2, disparity = procrustes(traj1, traj2)
    return disparity

def extract_frames_from_video(video_path, output_folder, fps=1):
    """
    Extracts frames from a video file at a specified frames per second (fps).
    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Folder to save the extracted frames.
        fps (int): Frames per second to extract.
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(output_folder, "frame_%04d.jpg")
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Frames extracted to: {output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error during frame extraction: {e.stderr.decode()}")

def downsample_video_fps(input_path, output_path, fps):
    """
    Downsamples a video to fps using ffmpeg.
    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the downsampled video.
    """
    if os.path.exists(output_path):
        os.remove(output_path)
    command = [
        "ffmpeg",
        "-i", input_path,
        "-filter:v", "fps={}".format(fps),
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video downsampled to {fps}fps: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during downsampling: {e.stderr.decode()}")

def save_track_json(tracks, pred_or_gt, output_path):
    """
    Saves tracking data to a JSON file.
    """
    track = tracks.cpu().numpy()[0]
    data = {
        pred_or_gt: track.tolist() if hasattr(track, 'tolist') else track
    }
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"Tracking data saved to: {output_path}")


def load_track_json(input_path, key=None, to_torch=False, torch_module=None):
    """
    Loads tracking data saved by `save_track_json`.

    Args:
        input_path (str): Path to the JSON file produced by `save_track_json`.
        key (str, optional): The top-level key to load (e.g. 'pred' or 'gt').
            If None, the first key found in the file will be used.
        to_torch (bool, optional): If True, converts the returned numpy array to a
            PyTorch tensor. The caller must either have PyTorch installed or pass
            the torch module via `torch_module`.
        torch_module (module, optional): The torch module to use for conversion
            (useful if torch isn't imported globally). If None and `to_torch` is
            True, the function will attempt to import torch.

    Returns:
        numpy.ndarray or torch.Tensor: The loaded track data. The shape will match
            what was saved (typically [T, D] or similar).

    Raises:
        FileNotFoundError: If `input_path` does not exist.
        ValueError: If the file contains no valid track data or the requested key
            is missing.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Track JSON not found: {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Invalid or empty track JSON: {input_path}")

    if key is None:
        # take the first key
        key = next(iter(data.keys()))

    if key not in data:
        keys_str = ','.join(list(data.keys()))
        raise ValueError(f"Key '{key}' not found in track JSON. Available keys: {keys_str}")

    arr = np.array(data[key])

    if to_torch:
        if torch_module is None:
            try:
                import torch
                torch_module = torch
            except Exception as e:
                raise ImportError("to_torch=True requested but torch is not available") from e
        return torch_module.from_numpy(arr).unsqueeze(0)

    return arr

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

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# init_code: 'gemini', 'qwen', 'gpt' or None
def send_code_request(host, port, prompt, prompt_json, code, intent, llm, client):
    req = {
        'prompt_a': prompt,
        'prompt_b': prompt_json,
        'video_path': None,
        'code': code,
        'intent': intent,
        'llm': llm,
        'client': client
    }
    data = json.dumps(req).encode('utf-8')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
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
