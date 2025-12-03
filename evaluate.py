# 1. open gt annotation file and load object name to be labeled
# 2. open videos and for each label ask user to click
# 3. track with cotracker3
# 4. load corresponding gt track
# 5. evaluate with our metrics
# 6. evaluate with trajan and videophy

import os
import cv2
import torch
import json
import numpy as np
from typing import List, Tuple
from util import *
from trajectory import *

cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to('cuda')

GT_LOC = r"F:/Projects/agent_synthetic/dataset/point_tracks"
VID_LOC = r"F:/Projects/agent_synthetic/experiments/veo3-outputs"
TARGET_FPS = 10
TARGET_HEIGHT = 480
TEMP_DIR = r"./data/eval_obj"
source = "veo" # ltx, veo, ours, wan, cog, ...

def click_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append((x, y))

def video_to_tensor(video_path: str, device: str = 'cuda') -> Tuple[torch.Tensor, List[np.ndarray], float]:
    """Read a video and return a tensor suitable for cotracker plus raw frames list and fps.

    Returns:
        video_tensor: torch.Tensor shaped (1, T, C, H, W) on device
        frames: list of HxWxC numpy arrays (uint8)
        fps: frames per second (float)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    # assume 120 if not readable
    if orig_fps <= 0:
        orig_fps = 120.0

    # compute step to downsample to TARGET_FPS
    step = max(1, int(round(orig_fps / float(TARGET_FPS))))
    out_fps = orig_fps / step

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # resize to target height preserving aspect
            h, w = frame.shape[:2]
            scale = float(TARGET_HEIGHT) / float(h)
            new_w = max(1, int(round(w * scale)))
            new_h = TARGET_HEIGHT
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # convert BGR -> RGB
            frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {video_path}")

    arr = np.stack(frames).astype(np.float32)
    # [T, H, W, C] -> [T, C, H, W] -> [1, T, C, H, W]
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)[None].float().to(device)
    return tensor, frames, float(out_fps)

def process_video(video_path: str, corr: dict, exp:str):
    """Prompt the user to click on the first frame; then run cotracker to track those points through the whole video.

    Saves an output video with tracked points alongside the input video (suffix _tracked.mp4).
    """
    print(f"Processing: {video_path}")
    clicked_points = []
    search_dir = os.path.join(TEMP_DIR, source)
    if exp+"_json.json" in os.listdir(search_dir):
        print(f"Skipping asking for existing video: {video_path}")
        clicked_points = json.load(open(os.path.join(search_dir, exp+"_json.json"), 'r'))
    # read first frame and resize it to TARGET_HEIGHT so clicks map to the processed frames
    cap = cv2.VideoCapture(video_path)
    ret, first = cap.read()
    cap.release()
    if not ret:
        print(f"Couldn't read first frame for {video_path}")
        return

    # resize first frame same as video_to_tensor (maintain aspect ratio)
    h, w = first.shape[:2]
    scale = float(TARGET_HEIGHT) / float(h)
    new_w = max(1, int(round(w * scale)))
    resized_first = cv2.resize(first, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    print(f"Resized first frame to: {new_w}x{TARGET_HEIGHT}")

    # We'll ask the user to click one point for each label in `corr`.
    # Support `corr` being a dict (index->label) or a list of labels.
    entries = []
    if isinstance(corr, dict):
        # sort by index for deterministic order
        try:
            entries = sorted(((int(k), v) for k, v in corr.items()), key=lambda kv: kv[0])
        except Exception:
            entries = list(enumerate(corr.values()))
    elif isinstance(corr, list):
        entries = list(enumerate(corr))
    else:
        # fallback: single unlabeled entry
        entries = [(0, "point")]

    if len(clicked_points) == 0:
        win = "Click points for labels (first frame only)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(win, new_w, TARGET_HEIGHT)

        # For each label, ask the user to click once. The user confirms with Enter.
        for idx, label in entries:
            current_clicks = []
            # set callback to collect clicks for this label
            cv2.setMouseCallback(win, click_callback, current_clicks)
            print(f"Please click the point for label '{label}' (index {idx}). Press Enter to confirm, or 'q' to skip the video.")

            while True:
                display = resized_first.copy()
                # draw the label text on the image
                try:
                    cv2.putText(display, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                except Exception:
                    pass

                # draw any clicks for this label (take the most recent)
                if len(current_clicks) > 0:
                    try:
                        cx, cy = current_clicks[-1]
                        cx, cy = int(round(cx)), int(round(cy))
                        if 0 <= cx < display.shape[1] and 0 <= cy < display.shape[0]:
                            cv2.circle(display, (cx, cy), 6, (0, 255, 0), -1)
                    except Exception:
                        pass

                cv2.imshow(win, display)
                key = cv2.waitKey(30) & 0xFF
                # Enter confirms this label's click
                if key in (13, 10):
                    # if user didn't click, store an empty placeholder
                    if len(current_clicks) == 0:
                        print(f"No click recorded for label '{label}'; storing empty placeholder.")
                        clicked_points.append([np.nan, np.nan])
                    else:
                        px, py = current_clicks[-1]
                        clicked_points.append([float(px), float(py)])
                    break
                # 'q' cancels the whole video
                if key == ord('q'):
                    cv2.setMouseCallback(win, lambda *args: None)
                    cv2.destroyWindow(win)
                    print("Skipping this video by user request.")
                    return

            # remove callback for this label before next
            cv2.setMouseCallback(win, lambda *args: None)

        # close window after all labels clicked
        cv2.destroyWindow(win)

        if len(clicked_points) == 0:
            print("No points clicked; skipping video.")
            return
        else:
            json.dump(clicked_points, open(os.path.join(search_dir, exp+"_json.json"), 'w'))

    # Prepare video tensor and raw frames
    video_tensor, frames, fps = video_to_tensor(video_path, device='cuda')

    # Prepare queries: shape (1, N, 3) where each entry is [start_frame, x, y]
    pts = np.array(clicked_points, dtype=np.float32)
    # start_frame is the index within the processed/downsampled frames; we used the first frame so default 0
    start_frame = 0
    # pts currently is (N, 2) = (x, y); stack start_frame as the first column -> (N, 3)
    starts = np.full((pts.shape[0], 1), float(start_frame), dtype=np.float32)
    pts3 = np.hstack([starts, pts])
    points_xy = torch.from_numpy(pts3)[None].float()

    # Call cotracker under autocast
    with torch.amp.autocast('cuda', enabled=True):
        tracks, visibility = cotracker(
            video_tensor,
            queries=points_xy.to('cuda'),
        )
    t_np = tracks.cpu().numpy()
    if t_np.ndim == 4 and t_np.shape[0] == 1:
        t_np = t_np[0]
        t_np = np.transpose(t_np, (1, 0, 2))
    else:
        raise RuntimeError(f"Unexpected tracks shape: {t_np.shape}")
    
    return t_np

def load_gt(exp_name: str):
    """Load ground truth tracking data for the given experiment name."""
    gt_path = os.path.join(GT_LOC, exp_name + "_tracks_corrected.json")
    if not os.path.exists(gt_path):
        print(f"GT file not found: {gt_path}")
        return None
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    # gt_data['tracks'] is expected to be a list of frames, where each frame is
    # a list of dict entries like {"index": int, "visible": bool, "x": float|None, "y": float|None}
    frames = gt_data.get('tracks', [])
    if not isinstance(frames, list) or len(frames) == 0:
        print(f"GT data has no 'tracks' or is empty: {gt_path}")
        return None

    # Determine number of unique track indices
    max_idx = -1
    for frame in frames:
        for entry in frame:
            try:
                idx = int(entry.get('index', -1))
            except Exception:
                idx = -1
            if idx > max_idx:
                max_idx = idx

    if max_idx < 0:
        print(f"No valid track indices found in GT: {gt_path}")
        return None

    N = max_idx + 1
    T = len(frames)

    # Initialize dict: index -> list of length T filled with [] (meaning not visible / missing)
    tracks_dict = {i: [[] for _ in range(T)] for i in range(N)}

    for t, frame in enumerate(frames):
        for entry in frame:
            idx = entry.get('index')
            if idx is None:
                continue
            try:
                idx = int(idx)
            except Exception:
                continue
            x = entry.get('x')
            y = entry.get('y')
            # If visible and x/y are not None, store as two-element list of floats
            if entry.get('visible', False) and x is not None and y is not None:
                try:
                    tracks_dict[idx][t] = [float(x), float(y)]
                except Exception:
                    tracks_dict[idx][t] = []
            else:
                # leave as [] to indicate not visible / missing
                tracks_dict[idx][t] = []

    return tracks_dict, gt_data['names']

def cvt_gt_to_array(tracks_dict: dict) -> np.ndarray:
    """Convert tracks_dict (index -> list of length T with [x,y] or []) to numpy array of shape (T, N, 2) with np.nan for missing."""
    N = len(tracks_dict)
    T = len(tracks_dict[0]) if N > 0 else 0
    arr = np.full((T, N, 2), np.nan, dtype=np.float32)
    for idx in range(N):
        track = tracks_dict.get(idx, [])
        for t in range(T):
            point = track[t] if t < len(track) else []
            if isinstance(point, list) and len(point) == 2:
                # ensure floats stored
                try:
                    arr[t, idx, 0] = float(point[0])
                    arr[t, idx, 1] = float(point[1])
                except Exception:
                    # keep as nan if conversion fails
                    pass
            else:
                # leave as np.nan
                pass
    # Forward-fill missing (np.nan) points temporally per track index.
    # For each track (index), iterate frames in time order and replace any
    # missing coordinates with the last seen (previous non-nan) coordinates.
    # Leading missing values (no previous observation) remain np.nan.
    for idx in range(N):
        last_x = np.nan
        last_y = np.nan
        for t in range(T):
            x = arr[t, idx, 0]
            y = arr[t, idx, 1]
            # treat point as missing if either coordinate is nan
            if np.isnan(x) or np.isnan(y):
                if not (np.isnan(last_x) or np.isnan(last_y)):
                    arr[t, idx, 0] = last_x
                    arr[t, idx, 1] = last_y
                # else: no previous value, leave as 0.0
                else:
                    arr[t, idx, 0] = 0.0
                    arr[t, idx, 1] = 0.0
            else:
                # update last seen
                last_x = x
                last_y = y

    return np.transpose(arr, (1, 0, 2))

def process_our_traj(video_json_path: str):
    """Process our own trajectory JSON output into numpy array of shape (N, T, 2)."""
    with open(video_json_path, 'r') as f:
        data = json.load(f)
    data_tracks = data.get('samples', [])
    for item in data_tracks:
        item = item["bodies"]
        item_name = item.get('name', '')

def batch_eval(source):
    dtw, dtw_n, proc = [], [], []
    for exp in os.listdir(VID_LOC):
        if source == "ltx":
            gt_tracks, corr = load_gt(exp)
            gt_tracks = cvt_gt_to_array(gt_tracks)
            vid = list(os.listdir(os.path.join(VID_LOC, exp)))[-1]
            track = process_video(os.path.join(VID_LOC, exp, vid), corr, exp.split(".")[0])
        elif source == "wan" or source == "cog" or source == "sora" or source == "ours" or source == "newtongen" or source == "grok" or source == "gf10" or source == "gr10":
            gt_tracks, corr = load_gt(exp.split(".")[0])
            gt_tracks = cvt_gt_to_array(gt_tracks)
            track = process_video(os.path.join(VID_LOC, exp), corr, exp.split(".")[0])
        elif source == "wisa":
            with open(r"F:/Projects/agent_synthetic/experiments/wisa_gen_asset_index.json", 'r') as f:
                asset_idx = json.load(f)
            exp_name_id = exp.split(".")[0].split("_")[1]
            gt_tracks, corr = load_gt(asset_idx[exp_name_id])
            gt_tracks = cvt_gt_to_array(gt_tracks)
            track = process_video(os.path.join(VID_LOC, exp), corr, asset_idx[exp_name_id])
        elif source == "veo":
            exp_name = exp.split(" ")[1].split(".")[0]
            try:
                gt_tracks, corr = load_gt(exp_name)
                gt_tracks = cvt_gt_to_array(gt_tracks)
                track = process_video(os.path.join(VID_LOC, exp), corr, exp_name)
            except Exception as e:
                print(f"Skipping {exp} due to error: {e}")
                continue

        # print(gt_tracks.shape, track.shape)
        print("pred", np.isnan(track).sum(), np.isnan(gt_tracks).sum())
        
        for i in range(track.shape[0]):
            _1, _2, dtw1, dtw_n1, proc1 = evaluate_trajectory(track[i], gt_tracks[i])
            dtw.append(dtw1)
            dtw_n.append(dtw_n1)
            proc.append(proc1)
    mean_dtw = np.mean(dtw)
    std_dtw = np.std(dtw)
    mean_dtw_n = np.mean(dtw_n)
    std_dtw_n = np.std(dtw_n)
    mean_proc = np.mean(proc)
    std_proc = np.std(proc)
    print(f"Mean DTW: {mean_dtw}, Mean DTW Normalized: {mean_dtw_n}, Mean Procrustes: {mean_proc}")
    print(f"STD DTW: {std_dtw}, STD DTW Normalized: {std_dtw_n}, STD Procrustes: {std_proc}")


if __name__ == "__main__":
    os.makedirs(os.path.join(TEMP_DIR, source), exist_ok=True)
    batch_eval(source)