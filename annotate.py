import cv2
import os
import json
import torch
import numpy as np
import imageio.v3 as iio
from typing import List, Tuple

# Configuration - update this to your folder
video_folder = r"D:/OneDrive/OneDrive - Northeastern University/newton/pully"

# Data structure for backward compatibility
trajectories = {}

# load cotracker once
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to('cuda')

# Target settings: downsample to 10Hz and resize to 480p height
TARGET_FPS = 10
TARGET_HEIGHT = 480


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


def fallback_save_overlay(frames: List[np.ndarray], tracks: torch.Tensor, visibility: torch.Tensor, out_path: str, fps: float = 10.0):
    """Draw tracked points onto frames and save an mp4 as a fallback when Visualizer is not available."""
    # tracks: either (num_tracks, T, 2) or (T, num_tracks, 2)
    tracks_np = tracks.cpu().numpy()
    vis_np = visibility.cpu().numpy() if isinstance(visibility, torch.Tensor) else np.array(visibility)

    if tracks_np.ndim == 3 and tracks_np.shape[0] == frames.__len__():
        # (T, num_tracks, 2) -> (num_tracks, T, 2)
        tracks_np = np.transpose(tracks_np, (1, 0, 2))

    num_tracks = tracks_np.shape[0]
    T = frames.__len__()

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    colors = [tuple(int(c) for c in np.random.randint(0, 255, 3).tolist()) for _ in range(num_tracks)]

    for t in range(T):
        frame = frames[t].copy()
        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        for i in range(num_tracks):
            x, y = tracks_np[i, t]
            # skip NaNs or out-of-bounds
            if np.isnan(x) or np.isnan(y):
                continue
            if vis_np.ndim == 2:
                visible = vis_np[i, t]
            else:
                # fallback: treat >0 as visible
                visible = vis_np[i, t] if vis_np.shape == (num_tracks, T) else 1
            if visible:
                cv2.circle(frame_bgr, (int(x), int(y)), 4, colors[i], -1)
        writer.write(frame_bgr)
    writer.release()


def visualize_tracks_live(frames: List[np.ndarray], tracks, visibility, fps: float = 10.0):
    """Display frames with tracked points drawn from `tracks`.

    Expected `tracks` shape: (1, T, N, 2) or (T, N, 2) or (N, T, 2). Visibility may be shaped similarly.
    Controls while viewing:
      - 'q' to quit early
      - 'p' to pause/resume
    """
    # Convert to numpy
    if torch.is_tensor(tracks):
        t_np = tracks.cpu().numpy()
    else:
        t_np = np.array(tracks)

    if torch.is_tensor(visibility):
        vis_np = visibility.cpu().numpy()
    else:
        vis_np = np.array(visibility)

    # remove leading batch dim if present
    if t_np.ndim == 4 and t_np.shape[0] == 1:
        t_np = t_np[0]

    # Now ensure shape is (T, N, 2)
    if t_np.ndim == 3 and t_np.shape[2] == 2:
        # if first dim is N and second is T, swap
        if t_np.shape[0] != len(frames) and t_np.shape[1] == len(frames):
            t_np = np.transpose(t_np, (1, 0, 2))
    else:
        raise RuntimeError(f"Unexpected tracks shape: {t_np.shape}")

    T = len(frames)
    if t_np.shape[0] != T:
        # best-effort: try to find a compatible axis permutation
        raise RuntimeError(f"Track length {t_np.shape[0]} doesn't match number of frames {T}")

    N = t_np.shape[1]

    # Normalize visibility into boolean [T, N]
    if vis_np.size == 0:
        vis_bool = np.ones((T, N), dtype=bool)
    else:
        if vis_np.ndim == 3 and vis_np.shape[0] == 1:
            vis_np = vis_np[0]
        if vis_np.shape == (T, N):
            vis_bool = vis_np.astype(bool)
        elif vis_np.shape == (N, T):
            vis_bool = vis_np.transpose(1, 0).astype(bool)
        elif vis_np.shape == (N,):
            vis_bool = np.tile(vis_np.reshape(1, N).astype(bool), (T, 1))
        else:
            # fallback: treat any nonzero as visible
            try:
                vis_bool = (vis_np != 0).reshape(T, N)
            except Exception:
                vis_bool = np.ones((T, N), dtype=bool)

    # Prepare colors
    rng = np.random.RandomState(1234)
    colors = [tuple(int(c) for c in rng.randint(0, 255, 3).tolist()) for _ in range(N)]

    win = "tracked_preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    paused = False
    t = 0
    delay = int(max(1, round(1000.0 / float(fps))))

    # interactive edit state
    edit_target = None  # track index being edited or None
    last_click = {'pos': None}

    def _mouse_cb(event, x, y, flags, param):
        # Only record clicks when we're in edit mode
        nonlocal edit_target
        if edit_target is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            last_click['pos'] = (x, y)

    cv2.setMouseCallback(win, _mouse_cb)

    while t < T:
        frame = frames[t].copy()
        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        for i in range(N):
            x, y = t_np[t, i]
            if np.isnan(x) or np.isnan(y):
                continue
            if not vis_bool[t, i]:
                continue
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < frame_bgr.shape[1] and 0 <= iy < frame_bgr.shape[0]:
                # draw the point
                cv2.circle(frame_bgr, (ix, iy), 4, colors[i], -1)
                # draw the class/index label next to the point with outline for readability
                label = str(i)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thickness = 1
                text_size, _ = cv2.getTextSize(label, font, scale, thickness)
                text_x = ix + 6
                text_y = iy - 6 if iy - 6 > text_size[1] else iy + text_size[1] + 6
                # outline
                cv2.putText(frame_bgr, label, (text_x, text_y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                # main text
                cv2.putText(frame_bgr, label, (text_x, text_y), font, scale, colors[i], thickness, cv2.LINE_AA)

        # if in edit mode highlight the target track
        if edit_target is not None and 0 <= edit_target < N:
            # draw instruction overlay
            cv2.putText(frame_bgr, f"Editing point {edit_target}: click new location", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
            # if user already clicked, show the tentative point
            if last_click['pos'] is not None:
                cx, cy = last_click['pos']
                cv2.circle(frame_bgr, (cx, cy), 6, (0, 0, 255), 2)

        cv2.imshow(win, frame_bgr)

        key = cv2.waitKey(0) & 0xFF
        # 'q' to quit
        if key == ord('q'):
            break
        # 'p' to pause/resume (no-op here since we block until action)
        if key == ord('p'):
            paused = not paused
            continue

        # Enter: confirm this frame and go to next
        if key in (13, 10):
            # clear any edit state and advance
            edit_target = None
            last_click['pos'] = None
            t += 1
            continue

        # number keys 0-9 to edit corresponding track index
        if ord('0') <= key <= ord('9'):
            idx = key - ord('0')
            if idx >= N:
                print(f"Pressed index {idx} out of range (N={N})")
                continue
            # enter edit mode for this index
            edit_target = idx
            last_click['pos'] = None
            print(f"Editing track {edit_target} at frame {t}. Click new location in window.")
            # wait until a click is recorded
            while last_click['pos'] is None:
                # small wait to process events
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            if last_click['pos'] is not None:
                cx, cy = last_click['pos']
                # write into tracks array for this frame and index
                t_np[t, edit_target, 0] = float(cx)
                t_np[t, edit_target, 1] = float(cy)
                vis_bool[t, edit_target] = True
                print(f"Replaced track {edit_target} at frame {t} with ({cx}, {cy})")
            # stay on same frame to allow more edits or confirm
            edit_target = None
            last_click['pos'] = None
            continue

        # other keys: advance one frame
        t += 1

    cv2.destroyWindow(win)
    # return corrected tracks (T,N,2) and visibility (T,N)
    return t_np, vis_bool


def save_tracks_json(tracks_np: np.ndarray, vis_np: np.ndarray, out_path: str, names: dict = None):
    """Save corrected tracks and visibility to a JSON file.

    tracks_np: (T, N, 2)
    vis_np: (T, N) boolean
    names: optional mapping (int or str index) -> short name (1-2 words)
    """
    T, N, _ = tracks_np.shape
    data = {
        'T': int(T),
        'N': int(N),
        'tracks': [],
    }
    for t in range(T):
        frame_entries = []
        for i in range(N):
            entry = {
                'index': int(i),
                'visible': bool(vis_np[t, i]),
                'x': float(tracks_np[t, i, 0]) if not np.isnan(tracks_np[t, i, 0]) else None,
                'y': float(tracks_np[t, i, 1]) if not np.isnan(tracks_np[t, i, 1]) else None,
            }
            frame_entries.append(entry)
        data['tracks'].append(frame_entries)

    # include optional names mapping (stringified indices)
    if names is not None:
        # normalize keys to strings and values to either None or short strings
        norm = {}
        for k, v in (names.items() if isinstance(names, dict) else []):
            try:
                key = str(int(k))
            except Exception:
                key = str(k)
            norm[key] = None if v is None else str(v)
        data['names'] = norm

    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)


def process_video(video_path: str, filename: str, out_dir: str):
    """Prompt the user to click on the first frame; then run cotracker to track those points through the whole video.

    Saves an output video with tracked points alongside the input video (suffix _tracked.mp4).
    """
    print(f"Processing: {video_path}")
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

    clicked_points = []
    win = "Click on 1 or 2 pixels (first frame only)"
    # create a resizable window but keep image aspect ratio and resize it to the frame size
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(win, new_w, TARGET_HEIGHT)
    # start receiving mouse events
    cv2.setMouseCallback(win, click_callback, clicked_points)
    print("Click the points on the first frame. They will appear immediately. Press Enter to start tracking, or 'q' to skip this video.")

    # interactive loop: update display live as user clicks; Enter to proceed
    while True:
        display = resized_first.copy()
        # draw any clicked points immediately
        for (x, y) in clicked_points:
            try:
                cx, cy = int(round(x)), int(round(y))
            except Exception:
                cx, cy = int(x), int(y)
            if 0 <= cx < display.shape[1] and 0 <= cy < display.shape[0]:
                cv2.circle(display, (cx, cy), 6, (0, 255, 0), -1)
        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF
        # Enter (start tracking)
        if key in (13, 10):
            break
        # 'q' to cancel/skip this video
        if key == ord('q'):
            cv2.setMouseCallback(win, lambda *args: None)
            cv2.destroyWindow(win)
            print("Skipping this video by user request.")
            return

    # stop receiving mouse events and close the window before tracking
    cv2.setMouseCallback(win, lambda *args: None)
    cv2.destroyWindow(win)

    if len(clicked_points) == 0:
        print("No points clicked; skipping video.")
        return

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

    # Try to visualize with Visualizer if available, otherwise fallback
    try:
        # show a live preview using local renderer
        print(f"Tracks shape: {tracks.shape}, visibility shape: {getattr(visibility, 'shape', 'N/A')}")
        corrected_tracks, corrected_visibility = visualize_tracks_live(frames, tracks, visibility, fps=fps or TARGET_FPS)

        # after user confirmed all frames, ask the user to name each object index (1-2 words)
        # corrected_tracks expected shape: (T, N, 2)
        try:
            _, N, _ = corrected_tracks.shape
        except Exception:
            # fallback: try other possible ordering
            try:
                N = corrected_tracks.shape[1]
            except Exception:
                N = 0

        names = {}
        if N > 0:
            print("All frames corrected. Please provide a short name (1-2 words) for each object index.")
            for i in range(N):
                while True:
                    try:
                        raw = input(f"Name for object index {i} (1-2 words): ").strip()
                    except EOFError:
                        # non-interactive environment: skip
                        raw = ""
                    if raw == "":
                        names[i] = None
                        continue
                    parts = raw.split()
                    if 1 <= len(parts) <= 2:
                        names[i] = " ".join(parts)
                        break
                    else:
                        print("Please enter at most two words. Try again.")

        # after naming, save corrected tracks to JSON
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, filename.split('.')[0] + '_tracks_corrected.json')
        save_tracks_json(corrected_tracks, corrected_visibility, json_path, names=names)
        print(f"Saved corrected tracks to: {json_path}")
    except Exception as e:
        print("Live visualization failed, falling back to saving overlay. Error:", e)
        out_path = "./data/tracked/" + filename.split(".")[0] + "_tracked.mp4"
        fallback_save_overlay(frames, tracks, visibility, out_path, fps=fps or TARGET_FPS)
        print(f"Saved fallback overlay video to: {out_path}")


def main():
    out_dir = r"F:/Projects/agent_synthetic/pymunk/data/tracked/"
    os.makedirs(out_dir, exist_ok=True)
    for filename in os.listdir(video_folder):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        base = os.path.splitext(filename)[0]
        # check for existing corrected output; skip if present
        candidate1 = os.path.join(out_dir, base + '_tracked_corrected.json')
        candidate2 = os.path.join(out_dir, base + '_tracks_corrected.json')
        if os.path.exists(candidate1) or os.path.exists(candidate2):
            existing = candidate1 if os.path.exists(candidate1) else candidate2
            print(f"Skipping {filename}: corrected JSON already exists: {existing}")
            continue

        process_video(os.path.join(video_folder, filename), filename, out_dir)


if __name__ == "__main__":
    main()
