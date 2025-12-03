import os
import torch
import cv2

import numpy as np
import imageio.v3 as iio

from util import *
from visualize import Visualizer
from PIL import Image
from glob import glob
from sklearn.cluster import DBSCAN
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to('cuda')

def remove_similar_points(locations, names, eps=0.1):
    """
    Removes near-duplicate [x, y] points from a tensor of [t, x, y] locations.
    
    Parameters:
        locations (np.ndarray): An (n, 3) array of [t, x, y] points.
        names (list): A list of n names corresponding to each point.
        eps (float): Distance threshold for clustering similar points.
    
    Returns:
        filtered_locations (np.ndarray): Deduplicated (m, 3) array.
        filtered_names (list): Corresponding names for filtered points.
    """
    assert locations.shape[0] == len(names), "Mismatch between locations and names length"
    
    # Cluster based on spatial proximity only
    spatial_coords = locations[:, 1:3]  # [x, y]
    clustering = DBSCAN(eps=eps, min_samples=1).fit(spatial_coords)
    labels = clustering.labels_
    
    # Keep the first point from each cluster
    unique_labels = np.unique(labels)
    filtered_locations = []
    filtered_names = []
    
    for label in unique_labels:
        idx = np.where(labels == label)[0][0]
        filtered_locations.append(locations[idx])
        filtered_names.append(names[idx])
    
    return torch.from_numpy(np.array(filtered_locations)).unsqueeze(0).unsqueeze(0), filtered_names

def _locate_objs_in_frame(img_location, start_frame, host, port, prompt_json, client, not_stuff=""):
    line = "please select one or two most important objects, name them in one or two words, and return in json format like this: ```json {'objects': ['obj1', 'obj2', ...]}```. Note this data is for grounded detection models, so please only include individual components not the whole system."
    resp = send_code_request(host, port, line+not_stuff, prompt_json, None, "inquire_json", "gpt", client)
    obj_json = json.loads(resp['json'])
    img_dirs = list(os.listdir(img_location))
    sorted(img_dirs)
    img = Image.open(img_location+"/"+img_dirs[0]).convert("RGB")
    og_w, og_h = img.size
    width, height  = og_w*0.5, og_h*0.5
    box_centers = []
    box_labels = []
    print("Objs from json:", obj_json['objects'])
    for obj in obj_json['objects']:
        if isinstance(obj, dict):
            text = next(iter(obj.values()))
        else:
            text = obj
        inputs = processor(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[img.size[::-1]]
        )

        boxes = results[0]['boxes'].cpu().numpy()
        text_label = results[0]['text_labels']
        box_labels += text_label
        # Each box is [x1, y1, x2, y2]; center is ((x1+x2)/2, (y1+y2)/2)
        centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
        # Add a column of 42s as the first column to centers
        print(centers.shape)
        centers = np.hstack([np.full((centers.shape[0], 1), start_frame), centers])
        box_centers.append(centers)

    # todo: remove >80% overlapping boxes

    pixels = np.concatenate(box_centers, axis=0)
    # Normalize: divide first column (x) by IMG_WEIGHT, second column (y) by IMG_HEIGHT
    pixels[..., 1] = pixels[..., 1] / og_w * width
    pixels[..., 2] = pixels[..., 2] / og_h * height
    return pixels, resp['json'], width, height, box_labels

def locate_objs_in_frame(img_location, start_frame, host, port, prompt_json, client):
    pixels, obj_stuff, width, height, box_labels = \
        _locate_objs_in_frame(img_location, start_frame, host, port, prompt_json, client, "")
    while len(pixels) < 1:
        pixels, obj_stuff, width, height, box_labels = \
            _locate_objs_in_frame(img_location, start_frame, host, port, prompt_json, client, \
                                  "Note that this is not what I am looking for: ```json"+obj_stuff+"``` as they do not correspond to the objects in the scene.")
    query_points = torch.tensor(pixels, dtype=torch.float32)[None, None].to('cuda')
    points_xy = query_points.squeeze(1).squeeze(2)
    if points_xy.max() <= 1.0:
        points_xy = points_xy * torch.tensor([width, height], dtype=torch.float32)
        points_xy = points_xy[None]
    else:
        points_xy = points_xy[None]
    print("Detected objects:", box_labels)
    return points_xy, width, height, box_labels

def cvt_video_tensor(frames_folder, width, height):
    frames = list(os.listdir(frames_folder))
    sorted(frames)
    frames = [cv2.resize(iio.imread(frames_folder+'/'+fp), (int(width), int(height))) for fp in frames]
    frames_tensor = torch.from_numpy(np.stack(frames))
    video = frames_tensor.permute(0, 3, 1, 2)[None].float().to('cuda')
    return video

def track_objs_in_video(video, points_xy, obj_names):
    points_xy, obj_names = remove_similar_points(points_xy[0,0].cpu().numpy(), obj_names, eps=2.0)
    with torch.amp.autocast('cuda', enabled=True):
        tracks, visibility = cotracker(
                video,
                queries=points_xy.squeeze(0).to('cuda'),
            )
    return tracks, visibility

def show_in_video(save_dir, video_tensor, file_name, tracks, visibility):
    vis = Visualizer(
        save_dir=save_dir,
        linewidth=2,
        fps=10,
        mode='cool',
        tracks_leave_trace=-1
    )
    vis.visualize(
        video=video_tensor,
        tracks=tracks,
        visibility=visibility,
        filename=file_name)

def normalize_trajectory(traj):
    if np.isnan(traj).any():
        raise ValueError("Input trajectory contains NaN values.")

    # Step 2: Center the trajectory
    traj_centered = traj - np.mean(traj, axis=0)

    # Step 3: Normalize scale (unit arc length)
    diffs = np.diff(traj_centered, axis=0)
    arc_length = np.sum(np.linalg.norm(diffs, axis=1))

    if arc_length == 0:
        traj_scaled = np.zeros_like(traj_centered)
    else:
        traj_scaled = traj_centered / arc_length

        traj_scaled = traj_centered / arc_length

    if np.isnan(traj_scaled).any():
        print("NaN values found in trajectory after normalization.")
    return traj_scaled

def evaluate_trajectory(tracked_pred, tracked_gt):
    tracked_gt = normalize_trajectory(tracked_gt)
    tracked_pred = normalize_trajectory(tracked_pred)
    dtw = compute_dtw_distance(tracked_pred, tracked_gt)
    # Create a new array from tracked_gt using the DTW alignment indices
    # print("DTW indices:", dtw.index1, dtw.index2)
    # print("Tracked pred shape:", tracked_pred.shape, "Tracked gt shape:", tracked_gt.shape)
    aligned_pred = tracked_pred[dtw.index1]
    aligned_gt = tracked_gt[dtw.index2]
    proc = procrustes_distance(aligned_pred, aligned_gt)
    # print(f"DTW distance: {dtw.distance}, DTW Normalized: {dtw.normalizedDistance}, Procrustes distance: {proc}")
    return aligned_pred, aligned_gt, dtw.distance, dtw.normalizedDistance, proc