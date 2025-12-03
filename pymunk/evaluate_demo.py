from util import *
from trajectory import *

import numpy as np
import torch
import json
import os

VIS_HOST = '127.0.0.1'  # Server address
VIS_PORT = 56789        # Server port
CODE_HOST = 'zzzura.duckdns.org'  # Server address
CODE_PORT = 10586        # Server port

BASE_DIR = r"F:/Projects/agent_synthetic/pymunk/generations/20250924-165819/"
pred_video = BASE_DIR + r"output_0001.mp4"
gt_video = BASE_DIR + r"Ball_slide_00.mp4"
veo_video = BASE_DIR + r"Ball_Rolls_Down_Slope_Hits_Plank.mp4"

if os.path.exists(BASE_DIR+"/tracked_gt.json") and os.path.exists(BASE_DIR+"/tracked_pred.json") and os.path.exists(BASE_DIR+"/tracked_veo.json"):
    print("Tracking files already exist. Evaluating...")
    with open(BASE_DIR + "/tracked_gt.json", "r") as f:
        tracked_gt = json.load(f)['track']
        tracked_gt = np.array(tracked_gt)[:,0,:]
    with open(BASE_DIR + "/tracked_pred.json", "r") as f:
        tracked_pred = json.load(f)['track']
        tracked_pred = np.array(tracked_pred)[:,0,:]
    with open(BASE_DIR + "/tracked_veo.json", "r") as f:
        tracked_veo = json.load(f)['track']
        tracked_veo = np.array(tracked_veo)[:,0,:]
    print("Ours: ")
    _1, _2, _3, _4 = evaluate_trajectory(tracked_pred, tracked_gt)
    print("VEO: ")
    _1, _2, _3, _4 = evaluate_trajectory(tracked_veo, tracked_gt)
    exit(0)

downsample_video_fps(gt_video, BASE_DIR+"/gt_video_10fps.mp4", fps=10)
extract_frames_from_video(BASE_DIR+"/gt_video_10fps.mp4", BASE_DIR+"/gt_frames", fps=10)
downsample_video_fps(pred_video, BASE_DIR+"/pred_video_10fps.mp4", fps=10)
extract_frames_from_video(BASE_DIR+"/pred_video_10fps.mp4", BASE_DIR+"/pred_frames", fps=10)
downsample_video_fps(veo_video, BASE_DIR+"/veo_video_10fps.mp4", fps=10)
extract_frames_from_video(BASE_DIR+"/veo_video_10fps.mp4", BASE_DIR+"/veo_frames", fps=10)

with open("prompt_json.json", 'r') as f:
    constructed_json = f.read()

obj_pts, w, h, obj_names = locate_objs_in_frame(BASE_DIR+"/pred_frames", start_frame=0, host=CODE_HOST, port=CODE_PORT, prompt_json=constructed_json, client='pymunk')
video_tensor = cvt_video_tensor(BASE_DIR+"/pred_frames", w, h)
tracks, visibility = track_objs_in_video(video_tensor, obj_pts, obj_names)
show_in_video(BASE_DIR, video_tensor, "tracked_pred", tracks, visibility)
save_track_json(tracks, 'track', BASE_DIR + "/tracked_pred.json")

obj_pts, w, h, obj_names = locate_objs_in_frame(BASE_DIR+"/gt_frames", start_frame=0, host=CODE_HOST, port=CODE_PORT, prompt_json=constructed_json, client='pymunk')
video_tensor = cvt_video_tensor(BASE_DIR+"/gt_frames", w, h)
tracks, visibility = track_objs_in_video(video_tensor, obj_pts, obj_names)
show_in_video(BASE_DIR, video_tensor, "tracked_gt", tracks, visibility)
save_track_json(tracks, 'track', BASE_DIR + "/tracked_gt.json")

obj_pts, w, h, obj_names = locate_objs_in_frame(BASE_DIR+"/veo_frames", start_frame=0, host=CODE_HOST, port=CODE_PORT, prompt_json=constructed_json, client='pymunk')
video_tensor = cvt_video_tensor(BASE_DIR+"/veo_frames", w, h)
tracks, visibility = track_objs_in_video(video_tensor, obj_pts, obj_names)
show_in_video(BASE_DIR, video_tensor, "tracked_veo", tracks, visibility)
save_track_json(tracks, 'track', BASE_DIR + "/tracked_veo.json")