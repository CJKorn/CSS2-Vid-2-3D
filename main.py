from Scripts.args import get_validated_args
import cv2
import math
import torch
import numpy as np
import random
import os
import shutil
from tqdm import tqdm
from processing import extract_videos, stitch_frames, combine_frames_to_video, check_temp_dir
import processing
import matplotlib.pyplot as plt

if torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST_SET"):
    # Gets capability for the current device
    cap = torch.cuda.get_device_capability(0)
    arch = f"{cap[0]}.{cap[1]}"
    os.environ['TORCH_CUDA_ARCH_LIST'] = arch
    os.environ["TORCH_CUDA_ARCH_LIST_SET"] = "1"
    print(f"Automatically set TORCH_CUDA_ARCH_LIST to {arch}")
elif not torch.cuda.is_available():
    print("CUDA not available; default settings used")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

#RVRT    
import RVRT.main_test_rvrt as rvrt

#FMANet
import FMANet.main_infer as fma
import FMANet.config as fma_config

def display_settings(args):
    print(f"Input path: {args.input}")
    print(f"Output path: {args.output}")
    print(f"Temp directory: {args.temp}")
    print(f"Model directory: {args.model}")
    print(f"Skip motion blur removal: {args.skip_motion_blur}")
    print(f"Keep similar frames: {args.keep_similar}")
    print(f"Time interval: {args.time_interval}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max frames: {args.max_frames}")
    print(f"Greedy percent: {args.greedy_percent}")
    print(f"Cluster percent: {args.cluster_percent}")
    print(f"Task: {args.task}")
    print(f"Tiles: {args.tiles}")
    print(f"Tile size: {args.tile}")
    print(f"Tile overlap: {args.tile_overlap}")
    print("=============================\n")

def inference(args, input_path, output_path, config):
    i = 0
    for video_dir in os.listdir(input_path):
        video_path = os.path.join(input_path, video_dir)
        if os.path.isdir(video_path):
            for batch in os.listdir(video_path):
                if batch.startswith("batch_"):
                    full_output_path = os.path.join(output_path, video_dir)
                    os.makedirs(full_output_path, exist_ok=True)
                    # batch_path = video_path
                    batch_path = os.path.join(video_path, batch)
                    i += 1
                    print(f"Processing batch {i}: {batch_path}")
                    print(f"Saving deblurred frames to: {full_output_path}")
                    if (args.task == "FMANet"):
                        # fma.test_custom(config, args)
                        config.dataset_path = batch_path
                        config.save_dir = full_output_path
                        config.custom_path = batch_path
                        fma.test_custom(config, args)
                    else:
                        rvrt.infer(args, batch_path, full_output_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def plot_similarity_scores(frame_indices, scores_only, change_point_indices, plot_save_path):
    plt.figure(figsize=(15, 7))
    plt.plot(frame_indices, scores_only, marker='o', linestyle='-', label='Similarity Score')
    plt.title('Frame Similarity Scores with Change Points')
    plt.xlabel('Frame Index')
    plt.ylabel('Weighted SSIM Score')
    plt.grid(True)

    valid_change_points = [idx for idx in change_point_indices if 0 <= idx < len(frame_indices)]
    if valid_change_points:
        min_score = min(scores_only) if scores_only else 0
        max_score = max(scores_only) if scores_only else 1
        
        plt.vlines(valid_change_points, ymin=min_score, ymax=max_score, color='red', linestyle='--', linewidth=2, label='Detected Change Points')
        valid_scores_at_change_points = [scores_only[i] for i in valid_change_points if i < len(scores_only)]
        valid_indices_at_change_points = [frame_indices[i] for i in valid_change_points if i < len(frame_indices) and i < len(scores_only)]

        if valid_indices_at_change_points and valid_scores_at_change_points:
            plt.plot(valid_indices_at_change_points, valid_scores_at_change_points, 'rx', markersize=10, label='_nolegend_')
            
    plt.legend()

    try:
        plt.savefig(plot_save_path, dpi=300)
        print(f"Similarity plot saved to: {plot_save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()

def main():
    args = get_validated_args()
    display_settings(args)

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FMANet", "experiment.cfg")
    config = fma_config.Config(config_path)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    check_temp_dir(args)
    deblurred = os.path.join(args.temp, "deblurred")
    extracted_frames = os.path.join(args.temp, "extracted_frames")
    os.makedirs(deblurred, exist_ok=True)
    os.makedirs(extracted_frames, exist_ok=True)
    extract_videos(args)
    os.makedirs(args.output, exist_ok=True)
    if args.skip_motion_blur:
        print("Skipping motion blur removal")
        shutil.copytree(args.input, deblurred, dirs_exist_ok=True)
    else:
        print("Removing motion blur")
        inference(args, extracted_frames, deblurred, config)
    stitched_frames = os.path.join(args.temp, "stitched_frames")
    stitch_frames(args, deblurred, stitched_frames)

    # If keep_similar is set, copy all stitched frames and exit early
    if args.keep_similar:
        print("Keep similar frames enabled: copying all stitched frames to output")
        for root, _, files in os.walk(stitched_frames):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, stitched_frames)
                dest_path = os.path.join(args.output, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(src_path, dest_path)
        return
    scores = processing.score_laplacian(stitched_frames)
    similarity_scores = processing.similarity(stitched_frames, 3) # higher is better but slower and with diminishing returns, 3 is usually good enough
    scores_only = [score for _, score in similarity_scores]
    frame_indices = range(len(scores_only))
    change_point_indices = processing.cluster(stitched_frames, similarity_scores)

    plot_save_path = os.path.join(args.output, "similarity_scores_plot_with_changes.png")
    plot_similarity_scores(frame_indices, scores_only, change_point_indices, plot_save_path)

    for i, (frame_path, score) in enumerate(similarity_scores):
        print(f"Frame {os.path.basename(frame_path)} (Index {i}): Weighted SSIM Score = {score}")
    print(f"Detected change point indices: {change_point_indices}")

    selected_frames = processing.select_frames(args, stitched_frames, scores, change_point_indices)
    for frame_path in selected_frames:
        rel_path = os.path.relpath(frame_path, stitched_frames)
        video_folder = rel_path.split(os.sep)[0]
        output_subdir = os.path.join(args.output, video_folder)
        os.makedirs(output_subdir, exist_ok=True)
        dest_frame = os.path.join(output_subdir, os.path.basename(frame_path))
        if (not (args.downscale)):
            shutil.copy(frame_path, dest_frame)
        else:
            img = cv2.imread(frame_path)
            h, w = img.shape[:2]
            new_w, new_h = max(1, w // 4), max(1, h // 4)
            downscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dest_frame, downscaled)

    # combine_frames_to_video(args, stitched_frames, args.output)
    # do upscaling (Uses too much memory)
    # args.task = "002_RVRT_videosr_bi_Vimeo_14frames"
    # args.tile = [6,64,64]
    # args.tile_overlap = [2,20,20]
    # args.num_workers = 1
    # upscaled = os.path.join(args.temp, "upscaled")
    # os.makedirs(upscaled, exist_ok=True)
    # inference(args, deblurred, upscaled)

if __name__ == "__main__":
    main()