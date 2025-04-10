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
    print(f"Skip motion blur removal: {args.skip_motion_blur}")
    print(f"Blur threshold: {args.blur_threshold}")
    print(f"Keep similar frames: {args.keep_similar}")
    print(f"Similarity threshold: {args.similarity}")
    print(f"Frame interval: {args.frame_interval}")
    print(f"Time interval: {args.time_interval}")
    print(f"Batch size: {args.batch_size}")
    print(f"Upscale factor: {args.upscale}")
    print(f"Export format: {args.format}")
    print(f"Compression level: {args.compression}")
    print(f"Max frames: {args.max_frames}")
    print(f"Color correction: {args.color_correction}")
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
                    # rvrt.infer(args, batch_path, full_output_path)
                    config.dataset_path = batch_path
                    config.save_dir = full_output_path
                    config.custom_path = batch_path
                    fma.test_custom(config, args)

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
    inference(args, extracted_frames, deblurred, config)
    stitched_frames = os.path.join(args.temp, "stitched_frames")
    stitch_frames(args, deblurred, stitched_frames)
    combine_frames_to_video(args, stitched_frames, args.output)
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