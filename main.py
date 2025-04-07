from Scripts.args import get_validated_args
import cv2
import math
import torch
import numpy as np
import random
import os
import shutil
from tqdm import tqdm
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

def extract_frames(args, file):
    print(f"Processing video: {file}")
    filename = os.path.basename(file)
    capture = cv2.VideoCapture(file)
    if not capture.isOpened():
        print("Error: Could not open video.")
        return
    temp_dir = os.path.join(args.temp, "extracted_frames", filename)
    print(f"Saving frames to: {temp_dir}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = (total_frames // args.frame_interval) if args.frame_interval > 0 else total_frames
    total_tiles = frame_count * args.tiles
    progress_bar = tqdm(total=total_tiles, desc="Extracting tiles", unit="tile")

    n = 0
    batch_count = 0
    corrected_batch_size = args.frame_interval * args.batch_size if args.batch_size > 0 else 0

    # Determine grid dimensions from args.tiles
    grid_dim = int(math.sqrt(args.tiles))
    if grid_dim * grid_dim != args.tiles:
        print("Error: args.tiles must be a perfect square (e.g. 4, 9, or 16).")
        return

    # Hold folder paths for current batch (keys: tile index)
    current_batch_dirs = {}

    while True:
        success, frame = capture.read()
        if not success:
            print("No more frames to read or error reading frame.")
            break

        if n % args.frame_interval == 0:
            # (Re)initialize batch directories when starting a new batch.
            if corrected_batch_size > 0 and (n % corrected_batch_size == 0):
                batch_count = n // corrected_batch_size
                current_batch_dirs = {}
                for tile_index in range(args.tiles):
                    tile_folder = os.path.join(temp_dir, f"batch_{batch_count}_tile_{tile_index}")
                    os.makedirs(tile_folder, exist_ok=True)
                    current_batch_dirs[tile_index] = tile_folder
            elif corrected_batch_size == 0 and not current_batch_dirs:
                # If no batching is defined, default to a single batch (batch_0).
                for tile_index in range(args.tiles):
                    tile_folder = os.path.join(temp_dir, f"batch_0_tile_{tile_index}")
                    os.makedirs(tile_folder, exist_ok=True)
                    current_batch_dirs[tile_index] = tile_folder

            # Get frame dimensions and calculate tile boundaries.
            height, width = frame.shape[:2]
            tile_height = height // grid_dim
            tile_width = width // grid_dim

            # Slice the frame into tiles and write each tile to its batch folder.
            for i in range(grid_dim):
                for j in range(grid_dim):
                    tile_index = i * grid_dim + j
                    y1 = i * tile_height
                    x1 = j * tile_width
                    # For the last row/column, include any remainder.
                    y2 = (i + 1) * tile_height if i < grid_dim - 1 else height
                    x2 = (j + 1) * tile_width if j < grid_dim - 1 else width
                    tile = frame[y1:y2, x1:x2]
                    tile_path = os.path.join(current_batch_dirs[tile_index], f"{n}.png")
                    cv2.imwrite(tile_path, tile)
                    progress_bar.update(1)  # <-- Update progress bar instead of printing
        n += 1
    progress_bar.close()  # <-- Close the progress bar
    print("Video processing complete!")

def stitch_frames(args, extracted_dir, stitched_dir):
    """
    Reassemble deblurred tiles back into full frames.
    This version accounts for the extra "test" folder in the deblurred directory.
    Expected deblurred structure:
      deblurred/
        <video_name>/
          test/                  <-- optional extra folder
             batch_<n>_tile_<tile_index>/
                <frame_number>.png
    The stitched full frames are saved under stitched_dir/<video_name>/batch_<n>/.
    """
    import cv2
    import math

    os.makedirs(stitched_dir, exist_ok=True)

    # Process each video folder in the deblurred directory.
    for video in os.listdir(extracted_dir):
        video_path = os.path.join(extracted_dir, video)
        if not os.path.isdir(video_path):
            continue

        base_folder = os.path.join(video_path, "test")
        if not os.path.isdir(base_folder):
            base_folder = video_path

        batch_groups = {}
        for folder in os.listdir(base_folder):
            if folder.startswith("batch_") and "_tile_" in folder:
                parts = folder.split('_')
                if len(parts) < 4:
                    continue
                batch_number = parts[1]
                try:
                    tile_index = int(parts[3])
                except ValueError:
                    continue
                tile_images_dir = os.path.join(base_folder, folder)
                batch_groups.setdefault(batch_number, {})[tile_index] = tile_images_dir

        for batch_number, tile_dirs in batch_groups.items():
            if len(tile_dirs) != args.tiles:
                print(f"Warning: batch {batch_number} in video {video} is missing some tiles.")
                continue

            grid_dim = int(math.sqrt(args.tiles))
            tile0_dir = tile_dirs[0]
            frame_files = sorted(
                [f for f in os.listdir(tile0_dir) if f.endswith(".png")],
                key=lambda f: int(os.path.splitext(f)[0])
            )

            out_batch_dir = os.path.join(stitched_dir, video, f"batch_{batch_number}")
            os.makedirs(out_batch_dir, exist_ok=True)

            for frame_file in frame_files:
                rows = []
                for i in range(grid_dim):
                    row_tiles = []
                    for j in range(grid_dim):
                        tile_index = i * grid_dim + j
                        tile_path = os.path.join(tile_dirs[tile_index], frame_file)
                        tile = cv2.imread(tile_path)
                        if tile is None:
                            print(f"Error reading tile: {tile_path}")
                            continue
                        row_tiles.append(tile)
                    if row_tiles:
                        row_img = cv2.hconcat(row_tiles)
                        rows.append(row_img)
                if rows:
                    full_frame = cv2.vconcat(rows)
                    out_frame_path = os.path.join(out_batch_dir, frame_file)
                    cv2.imwrite(out_frame_path, full_frame)
    print("Stitching complete!")

def check_temp_dir(args):
    if (os.listdir(args.temp) != []):
        cont = input("Error: Temp directory is not empty. Did you want to delete its contents? (If no will use the files in temp for inference) [y/n]: ")
        if cont.lower() != 'y':
            return
        # path = os.path.join(args.temp, "extracted_frames")
        path = args.temp
        os.makedirs(path, exist_ok=True)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def extract_videos(args):
    print(f"Reading video(s) from: {args.input}")
    if os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.endswith(".mp4"):
                extract_frames(args, os.path.join(args.input, file))
    else:
        extract_frames(args, args.input)
    print("Video processing complete!")

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

def combine_frames_to_video(args, stitched_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    for video in os.listdir(stitched_dir):
        video_folder_path = os.path.join(stitched_dir, video)
        if not os.path.isdir(video_folder_path):
            continue

        batch_folders = []
        for folder in os.listdir(video_folder_path):
            if folder.startswith("batch_"):
                try:
                    batch_number = int(folder.split('_')[1])
                    batch_folders.append((batch_number, folder))
                except ValueError:
                    continue
        batch_folders = sorted(batch_folders, key=lambda x: x[0])

        all_frame_paths = []
        for _, batch_folder in batch_folders:
            batch_folder_path = os.path.join(video_folder_path, batch_folder)
            frames = sorted(
                [f for f in os.listdir(batch_folder_path) if f.endswith(".png")],
                key=lambda f: int(os.path.splitext(f)[0])
            )
            for f in frames:
                all_frame_paths.append(os.path.join(batch_folder_path, f))
        if not all_frame_paths:
            print(f"No frames found for video: {video}")
            continue

        first_frame = cv2.imread(all_frame_paths[0])
        if first_frame is None:
            print(f"Error reading frame: {all_frame_paths[0]}")
            continue
        height, width, _ = first_frame.shape

        output_video_path = os.path.join(output_dir, f"{video}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        orig_video_path = os.path.join(args.input, video)
        if not os.path.isfile(orig_video_path):
            orig_video_path = os.path.join(args.input, video + '.mp4')
        capture = cv2.VideoCapture(orig_video_path)
        if capture.isOpened():
            fps = capture.get(cv2.CAP_PROP_FPS)
            capture.release()
        else:
            fps = 25
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Write each frame into the video.
        for frame_path in all_frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                out_writer.write(frame)
            else:
                print(f"Error reading frame: {frame_path}")
        out_writer.release()
        print(f"Created video: {output_video_path}")

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