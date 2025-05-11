import os
import cv2
import math
import torch
import numpy as np
import random
import shutil
from tqdm import tqdm
import ruptures as rpt
from skimage.metrics import structural_similarity as ssim
import concurrent.futures

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
    grid_dim = int(math.sqrt(args.tiles))
    if grid_dim * grid_dim != args.tiles:
        print("Error: args.tiles must be a perfect square (e.g. 4, 9, or 16).")
        return
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
            height, width = frame.shape[:2]
            tile_height = height // grid_dim
            tile_width = width // grid_dim
            for i in range(grid_dim):
                for j in range(grid_dim):
                    tile_index = i * grid_dim + j
                    y1 = i * tile_height
                    x1 = j * tile_width
                    y2 = (i + 1) * tile_height if i < grid_dim - 1 else height
                    x2 = (j + 1) * tile_width if j < grid_dim - 1 else width
                    tile = frame[y1:y2, x1:x2]
                    tile_path = os.path.join(current_batch_dirs[tile_index], f"{n}.png")
                    cv2.imwrite(tile_path, tile)
                    progress_bar.update(1)
        n += 1
    progress_bar.close()
    print("Video processing complete!")

def stitch_frames(args, extracted_dir, stitched_dir):
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
    if os.listdir(args.temp) != []:
        cont = input("Error: Temp directory is not empty. Did you want to delete its contents? (If no will use the files in temp for inference) [y/n]: ")
        if cont.lower() != 'y':
            return
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

        for frame_path in all_frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                out_writer.write(frame)
            else:
                print(f"Error reading frame: {frame_path}")
        out_writer.release()
        print(f"Created video: {output_video_path}")

def score_BRISQUE(root_folder): #Unused
    from brisque import BRISQUE
    import numpy as np
    from PIL import Image
    import os

    brisque_obj = BRISQUE(url=False)

    for folder, _, files in os.walk(root_folder):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            file_path = os.path.join(folder, filename)
            try:
                img = Image.open(file_path)
                ndarray = np.asarray(img)
                score = brisque_obj.score(img=ndarray)
                print(f"Image: {file_path}, BRISQUE Score: {score}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

def find_image_files(root_folder):
    all_files = []
    for folder, _, files in os.walk(root_folder):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg']:
                if os.path.splitext(filename)[0].isdigit():
                    file_path = os.path.join(folder, filename)
                    all_files.append(file_path)
    all_files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

    return all_files

def score_laplacian(root_folder):
    all_files = find_image_files(root_folder)
    scores = []
    for file_path in tqdm(all_files, desc="Analyzing images (Laplacian)", unit="image"):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        scores.append((file_path, laplacian_var))

    return scores

def _compare_pair(paths):
    path1, path2 = paths
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    ssim_score = 0.0

    if img1 is None:
        return (path1, path2, ssim_score)
    if img2 is None:
        return (path1, path2, ssim_score)

    try:
        ssim_score, _ = ssim(img1, img2, full=True, data_range=img1.max() - img1.min())
    except ValueError as ve:
         print(f"ValueError between {os.path.basename(path1)} ({img1.shape}) and {os.path.basename(path2)} ({img2.shape}): {ve}")
         ssim_score = 0.0
    except Exception as e:
         print(f"Error calculating SSIM between {os.path.basename(path1)} and {os.path.basename(path2)}: {e}")
         ssim_score = 0.0
    ssim_score = max(0, ssim_score)

    return (path1, path2, ssim_score)

def _calculate_weighted_ssim_for_frame(args):
    index_n, all_files, nplus = args
    frame_n_path = all_files[index_n]
    total_weighted_ssim = 0.0
    total_weight = 0.0
    for k in range(1, nplus + 1):
        index_nk = index_n + k
        if index_nk < len(all_files):
            frame_nk_path = all_files[index_nk]
            # Calculate SSIM for the pair (n, n+k)
            _, _, ssim_score = _compare_pair((frame_n_path, frame_nk_path))
            weight = float(nplus - k + 1)

            total_weighted_ssim += ssim_score * weight
            total_weight += weight
        else:
            break
    weighted_average_ssim = total_weighted_ssim / total_weight if total_weight > 0 else 0
    return (frame_n_path, weighted_average_ssim)


def similarity(root_folder, nplus, max_workers=6):
    all_files = find_image_files(root_folder)
    num_files = len(all_files)

    print(f"Comparing frames for similarity using weighted SSIM (nplus={nplus})...")
    tasks_args = [(i, all_files, nplus) for i in range(num_files)]

    similarity_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(_calculate_weighted_ssim_for_frame, tasks_args)
        similarity_results = list(tqdm(results_iterator, total=len(tasks_args), desc=f"Calculating weighted SSIM (nplus={nplus})", unit="frame"))

    # similarity_results will be a list of tuples: [(frame_path, weighted_ssim_score), ...]
    return similarity_results

def cluster(root_folder, similarity_scores):
    all_files = find_image_files(root_folder)
    num_files = len(all_files)
    scores_only_list = []  # Fixed indentation here
    for _, score in similarity_scores:
        scores_only_list.append(float(score))

    scores_only = np.array(scores_only_list).reshape(-1, 1)
    n_scores = scores_only.shape[0]

    change_points_indices = []

    print(f"Performing change point detection on {n_scores} similarity scores...")
    # model="rbf" is often good for changes in mean and variance.
    # model="l2" detects changes in the mean.
    algo = rpt.Pelt(model="rbf").fit(scores_only)

    # Higher penalty = fewer change points.
    # Common heuristics: 3*log(n), 2*log(n). BIC/AIC can also be used.
    penalty_value = 0.5 * np.log(n_scores)
    print(f"Using penalty value: {penalty_value:.2f}")

    try:
        predicted_bkps = algo.predict(pen=penalty_value)
        print(f"Ruptures detected breakpoints (end of segments): {predicted_bkps}")
        if predicted_bkps and predicted_bkps[-1] == n_scores:
                change_points_indices = [cp - 1 for cp in predicted_bkps[:-1]]
        else:
                change_points_indices = [cp - 1 for cp in predicted_bkps if cp < n_scores]

        print(f"Detected change point indices (positions *before* change): {change_points_indices}")

    except Exception as e:
        print(f"Error during change point detection: {e}")
        change_points_indices = []
    return change_points_indices

def select_frames(args, root_folder, scores, change_points_indices):
    # I would do this with a hashmap but python is stupuid
    all_files = find_image_files(root_folder)
    selected_frames = []
    cluster_selection_count = len(scores) * args.cluster_percent / len(change_points_indices)
    num_frames_to_select = max(1, int(len(scores) * args.greedy_percent))
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i][1], reverse=True)
    selected_indices = sorted_indices[:num_frames_to_select]
    selected_frames = [ scores[i][0] for i in selected_indices ]
    current_cluster = []
    # Then do clustering stuff
    for i in range(len(scores)):
        current_cluster.append(scores[i])
        if i in change_points_indices:
            select_count = min(cluster_selection_count, len(current_cluster))
            sorted_indices = sorted(
                range(len(current_cluster)),
                key=lambda j: current_cluster[j][1],
                reverse=True
            )
            count = 0
            for j in sorted_indices:
                frame_path = current_cluster[j][0]
                if frame_path in selected_frames:
                    continue
                selected_frames.append(frame_path)
                count += 1
                if count >= select_count:
                    break
            current_cluster = []
    return selected_frames