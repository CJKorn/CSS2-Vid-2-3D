from Scripts.args import get_validated_args
import cv2
import os
import shutil
os.environ['TORCH_CUDA_ARCH_LIST'] = "12.0" #TODO: Make not bad
import RVRT.main_test_rvrt as rvrt

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
    # os.makedirs(temp_dir, exist_ok=True)
    print(f"Saving frames to: {temp_dir}")
    n = 0
    batch_count = 0
    corrected_batch_size = args.frame_interval * args.batch_size
    while True:
        success, frame = capture.read()
        if not success:
            print("No more frames to read or error reading frame.")
            break
        if corrected_batch_size > 0:
            batch_dir = os.path.join(temp_dir, f"batch_{batch_count}")
            os.makedirs(batch_dir, exist_ok=True)
            frame_path = os.path.join(batch_dir, f"frame{n}.jpg")
        else:
            frame_path = os.path.join(temp_dir, "batch_0", f"frame{n}.jpg")
        
        if n % args.frame_interval == 0:
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame {n} to {frame_path}")
        
        n += 1
        if corrected_batch_size > 0 and n % corrected_batch_size == 0:
            batch_count += 1
    print("Video processing complete!")

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

def main():
    args = get_validated_args()
    display_settings(args)
    check_temp_dir(args)
    output = os.path.join(args.temp, "deblurred")
    extracted_frames = os.path.join(args.temp, "extracted_frames")
    os.makedirs(output, exist_ok=True)
    os.makedirs(extracted_frames, exist_ok=True)
    extract_videos(args)
    os.makedirs(args.output, exist_ok=True)
    i = 0
    for video_dir in os.listdir(extracted_frames):
        video_path = os.path.join(extracted_frames, video_dir)
        if os.path.isdir(video_path):
            for batch in os.listdir(video_path):
                if batch.startswith("batch_"):
                    deblur_path = os.path.join(output, video_dir)
                    os.makedirs(deblur_path, exist_ok=True)
                    batch_path = os.path.join(video_path, batch)
                    i += 1
                    print(f"Processing batch {i}: {batch_path}")
                    print(f"Saving deblurred frames to: {deblur_path}")
                    rvrt.infer(args, batch_path, deblur_path)
    # do upscaling (Uses too much memory)
    # args.task = "002_RVRT_videosr_bi_Vimeo_14frames"
    # args.tile = [4,48,48]
    # upscaled = os.path.join(args.temp, "upscaled")
    # os.makedirs(upscaled, exist_ok=True)
    # for video_dir in os.listdir(output):
    #     video_path = os.path.join(output, video_dir)
    #     if os.path.isdir(video_path):
    #         for batch in os.listdir(video_path):
    #             if batch.startswith("batch_"):
    #                 upscale_path = os.path.join(upscaled, video_dir)
    #                 os.makedirs(upscale_path, exist_ok=True)
    #                 batch_path = os.path.join(video_path, batch)
    #                 print(f"Upscaling batch: {batch_path}")
    #                 print(f"Saving upscaled frames to: {upscale_path}")
    #                 rvrt.infer(args, batch_path, upscale_path)

if __name__ == "__main__":
    main()