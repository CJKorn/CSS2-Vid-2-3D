from Scripts.args import get_validated_args
import cv2
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "12.0"
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
    temp_dir = os.path.join(args.temp, filename)
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
            batch_dir = os.path.join(args.temp, f"{filename}_batch_{batch_count}")
            os.makedirs(batch_dir, exist_ok=True)
            frame_path = os.path.join(batch_dir, f"frame{n}.jpg")
        else:
            frame_path = os.path.join(temp_dir, f"frame{n}.jpg")
        
        if n % args.frame_interval == 0:
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame {n} to {frame_path}")
        
        n += 1
        if corrected_batch_size > 0 and n % corrected_batch_size == 0:
            batch_count += 1
    print("Video processing complete!")

def extract_videos(args):
    if (os.listdir(args.temp) != []):
        cont = input("Error: Temp directory is not empty. Did you want to continue? (May overwrite but not delete existing files) [y/n]: ")
        if cont.lower() != 'y':
            return
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
    # display_settings(args)
    
    extract_videos(args)
    i = 0
    for batch in os.listdir(args.temp):
        batch_path = os.path.join(args.temp, batch)
        i += 1
        print(f"Processing batch {i}: {batch_path}")
        rvrt.infer(args, batch_path)

if __name__ == "__main__":
    main()