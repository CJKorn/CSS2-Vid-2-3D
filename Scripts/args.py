import argparse
import os
import sys

def parse_arguments():
    """Parse command line arguments for the photogrammetry processing tool."""
    
    parser = argparse.ArgumentParser(
        description='Tool for processing videos for photogrammetry, NERF generation and Gaussian splatting.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required paths
    parser.add_argument('-I', '--input', 
        default=os.path.join(os.getcwd(), 'input'),
        help='Specifies the input path of the video')
    
    parser.add_argument('-O', '--output', 
        default=os.path.join(os.getcwd(), 'output'),
        help='Specifies the output path of the frames')
    
    parser.add_argument('-T', '--temp', 
        default=os.path.join(os.getcwd(), 'temp'),
        help='Specifies the temp directory')
    
    parser.add_argument('-M', '--model',
        type=str,
        default=os.path.join(os.getcwd(), 'ckpt'),
        help='Specifies the directory of the models')

    # Processing options
    parser.add_argument('-SMB', '--skip-motion-blur', 
        action='store_true',
        help='Skips the motion blur removal processing')
    
    parser.add_argument('-BT', '--blur-threshold', 
        type=float, 
        default=0.5,
        help='Sets threshold for blurry frame detection (0.0-1.0)')
    
    parser.add_argument('-KSF', '--keep-similar', 
        action='store_true',
        help='Keeps frames even if they\'re similar')
    
    parser.add_argument('-ST', '--similarity', 
        type=float, 
        default=0.95,
        help='Sets threshold for frame similarity detection (0.0-1.0)')
    
    parser.add_argument('-FI', '--frame-interval', 
        type=int,  
        default=1,
        help='Extract frames at fixed intervals (in frames)')
    
    parser.add_argument('-TI', '--time-interval', 
        type=float, 
        default=0,
        help='Extract frames at fixed time intervals (in seconds)')
    
    parser.add_argument('-BS', '--batch-size',
        type=int,
        default=0,
        help='Batch size for processing')
    
    parser.add_argument('-UF', '--upscale', 
        type=float, 
        default=1.0,
        help='Factor by which to upscale output frames (1.0-4.0)')
    
    parser.add_argument('-EF', '--format', 
        choices=['png', 'jpg', 'tiff'], 
        default='png',
        help='Format for exported frames')
    
    parser.add_argument('-CL', '--compression', 
        type=int, 
        default=6,
        choices=range(1, 11),
        help='Compression level for output images (1-10)')
    
    parser.add_argument('-MF', '--max-frames', 
        default='All',
        help='Maximum number of frames to extract')
    
    parser.add_argument('-CC', '--color-correction', 
        action='store_true',
        help='Apply color correction across frames for consistency')
    
    parser.add_argument('--task',
        type=str,
        default='004_RVRT_videodeblurring_DVD_16frames',
        help='tasks: 001 to 006')
    
    parser.add_argument('--tiles',
        type=int,
        default=1,
        help='Number of tiles per frame (must be a perfect square: e.g. 1, 4, 9, or 16)')
    
    parser.add_argument('--sigma',
        type=int,
        default=0,
        help='noise level for denoising: 10, 20, 30, 40, 50')
    
    parser.add_argument('--tile',
        type=int, nargs='+',  
        default=[0,256,256],
        help='Tile size, [0,0,0] for no tile during testing (testing as a whole)')
    
    parser.add_argument('--tile_overlap',
        type=int,
        nargs='+',
        default=[2,64,64],
        help='Overlapping of different tiles')
    
    parser.add_argument('--num_workers',
        type=int,
        default=16,
        help='number of workers in data loading')
    
    parser.add_argument('--save_result',
        action='store_true',
        help='save resulting image')
    
    parser.add_argument('--scan_subdirectories',
        action='store_true',
        help='Scan subdirectories for frames instead of only the top directory')
    
    return parser.parse_args()

def validate_args(args):

    #Remove whitespaces from task, (Caused trouble when copypasting)
    args.task = args.task.replace(" ", "")

    # Check if input file/directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist.")
        return False
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        try:
            os.makedirs(args.output)
            print(f"Created output directory: {args.output}")
        except OSError as e:
            print(f"Error creating output directory: {e}")
            return False
    
    # Create temp directory if it doesn't exist
    if not os.path.exists(args.temp):
        try:
            os.makedirs(args.temp)
            print(f"Created temp directory: {args.temp}")
        except OSError as e:
            print(f"Error creating temp directory: {e}")
            return False
    
    # Validate numeric ranges
    if not 0.0 <= args.blur_threshold <= 1.0:
        print(f"Error: Blur threshold must be between 0.0 and 1.0")
        return False
    
    if not 0.0 <= args.similarity <= 1.0:
        print(f"Error: Similarity threshold must be between 0.0 and 1.0")
        return False
    
    if not 1.0 <= args.upscale <= 4.0:
        print(f"Error: Upscale factor must be between 1.0 and 4.0")
        return False
    
    # Handle max frames 
    if args.max_frames != 'All':
        try:
            args.max_frames = int(args.max_frames)
            if args.max_frames <= 0:
                print(f"Error: Max frames must be greater than 0")
                return False
        except ValueError:
            print(f"Error: Max frames must be 'All' or a positive integer")
            return False
    
    return True

def get_validated_args():
    """Parse and validate arguments, returning them if valid."""
    args = parse_arguments()
    if not validate_args(args):
        sys.exit(1)
    return args