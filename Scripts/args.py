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
    
    parser.add_argument('-DS', '--downscale',
        action='store_true',
        help='Downscales the frames to 1/4 of the original size after FMA-NET 4x upscale')

    parser.add_argument('-KSF', '--keep-similar', 
        action='store_true',
        help='Keeps frames even if they\'re similar')
    
    parser.add_argument('-TI', '--time-interval', 
        type=float, 
        default=0,
        help='Extract frames at fixed time intervals (in seconds)')
    
    parser.add_argument('-BS', '--batch-size',
        type=int,
        default=0,
        help='Batch size for processing')
    
    parser.add_argument('-MF', '--max-frames', 
        default='All',
        help='Maximum number of frames to extract')
    
    parser.add_argument('-GP', '--greedy-percent', 
        type=float,
        default=0.15,
        help='Percent of frames to select from greedy selection')

    parser.add_argument('-CP', '--cluster-percent', 
        type=float,
        default=0.25,
        help='Percent of frames to select from clusters')
    
    parser.add_argument('-DP' '--dumb_percent',
        type=float,
        default=0.1,
        help='Percent of frames to select from dumb selection')

    parser.add_argument('--task',
        type=str,
        default='FMANet',
        help='tasks: 001 to 006')
    
    parser.add_argument('--tiles',
        type=int,
        default=1,
        help='Number of tiles per frame for FMA-NET (must be a perfect square: e.g. 1, 4, 9, or 16)')
    
    parser.add_argument('--tile',
        type=int, nargs='+',  
        default=[0,256,256],
        help='Tile size, [0,0,0] for no tile during testing (for RVRT)')
    
    parser.add_argument('--tile_overlap',
        type=int,
        nargs='+',
        default=[2,64,64],
        help='Overlapping of different tiles for RVRT')
    

    parser.add_argument('-FI', '--frame-interval', 
        type=int,  
        default=1,
        help='Extract frames at fixed intervals (in frames)')

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
    return True

def get_validated_args():
    """Parse and validate arguments, returning them if valid."""
    args = parse_arguments()
    if not validate_args(args):
        sys.exit(1)
    return args