# CSS2-Vid-2-3D
A video restoration project based on [RVRT](https://github.com/JingyunLiang/RVRT) and [FMA-Net](https://github.com/KAIST-VICLab/FMA-Net), enabling higher quality 3D reconstruction and imaging from low quality input videos.

## Requirements
> - Python >= 3.8
> - [Cmake 3.31.6](https://cmake.org/download/)
> - [Cargo](https://rustup.rs/)
> - [Ninja](https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages)

### Hardware Requirements
Tested with an RTX 5080 16GB and 32GB of ram however with more aggressive tiling and batching this may be lowered at the cost of quality.

## Getting Started
**Clone the Repository**
```shell
git clone https://github.com/CJKorn/CSS2-Vid-2-3D
cd CSS2-Vid-2-3D
```

**Set up Python environment**
```shell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Download Model Files**
```
Download all .pth files from https://github.com/KAIST-VICLab/FMA-Net?tab=readme-ov-file
(RVRT files in https://github.com/JingyunLiang/RVRT/releases if you want to use RVRT)
Place them in the folder named ckpt
```

**Inference**
```
Place your video(s) in the input folder (may not work with multiple)
python main.py [Command line arguments]
```

**Using RVRT**

In out testing RVRT has shown worse performance than FMA-Net, if you wish to use it add an RVRT task to the command line. (example: --task 004_RVRT_videodeblurring_DVD_16frames)

## Command Line Arguments
| Name                       | CMD                                  | Description                                                               | Default                                           |
| -------------------------- | ------------------------------------ | ------------------------------------------------------------------------- | ------------------------------------------------- |
| Input Path                 | -I, --input                          | Specifies the input path of the video                                     | (CWD)\input                                       |
| Output Path                | -O, --output                         | Specifies the output path of the frames                                   | (CWD)\output                                      |
| Temp Path                  | -T, --temp                           | Specifies the temp directory                                              | (CWD)\temp                                        |
| Model Path                 | -M, --model                          | Specifies the directory of the models                                     | (CWD)\ckpt                                        |
| Skip Motion-Blur Removal   | -SMB, --skip-motion-blur             | Skips the motion blur removal processing                                  | False                                             |
| Keep Similar Frames        | -KSF, --keep-similar                 | Keeps frames even if they're similar                                       | False                                             |
| Time Interval              | -TI, --time-interval                 | Extract frames at fixed time intervals (in seconds)                       | 0.0                                               |
| Batch Size                 | -BS, --batch-size                    | Batch size for processing                                                 | 0                                                 |
| Max Frames                 | -MF, --max-frames                    | Maximum number of frames to extract                                        | All                                               |
| Greedy Selection Percent   | -GP, --greedy-percent                | Percent of frames to select from greedy selection                         | 0.15                                              |
| Cluster Selection Percent  | -CP, --cluster-percent               | Percent of frames to select from clusters                                 | 0.1                                               |
| Task                       | --task                               | RVRT task name (e.g. 004_RVRT_videodeblurring_DVD_16frames)               | 004_RVRT_videodeblurring_DVD_16frames             |
| Number of Tiles            | --tiles                              | Number of tiles per frame for FMA-NET (must be a perfect square)          | 1                                                 |
| Tile Size                  | --tile                               | Tile size [depth,height,width], [0,0,0] for no tiling during testing       | [0,256,256]                                       |
| Tile Overlap               | --tile_overlap                       | Overlap of different tiles for RVRT                                        | [2,64,64]                                         |

## Acknowledgements and License
This project builds upon RVRT (Recurrent Video Restoration Transformer) and FMA-Net. We've adapted the codebase for our specific use case of enhancing video quality for 3D reconstruction.

### Original RVRT Citation
```
@article{liang2022rvrt,
    title={Recurrent Video Restoration Transformer with Guided Deformable Attention},
    author={Liang, Jingyun and Fan, Yuchen and Xiang, Xiaoyu and Ranjan, Rakesh and Ilg, Eddy and Green, Simon and Cao, Jiezhang and Zhang, Kai and Timofte, Radu and Van Gool, Luc},
    journal={arXiv preprint arXiv:2206.02146},
    year={2022}
}
```

### Original FMA-Net Citation
```
@InProceedings{Youk_2024_CVPR,
    author    = {Youk, Geunhyuk and Oh, Jihyong and Kim, Munchurl},
    title     = {FMA-Net: Flow-Guided Dynamic Filtering and Iterative Feature Refinement with Multi-Attention for Joint Video Super-Resolution and Deblurring},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {44-55}
}
```

## License

This repository combines:
- RVRT (CC BY-NC 4.0)  
- BasicSR, Video-Swin-Transformer, mmediting (Apache 2.0)  
- FMA-Net (MIT)

**The entire work is being distributed under CC BY-NC 4.0**
