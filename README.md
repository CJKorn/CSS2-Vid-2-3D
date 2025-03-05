# CSS2-Vid-2-3D
A video restoration project based on [RVRT (Recurrent Video Restoration Transformer with Guided Deformable Attention)](https://github.com/JingyunLiang/RVRT), enabling higher quality 3D reconstruction and imaging from low quality input videos.

## Requirements
> - Python >= 3.8
> - [Cmake 3.31.6](https://cmake.org/download/)
> - [Cargo](https://rustup.rs/)
> - [Ninja](https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages)

### Hardware Requirements
Tested with an RTX 5080 16GB and 32GB of ram however with more aggressive tiling and batching this may be lowered at the cost of quality.
12GB of VRAM has been tested with a minimal drop in quality with 64x64 tiles and a batch size of 60 and a resolution of 2160 × 3840.

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
Download all .pth files from https://github.com/JingyunLiang/RVRT/releases
Place them in the folder named ckpt
```

**Inference**
```
Place your video(s) in the input folder
python main.py [Command line arguments]
```

## Command Line Arguments
| Name                     | CMD  | Long CMD           | Description                                             | Default      |
| ------------------------ | ---- | ------------------ | ------------------------------------------------------- | ------------ |
| Input Path               | -I   | --input            | Specifies the input path of the video                   | (CWD)\input  |
| Output Path              | -O   | --output           | Specifies the output path of the frames                 | (CWD)\output |
| Temp Path                | -T   | --temp             | Specifies the temp directory                            | (CWD)\temp   |
| Model Path               | -M   | --model            | Specifies the directory of the models                   | (CWD)\ckpt   |
| Skip Motion Blur Removal | -SMB | --skip-motion-blur | Skips the motion blur removal processing                | Off          |
| Blur Threshold           | -BT  | --blur-threshold   | Sets threshold for blurry frame detection (0.0-1.0)     | 0.5          |
| Keep Similar Frames      | -KSF | --keep-similar     | Keeps frames even if they're similar                    | Off          |
| Similarity Threshold     | -ST  | --similarity       | Sets threshold for frame similarity detection (0.0-1.0) | 0.95         |
| Frame Interval           | -FI  | --frame-interval   | Extract frames at fixed intervals (in frames)           | 1            |
| Time Interval            | -TI  | --time-interval    | Extract frames at fixed time intervals (in seconds)     | 0            |
| Batch Size               | -BS  | --batch-size       | Batch size for processing                               | 0            |
| Export Format            | -EF  | --format           | Format for exported frames (png/jpg/tiff)               | png          |
| Compression Level        | -CL  | --compression      | Compression level for output images (1-10)              | 6            |
| Max Frames               | -MF  | --max-frames       | Maximum number of frames to extract                     | All          |
| Color Correction         | -CC  | --color-correction | Apply color correction across frames for consistency    | Off          |
| EXIF Data                | -ED  | --exif             | Include camera EXIF data in output frames if available  | On           |

## Acknowledgements and License
This project builds upon the RVRT (Recurrent Video Restoration Transformer) framework. We've adapted the codebase for our specific use case of enhancing video quality for 3D reconstruction.

### Original RVRT Citation
```
@article{liang2022rvrt,
    title={Recurrent Video Restoration Transformer with Guided Deformable Attention},
    author={Liang, Jingyun and Fan, Yuchen and Xiang, Xiaoyu and Ranjan, Rakesh and Ilg, Eddy and Green, Simon and Cao, Jiezhang and Zhang, Kai and Timofte, Radu and Van Gool, Luc},
    journal={arXiv preprint arXiv:2206.02146},
    year={2022}
}
```

### License Information
This project is released under the CC-BY-NC license, consistent with the original RVRT project.

The original RVRT codebase incorporates code from several sources with different licenses:

[KAIR](https://github.com/cszn/KAIR) (MIT License)
[BasicSR](https://github.com/xinntao/BasicSR) (Apache 2.0 License)
[Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) (Apache 2.0 License)
[mmediting](https://github.com/open-mmlab/mmediting) (Apache 2.0 License)
Our modifications and additional code for 3D reconstruction purposes maintain compatibility with these license requirements.

For more details on the original implementation, please visit the [RVRT GitHub repository](https://github.com/JingyunLiang/RVRT).