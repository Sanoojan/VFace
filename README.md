# VFace

Official implementation of **VFace: A Training-Free Approach for Diffusion-Based Video Face Swapping** (WACV 2026).

[[Paper](https://openaccess.thecvf.com/content/WACV2026/papers/Baliah_VFace_A_Training-Free_Approach_for_Diffusion-Based_Video_Face_Swapping_WACV_2026_paper.pdf)] [[arXiv](https://arxiv.org/abs/2602.07835)]

[Sanoojan Baliah](https://www.linkedin.com/in/sanoojan/), Yohan Abeysinghe, Rusiru Thushara, Khan Muhammad, Abhinav Dhall, Karthik Nandakumar, and Muhammad Haris Khan

## Abstract

> We present VFace, a training-free, plug-and-play method for high-quality face swapping in videos. VFace can be integrated with image-based face-swapping methods built on diffusion models. It introduces Frequency Spectrum Attention Interpolation to preserve key identity characteristics, Target Structure Guidance through plug-and-play attention injection, and Flow-Guided Attention Temporal Smoothening to improve spatiotemporal coherence. VFace requires neither additional training nor video-specific fine-tuning.

## News

- **2025-12-06:** Released VFace with REFace as the image-level face-swapping backbone.

## Installation

The commands below are self-contained; you do not need to clone or configure another repository. The required REFace backbone code is included in the `REFace/` directory.

### 1. Prerequisites

- Linux
- An NVIDIA GPU and a compatible CUDA installation 
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
- Git
- FFmpeg available on `PATH`

The inference scripts call CUDA directly and do not support CPU-only inference without code changes.

### 2. Clone the repository and create the environment

```bash
git clone https://github.com/Sanoojan/VFace.git
cd VFace/REFace

conda create -n VFace python=3.10.13 -y
conda activate VFace
bash setup.sh
```

`setup.sh` installs the pinned Python packages, PyTorch, dlib, CLIP, Taming Transformers, and the evaluation dependencies.

### 3. Download the pretrained models

Install the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

From `VFace/REFace`, download the REFace checkpoint and the three auxiliary models into the paths expected by the code:

```bash
mkdir -p models/REFace/checkpoints

hf download Sanoojan/REFace last.ckpt \
  --local-dir models/REFace/checkpoints

hf download Sanoojan/REFace \
  Other_dependencies/face_parsing/79999_iter.pth \
  Other_dependencies/arcface/model_ir_se50.pth \
  Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat \
  --local-dir .
```

After downloading, verify that these files exist:

```text
REFace/
├── models/REFace/checkpoints/last.ckpt
└── Other_dependencies/
    ├── arcface/model_ir_se50.pth
    ├── DLIB_landmark_det/shape_predictor_68_face_landmarks.dat
    └── face_parsing/79999_iter.pth
```

The first inference run also downloads pretrained CLIP, Stable Diffusion safety-checker, and RAFT optical-flow weights. They are cached automatically.

## Quick start: included example

Run all inference commands from `VFace/REFace` so that the relative model paths resolve correctly.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/VFace_inference_single.py \
  --target_video examples/FaceSwap/Videos/vid.mp4 \
  --src_image examples/FaceSwap/Source/will_smith.jpeg \
  --outdir outputs/example \
  --Base_dir outputs/example/work \
  --config models/REFace/configs/project_ffhq.yaml \
  --ckpt models/REFace/checkpoints/last.ckpt \
  --n_frames 18 \
  --n_samples 6 \
  --scale 3.0 \
  --ddim_steps 50
```

The final swapped video and GIF are written to `outputs/example/`. Intermediate frames, aligned faces, masks, and inversion files are stored under the output and work directories.

## Run on your own video

Prepare:

- one target video containing a visible face; and
- one source image containing the identity to transfer.

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/VFace_inference_single.py \
  --target_video /path/to/target.mp4 \
  --src_image /path/to/source.jpg \
  --outdir outputs/my_swap \
  --Base_dir outputs/my_swap/work \
  --config models/REFace/configs/project_ffhq.yaml \
  --ckpt models/REFace/checkpoints/last.ckpt \
  --n_frames 24 \
  --n_samples 6 \
  --scale 3.0 \
  --ddim_steps 50
```

Important arguments:

| Argument | Meaning | Typical value |
| --- | --- | --- |
| `--target_video` | Target video whose pose, expression, motion, and background are retained | MP4 file |
| `--src_image` | Source identity image | JPG or PNG file |
| `--outdir` | Final frames, MP4, GIF, and model outputs | A new directory per run |
| `--Base_dir` | Preprocessed target frames, crops, masks, and cached transforms | A new work directory per run |
| `--n_frames` | Number of frames processed from the start of the target video | Must not exceed the available frames |
| `--n_samples` | Inference batch size | `6`; reduce if GPU memory is insufficient |
| `--scale` | Classifier-free guidance scale | `3.0` |
| `--ddim_steps` | DDIM sampling steps | `50` |

`--n_frames` should be divisible by `--n_samples`; otherwise the final incomplete batch is skipped. The released scripts encode the output at **10 FPS**, so use a 10 FPS input or convert the input to 10 FPS beforehand when preserving timing is important:

```bash
ffmpeg -i input.mp4 -vf fps=10 -c:v libx264 -pix_fmt yuv420p input_10fps.mp4
```

For best results, use a clear source image with one prominent face and target videos in which the face remains detectable. Large occlusions, extreme poses, multiple faces, or frames without a face may cause alignment failures.

## Batch inference

The batch script pairs each target-video directory with one source-image filename through a YAML file.

### 1. Organize the inputs

```text
inputs/
├── videos/
│   ├── clip_001/
│   │   └── target.mp4
│   └── clip_002/
│       └── target.mp4
└── sources/
    ├── identity_001.jpg
    └── identity_002.jpg
```

Each video subdirectory must contain an MP4 file. Create `inputs/data_matching.yaml`:

```yaml
clip_001: identity_001.jpg
clip_002: identity_002.jpg
```

Alternatively, generate the YAML automatically:

```bash
python generate_config.py \
  --video_base_dir inputs/videos \
  --image_dir inputs/sources \
  --output_yaml_path inputs/data_matching.yaml
```

The generator pairs naturally sorted video-directory names with naturally sorted image filenames. Check the generated YAML before inference to ensure that every pairing is correct.

### 2. Run the batch

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/VFace_inference_batch.py \
  --video_base_dir inputs/videos \
  --image_dir inputs/sources \
  --data_config inputs/data_matching.yaml \
  --output_base_dir outputs/batch/results \
  --Base_dir outputs/batch/work \
  --config models/REFace/configs/project_ffhq.yaml \
  --ckpt models/REFace/checkpoints/last.ckpt \
  --n_frames 24 \
  --n_samples 6 \
  --scale 3.0 \
  --ddim_steps 50
```

All videos in one batch invocation use the same `--n_frames` value. The script skips an item if its output directory already exists.

## Troubleshooting

- **A model file is not found:** confirm the four paths in the model tree above and run the command from `VFace/REFace`.
- **CUDA out of memory:** lower `--n_samples` while keeping it a divisor of `--n_frames`.
- **The last frames are missing:** make `--n_frames` divisible by `--n_samples` and ensure that the target video contains at least `--n_frames` frames.
- **Face alignment fails:** use a video with a clearly visible face and a source image containing one large, unobstructed face.
- **A rerun reuses stale preprocessing:** use fresh `--outdir` and `--Base_dir` paths, or remove the previous run directories after confirming that you no longer need them.
- **The output timing differs from the input:** the released inference scripts write at 10 FPS; convert the input to 10 FPS before running VFace.

## Citation

If you find this work useful, please cite the paper and consider starring the repository.

```bibtex
@inproceedings{baliah2026vface,
  title={VFace: A Training-Free Approach for Diffusion-Based Video Face Swapping},
  author={Baliah, Sanoojan and Abeysinghe, Yohan and Thushara, Rusiru and Muhammad, Khan and Dhall, Abhinav and Nandakumar, Karthik and Khan, Muhammad Haris},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={4315--4324},
  year={2026}
}
```

## Acknowledgements

This implementation builds on [Paint-By-Example](https://github.com/Fantasy-Studio/Paint-by-Example) and [REFace](https://github.com/Sanoojan/REFace).

## License and usage

The repository code is released under the MIT License; see [LICENSE](LICENSE). Components derived from Paint-By-Example remain subject to the CreativeML Open RAIL-M license and its use-based restrictions.

The [released checkpoint](https://huggingface.co/Sanoojan/REFace) is provided under CC BY-NC 4.0 and was trained using CelebAMask-HQ, which is provided for non-commercial research. Accordingly, the checkpoint and its outputs must be used consistently with the dataset and model-license restrictions. Use face-swapping technology responsibly and only with appropriate consent.

## Support

Please open a GitHub issue if you encounter a reproducible problem. Include the command you ran, the complete error message, your operating system, GPU model, CUDA version, and environment details.
