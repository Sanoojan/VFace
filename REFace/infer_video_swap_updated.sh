
##### EXPERIMENTAL #####

Base_dir="outputs/VFHQ_test_full"
Experiment_name="23_Final_fft0.8_alpha0.8_steps50"
device=2

CONFIG="models/Paint-by-Example/v5_Two_CLIP_proj_154/checkpoints/project_ffhq.yaml"
DATA_CONFIG="dataset/FaceData/Data/VFHQ-Test/data_matching.yaml"
CKPT="models/Paint-by-Example/v5_Two_CLIP_proj_154/checkpoints/last.ckpt"
# CKPT="models/Paint-by-Example/V5_without_FSA_154/checkpoints/epoch=000019.ckpt"

video_base_dir="/egr/research-sprintai/baliahsa/mbz-back/Paint_for_swap/dataset/FaceData/Data/VFHQ-Test/GT/Vid_Interval1_512x512_LANCZOS4"
image_dir="/egr/research-sprintai/baliahsa/mbz-back/Paint_for_swap/dataset/FaceData/Data/VFHQ-Test/Celeb_Source"
DATA_CONFIG="${Base_dir}/${Experiment_name}/results_new/data_matching.yaml"

if [ ! -d "${Base_dir}/${Experiment_name}/results_new" ]; then
    mkdir -p "${Base_dir}/${Experiment_name}/results_new"
fi

current_time=$(date +"%Y%m%d_%H%M%S")

python generate_config.py \
    --video_base_dir "${video_base_dir}" \
    --image_dir "${image_dir}" \
    --output_yaml_path "${DATA_CONFIG}"


CUDA_VISIBLE_DEVICES=${device} python scripts/inference_video.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --data_config "${DATA_CONFIG}" \
    --Base_dir "${Base_dir}/${Experiment_name}/results_video" \
    --video_base_dir "${video_base_dir}" \
    --image_dir "${image_dir}" \
    --output_base_dir "${Base_dir}/${Experiment_name}/results_new" \
    --scale 3.0 \
    --ddim_steps 50 

    

