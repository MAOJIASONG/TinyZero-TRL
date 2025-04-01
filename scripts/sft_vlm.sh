export CUDA_VISIBLE_DEVICES="0,1,2,3"


ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/accelerate_configs/zero2.yaml \
src/tinyzero_trl/sft_vlm.py \
--multirun +vlm=qwen2vl_sft_config.yaml