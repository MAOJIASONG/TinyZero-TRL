export CUDA_VISIBLE_DEVICES="0,1,2,3"


ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file src/r1-v/configs/zero2.yaml \
src/r1-v/src/open_r1/sft.py \
--multirun +vlm=qwen2vl_sft_config.yaml