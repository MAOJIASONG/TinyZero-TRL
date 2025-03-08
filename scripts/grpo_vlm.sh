export CUDA_VISIBLE_DEVICES="0,1,2,3"

ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 29501 \
--num_processes=3 \
--config_file recipes/accelerate_configs/zero1.yaml \
src/tinyzero_trl/grpo.py \
--multirun +vlm=qwen2_vl_base-geoqa

# --multirun +math=grpo-deepseek-distill-1.5b-inst,grpo-deepscaler 
# +math=grpo-deepseek-distill-1.5b-inst