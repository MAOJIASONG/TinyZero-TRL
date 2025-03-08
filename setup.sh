# Install the packages in r1-v .
pip install -e ".[dev]"

# Addtional modules
pip install -r requirement.txt
pip install flash-attn --no-build-isolation

# vLLM support 
pip install vllm==0.7.2