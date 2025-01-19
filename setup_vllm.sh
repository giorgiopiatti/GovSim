conda create -n GovComVLLMv2  python=3.11.5 -y
conda activate GovComVLLMv2
# conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit cuda -y
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install conda-forge::weasyprint -y
conda install -c conda-forge python-kaleido -y

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit cuda -y
pip3 install vllm==0.6.4
pip install flash-attn --no-build-isolation
pip3 install -r pathfinder/requirements.txt
pip3 install -r requirements.txt
