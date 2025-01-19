# conda create -n GovSim  python=3.11.5 -y
conda activate GovSim

conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit cuda -y

conda install conda-forge::weasyprint -y
conda install -c conda-forge python-kaleido -y

pip3 install -r pathfinder/requirements.txt
pip3 install auto-gptq
pip3 install bitsandbytes

pip3 install -r requirements.txt
pip3 install numpy=1.26.4

pip3 install transformers