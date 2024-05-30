conda create -n GovComGPTQ  python=3.11.5 -y
conda activate GovComGPTQ
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit cuda -y
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

conda install conda-forge::weasyprint -y
conda install -c conda-forge python-kaleido -y

pip3 install -r pathfinder/requirements.txt
pip3 install autogptq
pip3 install bitsandbytes

pip3 install -r requirements.txt


pip3 transformers==4.39.3
