FROM rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.1.2

RUN apt update && \
    apt install -y wget \
    git \
    rocsparse-dev \
    hipsparse-dev \
    rocthrust-dev \
    rocblas-dev \
    hipblas-dev && \
    rocm-smi && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --upgrade pip
RUN pip install --upgrade numpy setuptools wheel ninja packaging
RUN ROCM_VERSION=6.1 pip install -vvv --no-build-isolation git+https://github.com/AutoGPTQ/AutoGPTQ.git@v0.7.1

RUN pip install kaleido
RUN pip3 install accelerate transformers optimum einops tiktoken  hf_transfer backoff openai mistralai anthropic pygtrie numpy
RUN pip3 install wandb==0.16.1 pettingzoo==1.24.2 pytest==7.4.3 pydantic==1.10.11 typing_extensions==4.7.0 hydra_core==1.3.2 nbformat>=4.2.0 dash dash-bootstrap-components lifelines statsmodels dash-mantine-components python-dotenv flask_caching sentence_transformers randomname black