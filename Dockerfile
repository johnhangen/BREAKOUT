FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /.

COPY . .

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip 
RUN pip install gym==0.26.2 
RUN pip install gym[classic_control]
RUN pip install "gym[atari, accept-rom-license]" 
RUN pip install numpy scipy pandas scikit-learn matplotlib
RUN pip install scikit-image
RUN pip install wandb
RUN pip install --force-reinstall -v numpy==1.23.5 

RUN python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

CMD ["/bin/bash"]