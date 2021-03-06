FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
COPY . /pica
WORKDIR /pica
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN ["pip", "install", "-r", "./docs/requirements.txt"]
ENTRYPOINT ["python", "-m", "torch.distributed.launch"]
