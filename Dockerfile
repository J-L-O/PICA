FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
COPY . /pica
WORKDIR /pica
RUN ["pip", "install", "-r", "./docs/requirements.txt"]
ENTRYPOINT ["python", "main.py", "--cfgs", "./configs/base.yaml"]
