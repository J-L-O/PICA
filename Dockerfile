FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
COPY . /pica
RUN pip install -r /pica/requirements.txt
CMD python /pica/main.py --cfgs /pica/configs/base.yaml /pica/configs/stl10.yaml
