FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
RUN pip install -r requirements.txt
WORKDIR /workspace/cgmc