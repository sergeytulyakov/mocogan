FROM nvidia/cuda

# Install python
RUN \
  apt-get update && \
  apt-get install -y python python-dev python-pip && \
  apt-get install -y git ffmpeg && \
  apt-get install -y cython python-numpy && \
  rm -rf /var/lib/apt/lists/*

# Install pytorch
RUN \
  pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl && \
  pip install torchvision 

# Install dependencies
RUN \
  pip install -U docopt pyyaml numpy matplotlib tqdm Pillow tensorflow scipy

# Clone the repository
RUN \
  git clone https://github.com/sergeytulyakov/mocogan.git /mocogan

# Define working directory
WORKDIR /mocogan

# Define default command
CMD ["bash"]
