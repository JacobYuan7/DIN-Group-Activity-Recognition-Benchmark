FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
#FROM nvcr.io/nvidia/pytorch:19.09-py3

# install RoIAlign for Pytorch
WORKDIR /opt/roialign
RUN git clone https://github.com/longcw/RoIAlign.pytorch.git . && \
    sed -ie 's/torch.cuda.is_available()/True/g' setup.py && \
    python setup.py install

# clone DIN_GAR repo
WORKDIR /opt/DIN_GAR
RUN git clone https://github.com/JacobYuan7/DIN_GAR.git .
RUN pip install \
      scikit-image \
      pillow==6.2.0 \
      thop \
      fvcore=='0.1.2.*' \
      opencv-python==4.5.3.*
