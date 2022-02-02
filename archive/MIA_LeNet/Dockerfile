FROM nvidia/cuda:11.3.0-base-ubuntu20.04
WORKDIR /lab
COPY . .
RUN apt-get update \
&& apt-get -y install python3 \
&& apt-get -y install python3-pip \
&& pip3 install -r requirements.txt \
&& pip3 install --upgrade https://github.com/Theano/Theano/archive/master.zip \
&& pip3 install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip \
&& THEANO_FLAGS='device=cuda,floatX=float32'
CMD /bin/bash