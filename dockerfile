FROM ubuntu:latest
SHELL ["/bin/bash", "-c"]

USER root
RUN apt-get update
RUN apt-get install -y build-essential libsqlite3-dev zlib1g-dev gdal-bin python3 libpq-dev python3-dev python3-pip git tmux 

RUN git clone https://github.com/felt/tippecanoe.git
WORKDIR tippecanoe
RUN make -j
RUN make install

WORKDIR /home/root
RUN git clone https://github.com/insarlab/MintPy.git
WORKDIR MintPy
RUN if [ -z ${PYTHONPATH+x} ]; then export PYTHONPATH=""; fi
RUN export MINTPY_HOME=~/tools/MintPy
RUN export PATH=${PATH}:${MINTPY_HOME}/src/mintpy/cli
RUN export PYTHONPATH=${PYTHONPATH}:${MINTPY_HOME}/src

WORKDIR /home/root
RUN git clone https://github.com/geodesymiami/insarmaps_scripts.git
WORKDIR insarmaps_scripts
RUN pip3 install --break-system-packages -r requirements.txt
RUN pip3 install --break-system-packages h5py scipy pyresample

WORKDIR /home/root
RUN git clone https://github.com/stackTom/config_files.git
WORKDIR /home/root/config_files
RUN ./install.sh
RUN git config --global user.name "INSERT"
RUN git config --global user.email "INSERT@INSERT.com"

WORKDIR /home/root/insarmaps_scripts

CMD /home/root/insarmaps_scripts/docker/setup_env.sh

