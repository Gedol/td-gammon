# This needs to be put inside a directory has both td-gammon and gym-gammon as 
# subdirectories.


# Use Ubuntu ltr "Jammy Jellyfish" base docker
FROM ubuntu:22.04

# Install basic programs like git and python
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y update \ 
    && echo "------------------------------------------------------ Common" \
    && apt-get install -y sudo jq git \
#    && apt-get install -y curl wget telnet netcat dnsutils \
#    && apt-get install -y software-properties-common \
#    && apt-get install -y zip gzip tar \
#    && apt-get install -y gdebi-core \
#    && apt-get install -y acl \
#    && apt-get install -y psmisc \
 #   && echo "------------------------------------------------------ docker systemctl replacement" \  # as: maybe needed for running services
#    && wget https://raw.githubusercontent.com/gdraheim/docker-systemctl-replacement/master/files/docker/systemctl3.py -O /usr/local/bin/systemctl  \
#    && chown -R abc /usr/local/bin/systemctl \ 
    && echo "------------------------------------------------------ Python" \
    && apt-get install -y python3-distutils \
    && apt-get install -y python3-pip \
    && pip install --upgrade pip \
    && pip install --upgrade setuptools \
    && pip install --upgrade distlib \
    &&  apt-get install -y python-is-python3
    
    
# Add source files, gotten from calling
# git clone https://github.com/Gedol/gym-backgammon
# git clone -b exp https://github.com/Gedol/td-gammon
ADD td-gammon ./td-gammon 
ADD gym-backgammon ./gym-backgammon

# original code requires gym 0.21, but gym 0.21 installation is broken with
# more recent versions. To install gym 0.21 need earlier version of pip, as
# discussed here: https://github.com/openai/gym/issues/3176
# This step is expensive b/c it installs PyTorch, nvidia cuda, etc...
RUN export DEBIAN_FRONTEND=noninteractive \
&& pip install pyglet \
&& pip install setuptools==65.5.0 pip==21 \
&& pip3 install -r ./td-gammon/requirements.txt

# install gym-backgammon
RUN export DEBIAN_FRONTEND=noninteractive \
&& cd gym-backgammon/ \
&& pip3 install -e .

# make saved_models directory inside of td-gammon (maybe change this)
RUN export DEBIAN_FRONTEND=noninteractive \
&& cd td-gammon/td_gammon \
&& mkdir mkdir saved_models \
&& sudo apt-get install -y python3-packaging

# train test model
RUN export DEBIAN_FRONTEND=noninteractive \
&& cd td-gammon/td_gammon \
&& python main.py train --lamda 0 --save_path /td-gammon/td_gammon/saved_models --save_step 50 --episodes 100 --name test2 --type nn --hidden_units 40 > test2_output.txt

##### analyze with gnu


# install gnubg
RUN export DEBIAN_FRONTEND=noninteractive \
&& apt-get -y update \
&& apt-get install -y gnubg

CMD sh -c "ls -l -h -R td-gammon/td_gammon/"


