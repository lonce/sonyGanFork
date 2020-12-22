# docker image build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --file Dockerfile --tag foo:bar ../
FROM nvcr.io/nvidia/pytorch:20.11-py3

#SHELL [ "/bin/bash", "--login", "-c" ]



# create a project directory inside user home
ENV PROJECT_DIR /working
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

RUN pip install --upgrade --ignore-installed llvmlite
RUN pip install 'librosa==0.7.2'
RUN pip install 'numba==0.48.0'


RUN git clone https://github.com/MTG/freesound-python.git 
WORKDIR $PROJECT_DIR/freesound-python
RUN	python setup.py install 
WORKDIR $PROJECT_DIR

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install pyFFTW 
RUN pip install mido

RUN git clone https://github.com/grrrr/nsgt.git 
WORKDIR $PROJECT_DIR/nsgt
RUN python setup.py build
RUN python setup.py install
WORKDIR /sonyGan

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user



# default command will launch JupyterLab server for development
# CMD echo "Hello gan runner"
CMD ["bash"]

# To run - go to your CompareGan directory, run 
# docker run --ipc=host --gpus all -it -v $(pwd):/sonyGan -v /full/path/to/folder/containing/nsynth-train:/mydata foo:bar
