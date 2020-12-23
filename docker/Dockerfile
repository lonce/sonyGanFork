# docker image build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --file Dockerfile --tag foo:bar ../
FROM nvcr.io/nvidia/pytorch:20.12-py3

# create a project directory inside user home
ENV PROJECT_DIR /working
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

RUN pip install --upgrade --ignore-installed "llvmlite>=0.34.0.dev0,<0.35"

# install freesound
RUN git clone https://github.com/MTG/freesound-python.git 
WORKDIR $PROJECT_DIR/freesound-python
RUN	python setup.py install 
WORKDIR $PROJECT_DIR

COPY requirements.txt .
RUN pip install -r requirements.txt

# install nsgt
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
CMD ["bash"]

# To run - go to your CompareGan directory, run 
# docker run --ipc=host --gpus all -it -v $(pwd):/sonyGan -v /full/path/to/folder/containing/nsynth-train:/mydata foo:bar
