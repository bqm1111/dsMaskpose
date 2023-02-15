FROM nvidia/deepstream:6.0.1-devel as build_env

# Use cached repos
ENV TZ Asia/Ho_Chi_Minh
RUN curl "http://172.21.100.253:3000/qdt/Network/raw/branch/master/apt-interal.public.gpg.key" | apt-key add - && \
    echo "deb [arch=amd64] http://172.21.100.15:8081/repository/apt-internal bionic main # also provide hosted packages" > /etc/apt/sources.list && \
    echo "deb [arch=amd64] http://172.21.100.15:8081/repository/apt-proxy bionic main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb [arch=amd64] http://172.21.100.15:8081/repository/apt-proxy bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    wget http://172.21.100.253:3000/qdt/Network/raw/branch/master/nvidia/cuda-ubuntu1804.pin && \
    mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys http://172.21.100.253:3000/qdt/Network/raw/branch/master/nvidia/3bf863cc.pub && \
    echo "deb http://172.21.100.15:8081/repository/apt-proxy-nvidia/ /" > /etc/apt/sources.list.d/cuda.list

# update nvidia
RUN apt-get update && \
    apt-get install -y  libnvinfer-dev=8.4.1-1+cuda11.6 \
    libnvinfer-plugin8=8.4.1-1+cuda11.6 \
    libnvinfer-plugin-dev=8.4.1-1+cuda11.6 \
    libnvinfer-samples=8.4.1-1+cuda11.6 \
    libnvinfer8=8.4.1-1+cuda11.6 \
    libnvonnxparsers-dev=8.4.1-1+cuda11.6 \
    libnvonnxparsers8=8.4.1-1+cuda11.6 \
    libnvparsers-dev=8.4.1-1+cuda11.6 \
    libnvparsers8=8.4.1-1+cuda11.6 \
    python3-libnvinfer=8.4.1-1+cuda11.6 && \
    apt-get install -y libcudnn8=8.4.1.50-1+cuda11.6 libcudnn8-dev=8.4.1.50-1+cuda11.6 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# remove old cmake
RUN apt-get update && \
    apt-get -y install build-essential && \
    apt-get -y remove --purge cmake cmake-data && \
    rm -rf /root/.local/share/bash-completion/completions/cmake && \
    rm -rf /usr/local/bin/cmake && \
    rm -rf /usr/local/share/cmake-3.15/completions/cmake && \
    apt-get -y install -o Dpkg::Options::="--force-overwrite" cmake && \
    rm -rf /var/lib/apt/lists/*

# Install basic packages
RUN apt-get update && \
    apt-get -y install x11-xserver-utils && \
    apt-get -y install wget unzip build-essential cmake pkg-config && \
    apt-get -y install libboost-program-options-dev && \
    apt-get -y install libjansson4 libjansson-dev && \
    apt-get -y install libglib2.0-dev libjson-glib-dev uuid-dev && \
    apt-get -y install libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install OpenCV
ARG CV_VERSION=4.3.0
RUN apt-get purge -y libopencv* && \
    apt-get update && \
    apt-get autoremove -y && \
    apt-get install -y build-essential cmake \
    libjpeg-dev libtiff5-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev \
    libv4l-dev v4l-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgtk2.0-dev mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev \
    libatlas-base-dev gfortran libeigen3-dev \
    python2.7-dev python3-dev python-numpy python3-numpy && \
    mkdir -p /root/opencv && cd /root/opencv && \
    wget -O opencv.zip http://172.21.100.15:8081/repository/raw-cached/opencv/opencv/archive/$CV_VERSION.zip && \
    unzip opencv.zip && mv opencv-$CV_VERSION opencv && rm -f opencv.zip && \
    wget -O opencv_contrib.zip http://172.21.100.15:8081/repository/raw-cached/opencv/opencv_contrib/archive/$CV_VERSION.zip && \
    unzip opencv_contrib.zip && mv opencv_contrib-$CV_VERSION opencv_contrib && rm -f opencv_contrib.zip && \
    cd opencv/ && mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_TBB=OFF -D WITH_IPP=OFF -D WITH_1394=OFF \
    -D BUILD_WITH_DEBUG_INFO=OFF -D BUILD_DOCS=OFF -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D WITH_QT=OFF \
    -D WITH_GTK=ON -D WITH_OPENGL=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D WITH_V4L=ON -D WITH_FFMPEG=ON -D WITH_XINE=ON \
    -D BUILD_NEW_PYTHON_SUPPORT=ON -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D WITH_CUDA=OFF -DBUILD_opencv_xfeatures2d=OFF ../ && \
    make -j8 && make install && \
    sh -c 'echo '/usr/local/lib' > /etc/ld.so.conf.d/opencv.conf' && \
    ldconfig && cd /root/ && rm -rf opencv && \
    rm -rf /var/lib/apt/lists/*

# remove old librdkafka and install new libs
RUN rm /usr/local/lib/pkgconfig/rdkafka++.pc && \
    rm /usr/local/lib/pkgconfig/rdkafka.pc && \
    rm -rf /usr/local/include/librdkafka/ && \
    rm -rf /usr/local/lib/librdkafka* && \
    apt-get update && \
    apt-get install -y rapidjson-dev libjson-glib-dev libeigen3-dev libspdlog-dev spdlog librdkafka++1 librdkafka-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

FROM build_env as convert_stage
COPY data data
COPY tools tools
# RUN pip install -r requirements.txt
RUN cd tools/ && \
    python3 test.py 

FROM build_env as build_stage
COPY . .
RUN rm -rf build/ && mkdir build && cd build/ && cmake .. && make -j

FROM build_env as run_stage 
COPY --from=convert_stage /workspace/data/ /workspace/data/
COPY --from=build_stage /workspace /workspace

# Create user
ARG USERNAME=vscode
ARG USER_UID=1003
ARG USER_GID=999
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    chown -R ${USER_UID}:${USER_GID} .

USER $USERNAME

ENV TZ Asia/Ho_Chi_Minh

WORKDIR /workspace/build

# CMD ["./FaceDeepStream", "../configs/source_list.json" ]
