FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y \
    sudo \
    bash \
    git \
    curl \
    wget \
    nodejs \
    npm \
    vim \
    nano \
    less \
    unzip \
    zip \
    build-essential \
    cmake \
    gdb \
    gdbserver \
    lldb \
    clang \
    clang-format \
    clangd \
    python3-pip \
    python3-tk \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    iputils-ping \
    net-tools \
    tmux \
    htop \
    tree \
    dos2unix \
    rsync \
    libusb-1.0-0 \
    usbutils \
    udev \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

RUN if getent group "${GID}" >/dev/null; then \
        existing_group=$(getent group "${GID}" | cut -d: -f1); \
        if [ "${existing_group}" != "${USERNAME}" ]; then \
            if getent group "${USERNAME}" >/dev/null; then \
                group_name="${existing_group}"; \
            else \
                groupmod -n "${USERNAME}" "${existing_group}"; \
                group_name="${USERNAME}"; \
            fi; \
        else \
            group_name="${USERNAME}"; \
        fi; \
    else \
        groupadd -g "${GID}" "${USERNAME}"; \
        group_name="${USERNAME}"; \
    fi && \
    if getent passwd "${UID}" >/dev/null; then \
        existing_user=$(getent passwd "${UID}" | cut -d: -f1); \
        if [ "${existing_user}" != "${USERNAME}" ]; then \
            if id -u "${USERNAME}" >/dev/null 2>&1; then \
                user_name="${existing_user}"; \
            else \
                usermod -l "${USERNAME}" -d "/home/${USERNAME}" -m "${existing_user}"; \
                user_name="${USERNAME}"; \
            fi; \
        else \
            user_name="${USERNAME}"; \
        fi; \
        usermod -g "${GID}" "${user_name}"; \
    else \
        useradd -m -s /bin/bash -u "${UID}" -g "${GID}" "${USERNAME}"; \
        user_name="${USERNAME}"; \
    fi && \
    echo "${user_name} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN rosdep init || true

RUN python3 -m pip install --no-cache-dir --upgrade pip "setuptools<81" wheel && \
    python3 -m pip install --no-cache-dir \
    numpy \
    scipy \
    pyzmq \
    matplotlib \
    pyserial \
    vpython \
    dynamixel-sdk \
    taichi==1.7.4 \
    genesis-world \
    imgui \
    glfw \
    torch \
    torchvision \
    opencv-contrib-python \
    pyrealsense2 \
    open3d \
    ultralytics \
    fastsam \
    Pillow \
    PyYAML \
    requests \
    tqdm \
    pandas \
    seaborn && \
    python3 -m pip cache purge

USER ${USERNAME}
WORKDIR /home/${USERNAME}/ws

RUN echo "source /opt/ros/humble/setup.bash" >> "/home/${USERNAME}/.bashrc" && \
    echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> "/home/${USERNAME}/.bashrc" && \
    echo "if [ -f /home/${USERNAME}/ws/install/setup.bash ]; then source /home/${USERNAME}/ws/install/setup.bash; fi" >> "/home/${USERNAME}/.bashrc"

CMD ["sleep", "infinity"]
