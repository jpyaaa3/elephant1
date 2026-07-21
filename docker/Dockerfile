FROM ros:humble-ros-base-jammy

ENV PYTHONNOUSERSITE=1
ENV PATH=/opt/openrobots/bin:$PATH
ENV PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH
ENV CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH

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
    lsb-release \
    vim \
    build-essential \
    cmake \
    gdb \
    python3-pip \
    python3-tk \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    iputils-ping \
    libusb-1.0-0 \
    usbutils \
    udev \
    libgl1 \
    libegl1 \
    libglvnd0 \
    libglx-mesa0 \
    libglu1-mesa \
    libosmesa6 \
    mesa-utils \
    libglfw3 \
    libglib2.0-0 \
    libdbus-1-3 \
    libfontconfig1 \
    libfreetype6 \
    libsm6 \
    libice6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libxcb-xkb1 \
    libxext6 \
    libxft2 \
    libxrender1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings && \
    curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
        | tee /etc/apt/keyrings/robotpkg.asc >/dev/null && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
        > /etc/apt/sources.list.d/robotpkg.list && \
    apt-get update && \
    apt-get install -y robotpkg-py310-pinocchio && \
    rm -rf /var/lib/apt/lists/*

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

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip "setuptools>=68,<80" wheel && \
    python3 -m pip install --no-cache-dir "numpy==1.26.4" && \
    python3 -m pip install --no-cache-dir --upgrade-strategy only-if-needed -r /tmp/requirements.txt && \
    python3 -c "import aiortc, av, casadi, pinocchio, convex_mpc; from convex_mpc.go2_robot_data import PinGo2Model; from convex_mpc.centroidal_mpc import CentroidalMPC; print('runtime imports ok')"
RUN python3 -m pip cache purge

USER ${USERNAME}
WORKDIR /home/${USERNAME}/ws

RUN echo "source /opt/ros/humble/setup.bash" >> "/home/${USERNAME}/.bashrc" && \
    echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> "/home/${USERNAME}/.bashrc" && \
    echo "if [ -f /home/${USERNAME}/ws/install/setup.bash ]; then source /home/${USERNAME}/ws/install/setup.bash; fi" >> "/home/${USERNAME}/.bashrc"

CMD ["sleep", "infinity"]
