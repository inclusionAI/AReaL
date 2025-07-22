#!/bin/sh
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m | tr '[:upper:]' '[:lower:]')
DISTRO=""
GIT_LFS_VERSION="3.4.0"

# get distro
if [ "$OS" = "linux" ]; then
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
    else
        echo "unknown linux distribution"
        exit 1
    fi
else
    echo "Unsupport os: $OS"
    exit 1
fi

command_exists() {
    command -v "$@" > /dev/null 2>&1
}

install_git() {
    # check git exists
    if command_exists git; then
        return 0
    fi

    case "$DISTRO" in 
        ubuntu)
            sudo apt install -y git
        ;;
        centos|alinux|amzn)
            sudo yum install -y git || sudo dnf install -y git
        ;;
        *)
            echo "Unsupported operating system $DISTRO"
            exit 1
        ;;
    esac
}

install_git_lfs() {
    # check git exists
    if command_exists "git lfs"; then
        return 0
    fi

    case "$ARCH" in
        x86_64)
            GIT_LFS_URL="https://github.com/git-lfs/git-lfs/releases/download/v3.4.0/git-lfs-linux-amd64-v${GIT_LFS_VERSION}.tar.gz"
        ;;
        aarch64)
            GIT_LFS_URL="https://github.com/git-lfs/git-lfs/releases/download/v3.4.0/git-lfs-linux-arm64-v${GIT_LFS_VERSION}.tar.gz"
        ;;
        *)
            echo "Unsupported arch $ARCH"
            exit 1
        ;;
    esac

    TEMP_DIR=$(mktemp -d)
    echo "Start download to ${TEMP_DIR}"
    curl -sL "$GIT_LFS_URL" -o "$TEMP_DIR/git-lfs.tar.gz"
    if [ $? -ne 0 ]; then
        echo "fail to download"
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    echo "Start unzip"
    tar -xzf "$TEMP_DIR/git-lfs.tar.gz" -C "$TEMP_DIR"
    if [ $? -ne 0 ]; then
        echo "fail to unzip"
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    echo "Start install"
    cd "$TEMP_DIR/git-lfs-${GIT_LFS_VERSION}" || exit 1
    sudo ./install.sh
    if [ $? -ne 0 ]; then
        echo "install fail"
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    echo "Start config"
    git lfs install
    if [ $? -ne 0 ]; then
        echo "git-lfs config error"
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    rm -rf "$TEMP_DIR"
}

install_docker() {
    # check docker exists
    if command_exists docker; then
        return 0
    fi

    case "$DISTRO" in
        ubuntu)
            sudo apt-get update
            sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common
            sudo curl -fsSL http://mirrors.cloud.aliyuncs.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
            sudo add-apt-repository -y "deb [arch=$(dpkg --print-architecture)] http://mirrors.cloud.aliyuncs.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
            sudo apt-get -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        ;;
        centos)
            sudo wget -O /etc/yum.repos.d/docker-ce.repo http://mirrors.cloud.aliyuncs.com/docker-ce/linux/centos/docker-ce.repo
            sudo sed -i 's|https://mirrors.aliyun.com|http://mirrors.cloud.aliyuncs.com|g' /etc/yum.repos.d/docker-ce.repo
            sudo dnf -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        ;;
        alinux)
            sudo rm -f /etc/yum.repos.d/docker*.repo
            sudo dnf -y remove docker-ce containerd.io docker-ce-rootless-extras docker-buildx-plugin docker-ce-cli docker-compose-plugin
            sudo wget -O /etc/yum.repos.d/docker-ce.repo http://mirrors.cloud.aliyuncs.com/docker-ce/linux/centos/docker-ce.repo
            sudo sed -i 's|https://mirrors.aliyun.com|http://mirrors.cloud.aliyuncs.com|g' /etc/yum.repos.d/docker-ce.repo
            sudo dnf -y install dnf-plugin-releasever-adapter --repo alinux3-plus
            sudo dnf -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        ;;
        amzn)
            sudo yum update -y
            sudo dnf -y install docker
        ;;
        *)
            echo "Unsupported operating system $DISTRO"
            exit 1
        ;;
    esac

    sudo systemctl enable docker
    sudo systemctl start docker
}

install_nvidia_container_toolkit() {
    # check nvidia-container-toolkit exits
    if command_exists nvidia-container-toolkit; then
        return 0
    fi

    case "$DISTRO" in
        ubuntu)
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
            && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            sudo apt-get update
            sudo apt-get install -y nvidia-container-toolkit
        ;;
        centos|alinux|amzn)
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
            sudo dnf install -y nvidia-container-toolkit
        ;;
        *)
            echo "Unsupported operating system $DISTRO"
            exit 1
        ;;
    esac
}

restart_docker() {
    sudo systemctl restart docker
}

install_git
install_git_lfs
install_docker
install_nvidia_container_toolkit
restart_docker