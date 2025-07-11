import argparse
import re

import paramiko


def run_command_by_ssh(
    private_key_file: str,
    hostname: str,
    port: int,
    username: str,
    command: str,
):
    private_key = paramiko.RSAKey.from_private_key_file(private_key_file)
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(
        hostname=hostname,
        port=port,
        username=username,
        pkey=private_key,
    )
    stdin, stdout, stderr = ssh_client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    if exit_status == 0:
        res = stdout.read().decode()
        err = ""
    else:
        res = ""
        err = stderr.read().decode()
    ssh_client.close()
    return res,err,exit_status

def setup_mount_nas(
    private_key_file: str,
    hostname: str,
    port: int,
    username: str,
    nas_url: str,
):
    cmd = 'mkdir -p /storage;mount -t nfs -o vers=4,minorversion=0,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport {}:/ /storage'.format(nas_url)
    res, err, exit_status = run_command_by_ssh(private_key_file,hostname,port,username,cmd)
    if exit_status != 0:
        print(err)
        raise Exception('fail to mount_nas')

def setup_download_codes(
    private_key_file: str,
    hostname: str,
    port: int,
    username: str,
):
    cmd = 'apt install -y git || yum install -y git || dnf install -y git;mkdir -p /storage/codes;cd /storage/codes/;test -d AReaL || git clone https://github.com/inclusionAI/AReaL'
    res, err, exit_status = run_command_by_ssh(private_key_file,hostname,port,username,cmd)
    if exit_status != 0:
        print(err)
        raise Exception('fail to download codes')

def setup_install_dependency(
    private_key_file: str,
    hostname: str,
    port: int,
    username: str,
):
    cmd = 'bash /storage/codes/AReaL/examples/env/scripts/install-dependency.sh'
    res, err, exit_status = run_command_by_ssh(private_key_file,hostname,port,username,cmd)
    if exit_status != 0:
        print(err)
        raise Exception('fail to install dependency')


def setup_download_dataset(
    private_key_file: str,
    hostname: str,
    port: int,
    username: str,
):
    cmd = 'bash /storage/codes/AReaL/examples/env/scripts/download-dataset.sh'
    res, err, exit_status = run_command_by_ssh(private_key_file,hostname,port,username,cmd)
    if exit_status != 0:
        print(err)
        raise Exception('fail to download dataset')

def setup_download_model(
    private_key_file: str,
    hostname: str,
    port: int,
    username: str,
):
    cmd = 'bash /storage/codes/AReaL/examples/env/scripts/download-model.sh'
    res, err, exit_status = run_command_by_ssh(private_key_file,hostname,port,username,cmd)
    if exit_status != 0:
        print(err)
        raise Exception('fail to download model')

def setup_start_ray_header(
    private_key_file: str,
    hostname: str,
    port: int,
    username: str,
):
    cmd = 'mkdir -p /storage/ray;cp /storage/codes/AReaL/examples/cluster_config_on_ray.json /storage/ray/cluster_config_on_ray.json; docker run -d --name r1-ray-head --privileged --gpus all --network host --shm-size 700g -v /storage:/storage ghcr.io/inclusionai/areal-runtime:v0.1.0 /bin/bash -c "ray start --head --port=6379 && tail -f /dev/null"'
    res, err, exit_status = run_command_by_ssh(private_key_file,hostname,port,username,cmd)
    if exit_status != 0:
        print(err)
        raise Exception('fail to start ray header')

def setup_start_ray_worker(
    private_key_file: str,
    hostname: str,
    port: int,
    username: str,
    header_hostname: str,
):
    cmd = 'RAY_HEAD_IP={};docker run -d --name r1-ray-worker --privileged --gpus all --network host --shm-size 700g -v /storage:/storage ghcr.io/inclusionai/areal-runtime:v0.1.0 /bin/bash -c "ray start --address=$RAY_HEAD_IP:6379 && tail -f /dev/null"'.format(header_hostname)
    res, err, exit_status = run_command_by_ssh(private_key_file,hostname,port,username,cmd)
    if exit_status != 0:
        print(err)
        raise Exception('fail to start ray worker')

def setup_start_traning(
    train_param: str,
    private_key_file: str,
    hostname: str,
    port: int,
    username: str,
):
    cmd = 'docker exec r1-ray-head /bin/bash -c "cd /storage/codes/AReaL;mkdir -p /storage/ray/train_batch_logs/;nohup bash ./examples/train_batch_{}.sh &> /storage/ray/train_batch_logs/{}.log &"'.format(train_param,train_param)
    res, err, exit_status = run_command_by_ssh(private_key_file,hostname,port,username,cmd)
    if exit_status != 0:
        print(err)
        raise Exception('fail to start train')

# setup running environment
def setup(args):
    total_nodes = len(args.hostnames)
    # param check
    if args.train_param != "":
        node_required = int(re.findall(r"\d+", args.train_param)[-1])
        if node_required > total_nodes:
            print("training needs {} node but {} found".format(node_required,total_nodes))
            return
    # mount nas if needed
    if args.nas_url != "":
        print("starting mount nas")
        for index in range(0,total_nodes):
            private_key_file = args.private_key_file
            hostname = args.hostnames[index]
            port = args.ssh_port
            username = args.username
            setup_mount_nas(
                private_key_file,
                hostname,
                port,
                username,
                args.nas_url,
            )
    # download codes
    print("starting download codes")
    setup_download_codes(
        args.private_key_file,
        args.hostnames[0],
        args.ssh_port,
        args.username,
    )
    # install dependency
    print("starting install dependency")
    for index in range(0,total_nodes):
        private_key_file = args.private_key_file
        hostname = args.hostnames[index]
        port = args.ssh_port
        username = args.username
        setup_install_dependency(
            private_key_file,
            hostname,
            port,
            username,
        )
    # download datasets
    print("starting download datasets")
    setup_download_dataset(
        args.private_key_file,
        args.hostnames[0],
        args.ssh_port,
        args.username,
    )
    # download model
    print("starting download model")
    setup_download_model(
        args.private_key_file,
        args.hostnames[0],
        args.ssh_port,
        args.username,
    )
    # start ray header
    print("starting ray header")
    setup_start_ray_header(
        args.private_key_file,
        args.hostnames[0],
        args.ssh_port,
        args.username,
    )
    # start ray worker if needed
    if total_nodes > 1:
        print("starting ray worker")
        for index in range(1,total_nodes):
            private_key_file = args.private_key_file
            hostname = args.hostnames[index]
            port = args.ssh_port
            username = args.username
            header_hostname = args.hostnames[0]
            setup_start_ray_worker(
                private_key_file,
                hostname,
                port,
                username,
                header_hostname,
            )
    print("setup success")
    # start training if needed
    if args.train_param != "":
        print("starting training process")
        setup_start_traning(
            args.train_param,
            args.private_key_file,
            args.hostnames[0],
            args.ssh_port,
            args.username
        )
        print("training process started")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_setup = subparsers.add_parser(
        name="setup",
        help="setup training environment",
    )
    parser_setup.add_argument(
        "--private_key_file",
        required=True,
        help="",
        type=str,
    )
    parser_setup.add_argument(
        "--hostnames",
        required=True,
        help="",
        nargs="+",
        type=str,
    )
    parser_setup.add_argument(
        "--ssh_port",
        help="",
        default=22,
        type=int,
    )
    parser_setup.add_argument(
        "--username",
        help="",
        default="root",
        type=str,
    )
    parser_setup.add_argument(
        "--nas_url",
        help="",
        default="",
        type=str,
    )
    parser_setup.add_argument(
        "--train_param",
        help="",
        choices=["","1.5B_n1","1.5B_n4","1.5B_n16","7B_n4","7B_n16"],
        default="",
        type=str,
    )
    parser_setup.set_defaults(func=setup)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
