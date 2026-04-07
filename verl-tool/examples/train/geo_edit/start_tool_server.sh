#!/usr/bin/env bash
set -e

VERL_ROOT="/storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL/verl-tool"
AREAL_ROOT="/storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL"
LOG_DIR="/storage/openpsi/models/lcy_image_edit/rl_workspace/logs/tool-server"
PORT=30888

echo "=== Step 1: Cleanup ==="
python3 << 'STEP1'
import ray, time
ray.init(address='auto', ignore_reinit_error=True)

for name in ['tool_srv']:
    for ns in ['tool', 'tool_server']:
        try: ray.kill(ray.get_actor(name, namespace=ns))
        except: pass

@ray.remote(resources={'tool_agent': 0.001})
def cleanup():
    import subprocess, time, os, signal
    subprocess.run('pkill -9 -f verl_tool.servers', shell=True)
    time.sleep(1)
    subprocess.run('fuser -k 30888/tcp', shell=True)
    time.sleep(2)
    r = subprocess.run('ss -tlnp | grep 30888 || echo PORT_FREE', shell=True, capture_output=True, text=True)
    return r.stdout.strip()

print('Port:', ray.get(cleanup.remote()))

@ray.remote(resources={'tool_agent': 0.001})
def kill_idle():
    import subprocess, os, signal
    r = subprocess.run("ps aux | grep 'ray::IDLE' | grep -v grep | awk '{print $2}'", shell=True, capture_output=True, text=True)
    pids = [p for p in r.stdout.strip().split('\n') if p]
    for pid in pids:
        try: os.kill(int(pid), signal.SIGKILL)
        except: pass
    return f'Killed {len(pids)} idle workers'

print(ray.get(kill_idle.remote()))
STEP1

sleep 3

echo ""
echo "=== Step 2: Launch tool server on worker ==="
python3 << STEP2
import ray, time

ray.init(address='auto', ignore_reinit_error=True)

head_ip = ray.util.get_node_ip_address()
runtime_env = {
    'env_vars': {
        'PYTHONPATH': '${VERL_ROOT}/verl:${VERL_ROOT}:${AREAL_ROOT}',
        'GEOEDIT_ENABLE_TOOLS': 'general,chart',
        'RAY_ADDRESS': f'{head_ip}:6379',
    }
}

@ray.remote(resources={'tool_agent': 0.001}, num_cpus=1, runtime_env=runtime_env)
class ToolServer:
    def __init__(self):
        import os, subprocess, socket
        self.ip = socket.gethostbyname(socket.gethostname())
        env = os.environ.copy()
        log_dir = '${LOG_DIR}'
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = f'{log_dir}/serve.log'
        log_file = open(self.log_path, 'w')
        cmd = ['python3', '-m', 'verl_tool.servers.serve',
               '--host', self.ip, '--port', '${PORT}',
               '--tool_type', 'geo_edit_tool',
               '--workers_per_tool', '1', '--uvi_workers', '1', '--router_workers', '1',
               '--max_concurrent_requests', '128', '--use_ray', 'True']
        self.proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        print(f'Tool server on {self.ip}:${PORT} PID={self.proc.pid}')

    def status(self):
        return {'ip': self.ip, 'pid': self.proc.pid, 'running': self.proc.poll() is None}

    def tail_log(self, n=20):
        import subprocess as sp
        return sp.run(['tail', '-n', str(n), self.log_path], capture_output=True, text=True).stdout

srv = ToolServer.options(name='tool_srv', lifetime='detached', namespace='tool').remote()
time.sleep(10)
status = ray.get(srv.status.remote())
print(f'Status: {status}')
if not status['running']:
    print('DIED! Logs:')
    print(ray.get(srv.tail_log.remote(20)))
    exit(1)
STEP2

echo ""
echo "=== Step 3: Health check ==="
WORKER_IP=$(python3 -c "
import ray; ray.init(address='auto',ignore_reinit_error=True)
for n in ray.nodes():
    if n['Resources'].get('tool_agent',0)>0 and n['Alive']:
        print(n['NodeManagerAddress']); break
")

for i in $(seq 1 90); do
    if curl -s "http://$WORKER_IP:$PORT/health" 2>/dev/null | grep -q "healthy"; then
        echo "HEALTHY on $WORKER_IP:$PORT (${i}x2s)"
        break
    fi
    [ $((i % 10)) -eq 0 ] && echo "  Waiting... ($i/90)"
    sleep 2
done

echo ""
echo "=== Step 4: Verify GPU usage ==="
sleep 15
python3 << 'STEP4'
import ray
ray.init(address='auto', ignore_reinit_error=True)
a = ray.available_resources()
t = ray.cluster_resources()
gpu_used = t['GPU'] - a.get('GPU', 0)
tool_used = t['tool_agent'] - a.get('tool_agent', 0)
print(f'GPU used: {gpu_used:.0f}/{t["GPU"]:.0f}')
print(f'tool_agent used: {tool_used:.0f}/{t["tool_agent"]:.0f}')
if gpu_used >= 6:
    print('SUCCESS: All 6 tool agents loaded!')
elif gpu_used > 0:
    print(f'PARTIAL: {gpu_used:.0f} agents loaded. Check backend log:')
    print('  tail -100 /storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL/tool-server-logs/tool_server_backend_0.log')
else:
    print('FAIL: No agents loaded.')
STEP4
