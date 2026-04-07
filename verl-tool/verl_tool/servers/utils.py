import os
import signal
import subprocess
import sys
import hashlib

def kill_python_subprocess_processes():
    """
    Kill any lingering Python processes that were spawned with the -c flag.
    This is useful for cleaning up processes that might have escaped the timeout mechanism.
    Only kills individual processes, not process groups, to avoid affecting unrelated processes.
    
    Returns:
        int: Number of processes killed
    """
    try:
        # Find Python processes
        ps_process = subprocess.Popen(
            ["ps", "-ef"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, _ = ps_process.communicate()
        
        # Track our PID to avoid killing ourselves
        own_pid = os.getpid()
        ps_pid = ps_process.pid
        
        killed_count = 0
        
        for line in stdout.splitlines():
            parts = line.split()
            if len(parts) < 8:  # Ensure there are enough parts in the line
                continue
                
            pid_str = parts[1]
            # The command starts at index 7 in ps -ef output
            cmd = " ".join(parts[7:])
            
            # Check for python/python3 -c pattern which indicates code execution
            if (("python -c" in cmd or "python3 -c" in cmd) and 
                pid_str.isdigit()):
                pid = int(pid_str)
                
                # Don't kill our own process or the ps process
                if pid != own_pid and pid != ps_pid:
                    try:
                        # Kill only this specific process, not its process group
                        os.kill(pid, signal.SIGKILL)
                        killed_count += 1
                    except (ProcessLookupError, PermissionError) as e:
                        # Process may have already terminated or we don't have permission
                        print(f"Error killing process {pid}: {e}")
        
        return killed_count
            
    except Exception as e:
        print(f"Error during process cleanup: {e}")
        return 0
    
    
def hash_requests(data):
    """
    Hash the input data to create a unique identifier.
    
    Args:
        data: Input data to hash
    
    Returns:
        str: Hexadecimal hash string
    """
    # Convert the data to a string and encode it
    data_str = str(data).encode('utf-8')
    hash_object = hashlib.sha256()
    hash_object.update(data_str)
    return hash_object.hexdigest()