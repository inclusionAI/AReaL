"""
Enhanced Bash Terminal Tool with proper persistent state management
"""
import regex as re
import subprocess
import os
import uuid
import shutil
import resource
import threading
import time
import signal
from typing import Tuple, Dict, Any, Optional, Union, List
import pty
import select
import json

# Timeout for command execution in seconds
TIMEOUT = 30

def check_forbidden_commands(command: str) -> bool:
    """
    Checks if the command contains potentially dangerous operations.
    """
    forbidden_commands = [
        'rm -rf /', 'dd if=', 'mkfs', 'fdisk', 'mount', 'umount',
        'passwd', 'su ', 'sudo ', 'chroot', 'systemctl', 'service',
        'iptables', 'ufw', 'firewall-cmd',
        'nc ', 'ncat ', 'telnet ', 'ssh ', 'scp ', 'rsync ',
        'curl http', 'wget http', 'lynx', 'w3m',
        'crontab', 'batch',
        'kill -9', 'killall', 'pkill ',
        '> /dev/', '< /dev/', 'mknod', 'losetup'
    ]
    
    dangerous_patterns = [
        r'rm\s+.*-rf\s+/',
        r'>\s*/etc/',
        r'>\s*/bin/',
        r'>\s*/usr/',
        r'>\s*/var/',
        r'chmod\s+777',
        r'find\s+/.*-exec',
        r'eval\s+.*[;&|]',
        r'source\s+/',
        r'\.\s+/',
    ]
    
    command_lower = command.lower()
    
    for forbidden in forbidden_commands:
        if forbidden in command_lower:
            return [forbidden]
    
    for pattern in dangerous_patterns:
        detected_forbidden = re.findall(pattern, command_lower)
        if detected_forbidden:
            return detected_forbidden
    
    return False

def simulate_terminal_output(command: str, stdout: str, stderr: str, exit_code: int, prompt: str) -> str:
    """Simulate realistic terminal output"""
    output_lines = []
    
    # Show the command being executed
    output_lines.append(f"{prompt}{command}")
    
    # Add stdout if present
    if stdout:
        output_lines.append(stdout)
    
    # Add stderr if present
    if stderr:
        output_lines.append(stderr)
    
    return "\\n".join(output_lines)

def format_output(stdout: str, stderr: str, exit_code: int) -> str:
    """Format command output to look like a real terminal"""
    output_parts = []
    
    if stdout:
        output_parts.append(stdout)
    
    if stderr:
        output_parts.append(stderr)
    
    if exit_code != 0 and not stderr:
        output_parts.append(f"Command exited with code {exit_code}")
    
    return "\n".join(output_parts)


class BashSession:
    """Manages a persistent bash shell session with proper state persistence"""
    
    def __init__(self, temp_dir: str, use_firejail: bool = False):
        self.temp_dir = temp_dir
        self.use_firejail = use_firejail
        self.session_id = str(uuid.uuid4().hex)
        self.home_dir = None
        self.current_dir = temp_dir
        self.bashrc_file = os.path.join(self.temp_dir, ".bashrc")
        self.history_file = os.path.join(self.temp_dir, ".bash_history")
        self.command_counter = 0
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize the session with a proper bashrc"""
        bashrc_content = f"""#!/bin/bash
# Enhanced bash session initialization

# Basic settings
export HISTFILE=$HOME/.bash_history
export HISTSIZE=1000
export HISTFILESIZE=2000
export PS1="user@bash-session:\\w\\$ "

# Enable history
set -o history
shopt -s histappend


# Source any existing state
if [ -f "$HOME/.bash_env" ]; then
    source "$HOME/.bash_env" 2>/dev/null || true
fi

if [ -f "$HOME/.bash_aliases" ]; then
    source "$HOME/.bash_aliases" 2>/dev/null || true
fi

if [ -f "$HOME/.bash_functions" ]; then
    source "$HOME/.bash_functions" 2>/dev/null || true
fi
# cd "$HOME" 2>/dev/null || true
"""
        
        with open(self.bashrc_file, 'w') as f:
            f.write(bashrc_content)
        os.chmod(self.bashrc_file, 0o644)
        
        # Initialize state files
        for state_file in [".bash_env", ".bash_aliases", ".bash_functions"]:
            file_path = os.path.join(self.temp_dir, state_file)
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write("")
        
        # Initialize history file
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w') as f:
                f.write("")
    
    def _prepare_environment(self):
        """Prepare safe environment variables"""
        env = os.environ.copy()
        
        # Keep essential variables
        essential_vars = [
            "PATH", "USER", "SHELL", "LANG", "LC_ALL", 
            "LC_CTYPE", "TERM", "TMPDIR", "TEMP", "TMP"
        ]
        
        safe_env = {}
        for var in essential_vars:
            if var in env:
                safe_env[var] = env[var]
        
        # Set safe defaults
        safe_env["PATH"] = "/usr/bin:/bin:/usr/local/bin:/usr/sbin:/sbin"
        safe_env["TERM"] = "xterm-256color"
        safe_env["BASH_SILENCE_DEPRECATION_WARNING"] = "1"
        
        return safe_env
    
    def _set_limits(self):
        """Set resource limits for the bash process"""
        try:
            # Memory limit: 1GB virtual memory
            resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024, 1024*1024*1024))
            # Process limit: Allow enough processes for bash operations
            resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
            # File size limit: 100MB
            resource.setrlimit(resource.RLIMIT_FSIZE, (100*1024*1024, 100*1024*1024))
            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT * 2, TIMEOUT * 2))
            # File descriptor limit
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        except (OSError, ValueError):
            pass
    
    def execute_command(self, command: str, timeout: float = TIMEOUT) -> Tuple[str, str, int]:
        """Execute a command in the persistent bash session context"""
        
        if not command.strip():
            return "", "", 0
        
        self.command_counter += 1
        
        # Create execution script that properly handles state
        script_content = f"""#!/bin/bash
# Load the session environment
if [ -f $HOME/.bashrc ]; then
    source $HOME/.bashrc
fi

# Change to current directory (in case it was changed in previous commands)
if [ -f "$HOME/.current_dir" ]; then
    SAVED_DIR=$(cat "$HOME/.current_dir" 2>/dev/null)
    if [ -d "$SAVED_DIR" ]; then
        cd "$SAVED_DIR"
    fi
fi

# Execute the command
{command}
COMMAND_EXIT_CODE=$?

# Save current directory
pwd > "$HOME/.current_dir" 2>/dev/null

# Save environment variables (exports only)
env | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' | grep -v '^_' | \\
grep -v '^BASH_' | grep -v '^COMP_' | grep -v '^FUNCNAME' | \\
grep -v '^GROUPS' | grep -v '^HOSTNAME' | grep -v '^MACHTYPE' | \\
grep -v '^OLDPWD' | grep -v '^OSTYPE' | grep -v '^PPID' | \\
grep -v '^PWD' | grep -v '^RANDOM' | grep -v '^SECONDS' | \\
grep -v '^SHELLOPTS' | grep -v '^SHLVL' | grep -v '^UID' | \\
grep -v '^EUID' | grep -v '^HISTFILE' | grep -v '^PS1' | \\
grep -v '^PROMPT_COMMAND' | grep -v '^TERM' | grep -v '^PATH' | \\
grep -v '^HOME' | grep -v '^TMPDIR' | grep -v '^LANG' | \\
grep -v '^LC_' | sed 's/^/export /' > "$HOME/.bash_env.new" 2>/dev/null

# Only update if we got some content
if [ -s "$HOME/.bash_env.new" ]; then
    mv "$HOME/.bash_env.new" "$HOME/.bash_env"
else
    rm -f "$HOME/.bash_env.new" 2>/dev/null
fi

# Save aliases
alias 2>/dev/null | grep -v "^alias l[sla]=" > "$HOME/.bash_aliases.new"
if [ -s "$HOME/.bash_aliases.new" ]; then
    mv "$HOME/.bash_aliases.new" "$HOME/.bash_aliases"
else
    rm -f "$HOME/.bash_aliases.new" 2>/dev/null
fi

# Save functions
{{
    declare -F 2>/dev/null | awk '{{print $3}}' | while read func; do
        if [[ "$func" != "_"* ]] && [[ "$func" != "command_not_found_handle" ]]; then
            declare -f "$func" 2>/dev/null
        fi
    done
}} > "$HOME/.bash_functions.new" 2>/dev/null

if [ -s "$HOME/.bash_functions.new" ]; then
    mv "$HOME/.bash_functions.new" "$HOME/.bash_functions"
else
    rm -f "$HOME/.bash_functions.new" 2>/dev/null
fi

# Add to history
echo "{command}" >> "$HOME/.bash_history" 2>/dev/null

exit $COMMAND_EXIT_CODE
"""
        
        # Write the script
        script_path = os.path.join(self.temp_dir, f"cmd_{uuid.uuid4().hex[:8]}.sh")
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
            
            # Prepare environment
            env = self._prepare_environment()
            
            # Build command
            if self.use_firejail and shutil.which("firejail"):
                cmd = [
                    "firejail",
                    "--quiet",
                    "--noprofile",
                    "--private-tmp",
                    f"--private={self.temp_dir}",
                    # "--rlimit-nofile=256",
                    # "--rlimit-fsize=100m",
                    # "--rlimit-as=1g",
                    # "--rlimit-nproc=128",
                    "--net=none",
                    "--nosound",
                    "--no3d",
                    "--nodvd",
                    "--notv",
                    "--nou2f",
                    "bash", os.path.basename(script_path)
                ]
                cwd = None
                env["HOME"] = os.path.expanduser("~")
                env["TMPDIR"] = os.path.expanduser("~")
                # actual home directory, /home/username
                self.home_dir = os.path.expanduser("~")
            else:
                cmd = ["bash", script_path]
                env["HOME"] = self.temp_dir
                env["TMPDIR"] = self.temp_dir
                cwd = self.temp_dir
                self.home_dir = self.temp_dir
            
            # Execute the command
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    cwd=cwd,
                    text=True,
                    timeout=timeout,
                )
                
                stdout = result.stdout.rstrip() if result.stdout else ""
                stderr = result.stderr.rstrip() if result.stderr else ""
                exit_code = result.returncode
                
                if exit_code == 124:
                    stderr += f"\\nCommand timed out after {timeout} seconds"
                
                # Update current directory from saved state
                self._update_current_dir()
                
                return stdout, stderr, exit_code
                
            except subprocess.TimeoutExpired:
                return "", f"Process timed out after {timeout} seconds", 124
                
        except Exception as e:
            return "", f"Error executing command: {str(e)}", 1
            
        finally:
            # Clean up the temporary script
            try:
                if os.path.exists(script_path):
                    os.remove(script_path)
            except Exception:
                pass
    
    def execute_command_like_shell(self, commands: Union[str, List[str]], timeout: float = TIMEOUT) -> str:
        """Execute a command in the session, simulating a shell-like environment"""
        terminal_outputs = ""
        if isinstance(commands, str):
            commands = [commands]
        for cmd in commands:
            terminal_outputs += f"{self.get_prompt()}{cmd}\n"
            stdout, stderr, exit_code = self.execute_command(cmd, timeout)
            output = format_output(stdout, stderr, exit_code)
            if output:
                # Remove the prompt line that was echoed in the output
                lines = output.split('\n')
                if lines and lines[0].endswith(cmd):
                    output = '\n'.join(lines[1:])
                terminal_outputs += output + "\n"
        return terminal_outputs
            
    def _update_current_dir(self):
        """Update current directory from saved state"""
        try:
            current_dir_file = os.path.join(self.temp_dir, ".current_dir")
            if os.path.exists(current_dir_file):
                with open(current_dir_file, 'r') as f:
                    saved_dir = f.read().strip()
                    if os.path.exists(saved_dir):
                        self.current_dir = saved_dir
        except Exception:
            pass
    
    def get_prompt(self) -> str:
        """Get the current shell prompt"""
        try:
            # Get relative path for display
            if self.current_dir == self.home_dir:
                path_display = "~"
            elif self.current_dir.startswith(self.home_dir):
                path_display = "~" + self.current_dir[len(self.home_dir):]
            else:
                path_display = self.current_dir
            
            return f"user@bash-session:{path_display}$ "
        except:
            return "user@bash-session:~$ "
    
    def get_history(self) -> List[str]:
        """Get command history"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return [line.strip() for line in f.readlines() if line.strip()]
            return []
        except:
            return []
    
    def cleanup(self):
        """Clean up the session"""
        files_to_remove = [
            self.bashrc_file,
            self.history_file,
            os.path.join(self.temp_dir, ".bash_env"),
            os.path.join(self.temp_dir, ".bash_aliases"),
            os.path.join(self.temp_dir, ".bash_functions"),
            os.path.join(self.temp_dir, ".current_dir")
        ]
        
        for file_path in files_to_remove:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass

# Example usage and testing
if __name__ == "__main__":
    # Create session
    temp_dir = os.path.join(os.getcwd(), "tmp", "bash", "enhanced_test")
    os.makedirs(temp_dir, exist_ok=True)
    session = BashSession(temp_dir, use_firejail=True)
    
    # Test commands with persistent variables
    test_commands = [
        "ls -la",
        "export MY_VAR='Hello World'",
        "echo $MY_VAR",
        "mkdir test_dir",
        "cd test_dir",
        "pwd",
        "echo 'test content' > test_file.txt",
        "cat test_file.txt",
        "MY_LOCAL_VAR='Local Variable'",
        "echo $MY_LOCAL_VAR",
        "alias ll='ls -la'",
        "ll",
        "function greet() { echo \"Hello, $1!\"; }",
        "greet World",
        "cd ..",
        "pwd",
        "echo $MY_VAR",  # Should still be available
        "history | tail -5"
    ]
    
    print("Enhanced Bash Terminal Session")
    print("=" * 40)
    
    print(session.execute_command_like_shell(test_commands))
    
    print(f"\n{session.get_prompt()}", end="")
    print("[Session ended]")
    
    # Cleanup
    session.cleanup()

"""
python verl_tool/servers/tools/utils/bash_session.py
"""