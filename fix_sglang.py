import re

with open("areal/engine/sglang_remote.py", "r") as f:
    code = f.read()

# Add missing imports if needed
if "from areal.infra.platforms import current_platform" not in code:
    code = code.replace("from areal.utils.network import format_host_for_url", 
                        "from areal.infra.platforms import current_platform\nfrom areal.utils.network import format_host_for_url")

# Read the fix file
with open("/Users/bytedance/Downloads/fixes/02_sglang_remote_fix.py", "r") as f:
    fix_code = f.read()

# Extract the fixed methods
init_req_match = re.search(r"def build_init_weights_group_request\(.*?return HttpRequest\(endpoint=\"/init_weights_update_group\", payload=payload\)", fix_code, re.DOTALL)
update_req_match = re.search(r"def build_update_weights_from_distributed_request\(.*?return HttpRequest\(endpoint=\"/update_weights_from_distributed\", payload=payload\)", fix_code, re.DOTALL)

init_req = "    " + init_req_match.group(0).replace("\n", "\n    ")
update_req = "    " + update_req_match.group(0).replace("\n", "\n    ")

# Find the existing methods in code
old_init_req = re.search(r"    def build_init_weights_group_request\(.*?return server_address, request", code, re.DOTALL).group(0)
old_update_req = re.search(r"    def build_update_weights_from_distributed_request\(.*?return server_address, request", code, re.DOTALL).group(0)

# Replace them
code = code.replace(old_init_req, init_req)
code = code.replace(old_update_req, update_req)

with open("areal/engine/sglang_remote.py", "w") as f:
    f.write(code)

