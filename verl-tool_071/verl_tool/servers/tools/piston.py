import asyncio
import aiohttp
import json
import re
import xml.etree.ElementTree as ET
from .base import BaseTool, register_tool
import logging

logger = logging.getLogger(__name__)

@register_tool
class PistonTool(BaseTool):
    tool_type = "piston"
    
    def __init__(self, num_workers=1, api_url=None, use_local=False):
        super().__init__(num_workers)
        
        # Determine API URL
        if api_url is not None:
            self.api_url = api_url
            self.is_public_api = "emkc.org" in api_url
        elif use_local:
            self.api_url = "http://localhost:2000/api/v2"
            self.is_public_api = False
        else:
            # Default to public API
            self.api_url = "https://emkc.org/api/v2/piston"
            self.is_public_api = True
            self._show_public_api_info()
        
        # Test connection
        try:
            asyncio.get_event_loop().run_until_complete(self._test_connection())
            if self.is_public_api:
                logger.info("✅ Connected to Piston public API")
            else:
                logger.info("✅ Connected to local Piston API")
        except ConnectionError as e:
            if not self.is_public_api:
                self._show_docker_guide()
            raise e
    
    def _show_docker_guide(self):
        """Display Docker startup guide"""
        guide = """
❌ Piston Docker container is not running or API is not accessible

Please follow these steps to start the Piston Docker container:

1. System requirements:
   - Make sure Docker is installed and running
   - Ensure cgroup v2 is enabled and cgroup v1 is disabled (critical for Piston)

2. Run the following command to start the Piston container:

   docker run --privileged -dit -p 2000:2000 --name piston_api ghcr.io/engineer-man/piston

   Note: The --privileged parameter is required for sandboxing

3. Wait for the container to start (usually takes a few seconds)

4. Check if the container is running:

   docker ps | grep piston_api

If the container exists but is stopped, you can start it with:

   docker start piston_api

After starting the container, please restart this service.

For Python client (optional):
   pip install pyston
"""
        logger.error(guide)
    
    def _show_public_api_info(self):
        """Display public API information and rate limits"""
        info = """
Piston Public API Information:
- Using https://emkc.org/api/v2/piston
- Rate limited to 5 requests per second
- For higher usage needs, consider self-hosting

To use local instance:
PistonTool(use_local=True) or PistonTool(api_url="http://localhost:2000/api/v2")
"""
        logger.info(info)
    
    def _get_api_endpoint(self, endpoint):
        """Build full endpoint path based on API URL"""
        if self.is_public_api:
            # Public API endpoint format already includes /api/v2/piston
            return f"{self.api_url}/{endpoint}"
        else:
            # Local API may or may not include /api/v2
            if "/api/v2" in self.api_url:
                return f"{self.api_url}/{endpoint}"
            else:
                return f"{self.api_url}/api/v2/{endpoint}"
    
    async def _test_connection(self):
        """Test connection to the Piston API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = self._get_api_endpoint("runtimes")
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ConnectionError(f"Failed to connect to Piston API: HTTP {response.status}")
                    
                    # Get list of available runtimes for info
                    runtimes = await response.json()
                    languages = [f"{r['language']} ({r['version']})" for r in runtimes[:5]]
                    logger.info(f"Piston API connected. Available languages (showing 5 of {len(runtimes)}): {', '.join(languages)}...")
                        
        except aiohttp.ClientConnectorError:
            raise ConnectionError("Cannot connect to Piston API. Is the Docker container running?")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Piston API: {str(e)}")
    
    def parse_action(self, action:str):
        """Parse action string in either XML or JSON format"""
        action = action.strip()
        
        # Try to parse as XML format
        if action.startswith("<piston>") and action.endswith("</piston>"):
            return self._parse_xml_action(action)
        
        # Try to parse as JSON format
        elif action.startswith("{") and action.endswith("}"):
            return self._parse_json_action(action)
        
        # Invalid format
        else:
            logger.error("Unrecognized action format")
            return None, False
    
    def _parse_xml_action(self, action:str):
        """Parse XML formatted action"""
        try:
            # Process XML
            root = ET.fromstring(action)
            if root.tag != "piston":
                return None, False
            
            parsed = {}
            
            # Parse basic attributes
            for elem in root:
                if elem.tag in ["language", "version", "args", "stdin"]:
                    parsed[elem.tag] = elem.text.strip() if elem.text else ""
                elif elem.tag == "file":
                    if "files" not in parsed:
                        parsed["files"] = []
                    
                    filename = elem.get("name", f"file{len(parsed['files'])}")
                    content = elem.text if elem.text else ""
                    
                    parsed["files"].append({
                        "name": filename,
                        "content": content
                    })
            
            # Ensure required fields exist
            if "language" not in parsed:
                logger.error("Missing required language field")
                return None, False
                
            if "files" not in parsed or len(parsed["files"]) == 0:
                logger.error("Missing file content")
                return None, False
                
            # Process args
            if "args" in parsed:
                parsed["args"] = parsed["args"].split()
                
            return parsed, True
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            return None, False
        except Exception as e:
            logger.error(f"Error parsing XML action: {str(e)}")
            return None, False
    
    def _parse_json_action(self, action:str):
        """Parse JSON formatted action"""
        try:
            parsed = json.loads(action)
            
            # Ensure required fields exist
            if "language" not in parsed:
                logger.error("Missing required language field")
                return None, False
                
            if "files" not in parsed or not isinstance(parsed["files"], list) or len(parsed["files"]) == 0:
                logger.error("Missing file content or files field is not a valid array")
                return None, False
                
            # Validate files structure
            for i, file in enumerate(parsed["files"]):
                if not isinstance(file, dict) or "content" not in file:
                    logger.error(f"File #{i+1} is missing content or has invalid format")
                    return None, False
                    
                if "name" not in file:
                    # Generate default filename
                    extension = self._get_extension_for_language(parsed["language"])
                    file["name"] = f"file{i}{extension}"
            
            return parsed, True
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return None, False
        except Exception as e:
            logger.error(f"Error parsing JSON action: {str(e)}")
            return None, False
    
    def _get_extension_for_language(self, language):
        """Get file extension for a given language"""
        extensions = {
            "python": ".py",
            "javascript": ".js", 
            "typescript": ".ts",
            "java": ".java",
            "c": ".c",
            "cpp": ".cpp",
            "csharp": ".cs",
            "go": ".go",
            "rust": ".rs",
            "ruby": ".rb",
            "php": ".php",
            "swift": ".swift",
            "kotlin": ".kt",
            "scala": ".scala"
        }
        
        return extensions.get(language.lower(), f".{language}")
    
    async def _execute_code(self, parsed_action):
        """Execute code and return result"""
        try:
            language = parsed_action.get("language")
            version = parsed_action.get("version", "*")
            args = parsed_action.get("args", [])
            stdin = parsed_action.get("stdin", "")
            files = parsed_action.get("files", [])
            
            payload = {
                "language": language,
                "version": version,
                "files": files,
                "stdin": stdin,
                "args": args,
                "compile_timeout": 10000,
                "run_timeout": 3000,
                "compile_memory_limit": -1,
                "run_memory_limit": -1
            }
            
            async with aiohttp.ClientSession() as session:
                url = self._get_api_endpoint("execute")
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        # Handle rate limiting
                        if self.is_public_api and response.status == 429:
                            retry_after = response.headers.get('Retry-After', '60')
                            return {"error": f"Rate limit exceeded. Try again after {retry_after} seconds."}
                            
                        error_text = await response.text()
                        return {"error": f"HTTP {response.status}: {error_text}"}
                    
                    result = await response.json()
                    return result
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}")
            return {"error": f"Failed to execute code: {str(e)}"}
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """Execute action and return observation result"""
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = """
Invalid action format. Supported formats:

1. XML format:
<piston>
  <language>python</language>
  <version>3.9</version>
  <args>arg1 arg2</args>
  <stdin>input data</stdin>
  <file name="main.py">
print("Hello, World!")
for i in range(5):
    print(f"Number {i}")
  </file>
  <file name="helper.py">
def add(a, b):
    return a + b
  </file>
</piston>

2. JSON format:
{
  "language": "python",
  "version": "3.9",
  "args": ["arg1", "arg2"],
  "stdin": "input data",
  "files": [
    {
      "name": "main.py",
      "content": "print('Hello, World!')\nfor i in range(5):\n    print(f'Number {i}')"
    },
    {
      "name": "helper.py",
      "content": "def add(a, b):\n    return a + b"
    }
  ]
}
"""
            done = True
            valid = False
        else:
            try:
                # Create a new event loop for this thread if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No event loop in current thread, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Execute code
                result = loop.run_until_complete(self._execute_code(parsed_action))
                
                # Format output
                if "error" in result:
                    observation = f"Error: {result['error']}"
                    valid = False
                elif "run" in result:
                    stdout = result["run"].get("stdout", "")
                    stderr = result["run"].get("stderr", "")
                    code = result["run"].get("code")
                    signal = result["run"].get("signal")
                    cpu_time = result["run"].get("cpu_time", 0)
                    memory = result["run"].get("memory", 0)
                    
                    status_msg = ""
                    if result["run"].get("status"):
                        status_msg = f" ({result['run']['status']})"
                    
                    observation = f"""Execution result:

Language: {parsed_action.get('language')}
Version: {result.get('version', parsed_action.get('version', '*'))}

--- STDOUT ---
{stdout}

--- STDERR ---
{stderr}

Exit code: {code}{status_msg}
Signal: {signal if signal else 'None'}
CPU time: {cpu_time}ms
Memory usage: {memory/1000000:.2f}MB
"""
                    valid = True
                elif "compile" in result and result["compile"].get("status") is not None:
                    # Compilation error
                    stdout = result["compile"].get("stdout", "")
                    stderr = result["compile"].get("stderr", "")
                    code = result["compile"].get("code")
                    
                    observation = f"""Compilation error:

--- Compile output ---
{stdout}

--- Compile error ---
{stderr}

Compilation exit code: {code}
Status: {result["compile"].get("status", "Unknown")}
"""
                    valid = True
                else:
                    observation = f"Unknown result format: {json.dumps(result, indent=2)}"
                    valid = False
                
                done = True
            except Exception as e:
                observation = f"Error executing code: {str(e)}"
                done = True
                valid = False
        
        self.update_env(trajectory_id, env, parsed_action if is_valid else action, 
                        is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
