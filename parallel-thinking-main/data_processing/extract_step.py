file = open('temp.txt', 'r')


# content = file.read()
# print(content)
import re
def extract_steps_regex(content):
    # This pattern matches "Step <number>: " at the start of a line
    step_pattern = re.compile(r'Step \d+', re.MULTILINE)
    
    # Split the content by step headers
    steps = step_pattern.split(content)
    
    # The first element is empty (content before first step), so we ignore it
    steps = steps[1:]
    
    # Get all the step headers
    step_headers = step_pattern.findall(content)
    
    # Combine headers with their content
    result = []
    for header, step_content in zip(step_headers, steps):
        # Remove leading/trailing whitespace and combine header with content
        full_step = f"{step_content.strip()}"
        result.append(full_step)
    
    return result
# result = extract_steps_regex(content)
# print (result[14])
# print(len(result))