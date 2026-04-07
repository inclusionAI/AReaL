import regex as re
def replace_consecutive_tokens(text, token='<|image_pad|>'):
    """
    Replace consecutive tokens with compressed format token * n
    
    Args:
        text (str): Input string that may contain consecutive tokens
        token (str): The token to look for and replace (default: '<|image_pad|>')
    
    Returns:
        str: String with consecutive tokens replaced by compressed format
    """
    # Escape special regex characters in the token
    escaped_token = re.escape(token)
    
    # Pattern to match consecutive tokens
    pattern = f'(?:{escaped_token})+'
    
    def replacement_func(match):
        # Count how many consecutive tokens were found
        matched_text = match.group(0)
        count = matched_text.count(token)
        
        # If only one token, return as is
        if count == 1:
            return token
        else:
            # Return compressed format
            return f'{token}*{count}'
    
    # Replace all consecutive occurrences
    result = re.sub(pattern, replacement_func, text)
    return result
