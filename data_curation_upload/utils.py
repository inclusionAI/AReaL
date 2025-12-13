def label_steps(text: str) -> str:
    """
    Separates a string by '\\n\\n', labels each part with 'Step i', 
    and concatenates them into one string.
    
    Args:
        text: Input string to be labeled
        
    Returns:
        String with each part labeled with 'Step i' where i starts from 1
    """
    parts = text.split('\n\n')
    labeled_parts = [f"Step {i+1}\n{part}" for i, part in enumerate(parts)]
    return '\n\n'.join(labeled_parts)
