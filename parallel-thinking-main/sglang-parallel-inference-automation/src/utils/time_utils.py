def get_current_timestamp() -> str:
    """Get the current timestamp as a formatted string."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def format_timestamp(timestamp: str) -> str:
    """Format a given timestamp string into a more readable format."""
    from datetime import datetime
    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    return dt.strftime("%B %d, %Y at %I:%M %p")

def get_timestamped_filename(base_name: str) -> str:
    """Generate a filename with a timestamp."""
    timestamp = get_current_timestamp()
    return f"{base_name}_{timestamp}.txt"