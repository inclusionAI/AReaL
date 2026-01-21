"""
Airline environment implementation based on TAU bench.
"""

from typing import Optional
try:
    from .data_model import FlightDB, get_db
    from .tools import AirlineTools
except ImportError:
    from data_model import FlightDB, get_db
    from tools import AirlineTools


class AirlineEnvironment:
    """Airline environment for tool use tasks."""
    
    def __init__(self, db: Optional[FlightDB] = None):
        """Initialize the airline environment.
        
        Args:
            db: Optional FlightDB instance. If None, loads from default location.
        """
        if db is None:
            db = get_db()
        self.db = db
        self.tools = AirlineTools(db)
        self.domain_name = "airline"
        
    def get_tools(self) -> AirlineTools:
        """Get the airline tools instance."""
        return self.tools
        
    def get_statistics(self) -> dict:
        """Get statistics about the environment."""
        return self.db.get_statistics()
        
    def reset(self):
        """Reset the environment to initial state."""
        # Reload the database from disk
        self.db = get_db()
        self.tools = AirlineTools(self.db)
        
    def save_state(self, filepath: str = None):
        """Save the current state of the environment."""
        if filepath is None:
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(current_dir, "db.json")
        self.db.save(filepath)


def get_airline_environment(db: Optional[FlightDB] = None) -> AirlineEnvironment:
    """Get the airline environment instance.
    
    Args:
        db: Optional FlightDB instance.
        
    Returns:
        AirlineEnvironment instance.
    """
    return AirlineEnvironment(db)


if __name__ == "__main__":
    env = get_airline_environment()
    print(f"Environment statistics: {env.get_statistics()}")
    print(f"Available tools: {[name for name in dir(env.tools) if not name.startswith('_')]}")
