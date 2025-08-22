# TAU Airline Environment Implementation

This directory contains a complete implementation of the airline environment based on the TAU (Tool use in Agent-based systems) benchmark. The implementation provides a realistic airline reservation system that can be used for testing tool use capabilities in AI systems.

## Overview

The airline environment simulates a complex airline reservation system with:
- 300 flights with 9,000 flight instances across different dates
- 500 users with various membership levels and payment methods
- 2,000 existing reservations
- Full booking, modification, and cancellation workflows

## Components

### Core Files

1. **`data_model.py`** - Pydantic models for all airline entities
   - Flight, User, Reservation, Payment models
   - Type definitions for cabin classes, membership levels, etc.
   - Database loading and management functionality

2. **`tools.py`** - Complete set of airline tools
   - Flight search (direct and one-stop)
   - Booking and reservation management
   - User and reservation details retrieval
   - Payment processing and certificate handling
   - Cancellation and refund processing

3. **`environment.py`** - Environment wrapper and management
   - Environment initialization and state management
   - Tool access and coordination
   - Save/load functionality

4. **`__init__.py`** - Package initialization and exports

### Data Files

- **`db.json`** - Complete airline database (205,193 lines)
  - Flight schedules and availability
  - User profiles and payment methods
  - Existing reservations and transaction history

### Demo and Documentation

- **`airline_environment_demo.ipynb`** - Comprehensive usage examples
  - Environment setup and initialization
  - Flight search and booking operations
  - Reservation management and modifications
  - Error handling and edge cases

## Key Features

### Flight Operations
- **Search direct flights** between airports on specific dates
- **Search one-stop flights** with connection handling
- **Flight status checking** (available, cancelled, delayed, etc.)
- **Real-time seat availability** and pricing

### Booking System
- **Complete reservation creation** with validation
- **Multi-passenger support** with individual details
- **Payment processing** (credit cards, gift cards, certificates)
- **Baggage and insurance** options

### Reservation Management
- **Update passenger information** 
- **Modify flight itineraries** with price adjustments
- **Add/remove baggage** with fee calculation
- **Cabin class upgrades** and changes

### User Services
- **User profile management** with membership levels
- **Payment method handling** (multiple types supported)
- **Certificate issuance** for compensation
- **Human agent transfer** for complex issues

### Business Logic
- **Membership-based benefits** (Gold, Silver, Regular)
- **Dynamic pricing** based on availability
- **Cancellation policies** with refund processing
- **Seat inventory management**

## Usage Examples

### Basic Setup
```python
from airline_env import get_airline_environment

# Initialize environment
env = get_airline_environment()
tools = env.get_tools()

# Get environment statistics
print(env.get_statistics())
```

### Flight Search
```python
# Search for direct flights
flights = tools.search_direct_flight("JFK", "LAX", "2024-05-20")

# Search for one-stop flights
onestop = tools.search_onestop_flight("JFK", "LAX", "2024-05-20")
```

### User and Reservation Management
```python
# Get user details
user = tools.get_user_details("mia_li_3668")

# Get reservation details  
reservation = tools.get_reservation_details("ABC123")

# Update reservation
updated = tools.update_reservation_passengers(
    reservation_id="ABC123",
    passengers=[{"first_name": "John", "last_name": "Doe", "dob": "1990-01-01"}]
)
```

### Booking Operations
```python
# Create new booking
reservation = tools.book_reservation(
    user_id="mia_li_3668",
    origin="JFK",
    destination="LAX", 
    flight_type="one_way",
    cabin="economy",
    flights=[{"flight_number": "HAT001", "date": "2024-05-20"}],
    passengers=[{"first_name": "John", "last_name": "Doe", "dob": "1990-01-01"}],
    payment_methods=[{"payment_id": "credit_card_123", "amount": 299}],
    total_baggages=1,
    nonfree_baggages=0,
    insurance="no"
)
```

## Error Handling

The implementation includes comprehensive error handling for:
- Non-existent users, flights, or reservations
- Insufficient payment balances
- Seat availability constraints
- Invalid flight dates or routes
- Business rule violations

## Data Integrity

- **Atomic operations** - All booking/cancellation operations are transactional
- **Seat inventory tracking** - Prevents overbooking
- **Payment validation** - Ensures sufficient funds before processing
- **Referential integrity** - Maintains consistency between users and reservations

## Testing

Use the provided Jupyter notebook (`airline_environment_demo.ipynb`) to:
- Explore the environment capabilities
- Test different scenarios and edge cases
- Understand the API and data structures
- Validate implementations

## Integration

This implementation can be easily integrated into:
- **Tool use evaluation frameworks**
- **AI agent testing systems** 
- **Chatbot backends**
- **API service layers**

The modular design allows for easy customization and extension while maintaining compatibility with the TAU benchmark specification.

## Dependencies

- **pydantic** - For data validation and serialization
- **loguru** - For logging (optional, can be replaced)
- **Python 3.8+** - For modern type hints and features

## Architecture

The implementation follows a clean architecture pattern:
- **Data layer** (`data_model.py`) - Pure data structures and validation
- **Business layer** (`tools.py`) - Core airline operations and logic  
- **Service layer** (`environment.py`) - Environment management and coordination
- **Interface layer** (`__init__.py`) - Public API and exports

This separation ensures maintainability, testability, and easy integration into larger systems.
