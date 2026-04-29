# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Enhanced Travel Agent Tools
Combines mock tools with real hotel database access
"""
import os
from strands import tool

# Try to import Neo4j tools from the Graph-RAG demo (Module 1)
try:
    import sys
    sys.path.append('../01-graphrag-demo/tools')
    from graph_tool import search_hotels_by_country, get_top_rated_hotels
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# ============================================================================
# HOTEL TOOLS (Real Database + Mock)
# ============================================================================

@tool
def search_real_hotels(country: str, min_rating: float = 0.0) -> str:
    """Search real hotels in a specific country from our database."""
    if not NEO4J_AVAILABLE:
        return f"Mock: Hotels in {country} with rating >= {min_rating}"
    try:
        results = search_hotels_by_country(country, min_rating)
        return results
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_top_hotels(limit: int = 5) -> str:
    """Get the top-rated hotels from our database."""
    if not NEO4J_AVAILABLE:
        return f"Mock: Top {limit} hotels"
    try:
        results = get_top_rated_hotels(limit)
        return results
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def search_hotels(query: str) -> str:
    """Search for hotels by location name or city. Returns a list of available hotels matching the search criteria."""
    return f"Hotels found for: {query}"

@tool
def search_hotel_reviews(hotel: str) -> str:
    """Search and read customer reviews and ratings for a specific hotel. Useful for checking hotel quality and guest experiences."""
    return f"Reviews for {hotel}: 4.5 stars"

@tool
def get_hotel_details(hotel: str) -> str:
    """Get comprehensive hotel information including amenities, facilities, services, and room types. Does not include pricing."""
    return f"{hotel}: Pool, Spa, $200/night"

@tool
def get_hotel_pricing(hotel: str) -> str:
    """Get current room rates and pricing information for a specific hotel. Returns price ranges for different room types."""
    return f"{hotel}: $200-400/night"

@tool
def check_hotel_availability(hotel: str, date: str) -> str:
    """Check if a hotel has available rooms on a specific single date. For date ranges, use check_hotel_availability_dates instead."""
    return f"{hotel} available on {date}"

@tool
def book_hotel(hotel: str, guest: str) -> str:
    """Make a hotel reservation and book a room for a guest. Completes the booking process."""
    return f"BOOKED {hotel} for {guest}"

@tool
def check_hotel_availability_dates(hotel_name: str, check_in: str, check_out: str) -> str:
    """Check real-time hotel room availability for specific dates."""
    import secrets
    from datetime import datetime
    
    try:
        checkin_date = datetime.strptime(check_in, '%Y-%m-%d')
        checkout_date = datetime.strptime(check_out, '%Y-%m-%d')
        nights = (checkout_date - checkin_date).days
        
        if nights <= 0:
            return f"Error: Check-out must be after check-in"
        
        available = secrets.randbelow(10) > 3
        rooms_left = secrets.randbelow(8) + 1 if available else 0
        
        if available:
            price_per_night = secrets.randbelow(251) + 150
            total = price_per_night * nights
            return f"{hotel_name}: AVAILABLE - {rooms_left} rooms left, ${price_per_night}/night, Total: ${total} for {nights} nights"
        else:
            return f"{hotel_name}: SOLD OUT for {check_in} to {check_out}"
    except ValueError:
        return "Error: Use date format YYYY-MM-DD"

@tool
def compare_hotel_prices(city: str, check_in: str, check_out: str) -> str:
    """Compare prices across multiple hotels in a city for specific dates."""
    import secrets
    
    hotels = ["AnyCompany Lisbon Resort", "AnyCompany Porto City Hotel", "AnyCompany Faro Beach"]
    results = []
    for hotel in hotels:
        price = secrets.randbelow(231) + 120
        rating = round((secrets.randbelow(16) + 80) / 10, 1)
        results.append(f"{hotel}: ${price}/night (Rating: {rating}/10)")
    
    return f"Price comparison for {city} ({check_in} to {check_out}):\n" + "\n".join(results)

# ============================================================================
# FLIGHT TOOLS (Mock)
# ============================================================================

@tool
def search_flights(origin: str, dest: str) -> str:
    """Search for available flights between two cities. Returns flight options with times and airlines."""
    return f"Flights {origin}-{dest}: $300-500"

@tool
def search_flight_prices(origin: str, dest: str) -> str:
    """Get price comparison for flights between two cities. Shows price ranges across different airlines and times."""
    return f"Prices {origin}-{dest}: $300-500"

@tool
def get_flight_details(flight: str) -> str:
    """Get detailed information about a specific flight including aircraft type, duration, and route."""
    return f"Flight {flight}: Boeing 737, 3h"

@tool
def get_flight_status(flight: str) -> str:
    """Check real-time flight status including delays, gate information, and departure/arrival times."""
    return f"Flight {flight}: On time, Gate B4"

@tool
def check_flight_availability(flight: str) -> str:
    """Check how many seats are available on a specific flight. Useful for group bookings."""
    return f"Flight {flight}: 23 seats left"

@tool
def book_flight(flight: str, passenger: str) -> str:
    """Make a flight reservation and book a seat for a passenger. Completes the booking process."""
    return f"BOOKED {flight} for {passenger}"

# ============================================================================
# WEATHER TOOLS (Mock)
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather."""
    return f"{city}: 22°C, Sunny"

@tool
def get_weather_forecast(city: str) -> str:
    """Get weather forecast."""
    return f"{city} forecast: 22°C today, 20°C tomorrow"

@tool
def get_weather_alerts(city: str) -> str:
    """Get weather alerts."""
    return f"{city}: No alerts"

# ============================================================================
# PAYMENT TOOLS (Mock)
# ============================================================================

@tool
def process_payment(amount: float) -> str:
    """Process a payment."""
    return f"PAID ${amount}"

@tool
def check_payment(transaction_id: str) -> str:
    """Check payment status."""
    return f"Transaction {transaction_id}: Complete"

@tool
def refund_payment(transaction_id: str) -> str:
    """Refund a payment."""
    return f"REFUNDED {transaction_id}"

# ============================================================================
# TRAVEL UTILITY TOOLS
# ============================================================================

@tool
def get_currency_exchange(from_currency: str, to_currency: str, amount: float) -> str:
    """Convert currency for international travel."""
    rates = {
        ('USD', 'EUR'): 0.92,
        ('EUR', 'USD'): 1.09,
        ('USD', 'GBP'): 0.79,
        ('GBP', 'USD'): 1.27,
        ('EUR', 'GBP'): 0.86,
        ('GBP', 'EUR'): 1.16,
    }
    rate = rates.get((from_currency, to_currency), 1.0)
    converted = amount * rate
    return f"{amount} {from_currency} = {converted:.2f} {to_currency} (rate: {rate})"

@tool
def get_travel_documents(destination_country: str, origin_country: str) -> str:
    """Get visa and travel document requirements."""
    schengen_countries = ['France', 'Spain', 'Italy', 'Netherlands', 'Germany', 'Portugal']
    visa_free_origins = ['USA', 'Canada', 'UK', 'Australia', 'Japan', 'Brazil', 'Mexico']
    if destination_country in schengen_countries:
        if origin_country in visa_free_origins:
            return f"{origin_country} passport holders can visit {destination_country} visa-free for up to 90 days (Schengen area). Valid passport required."
    return f"Check embassy website for {destination_country} visa requirements from {origin_country}."

# ============================================================================
# GENERIC/AMBIGUOUS TOOLS (High confusion risk)
# ============================================================================

@tool
def search(query: str) -> str:
    """Perform a generic search across all travel services. Use this only when the search type is unclear or spans multiple categories."""
    return f"Results for: {query}"

@tool
def check(item: str) -> str:
    """Perform a generic check on any item. Use this only when the specific check type is unclear."""
    return f"Checked: {item}"

@tool
def get_details(item: str) -> str:
    """Get general details about any travel item. Use this only when you need generic information that doesn't fit specific categories."""
    return f"Details: {item}"

@tool
def get_status(item: str) -> str:
    """Check the general status of any travel item. Use this only when the specific status type is unclear."""
    return f"Status: {item} OK"

@tool
def get_info(item: str) -> str:
    """Get general information about any travel item. Use this only when you need generic info that doesn't fit specific categories."""
    return f"Info: {item}"

@tool
def book(item: str, name: str) -> str:
    """Make a generic booking for any travel service. Use this only when the booking type is unclear or spans multiple services."""
    return f"BOOKED {item} for {name}"

@tool
def cancel(item: str) -> str:
    """Cancel any travel booking or reservation. Use this for general cancellations when the specific type is unclear."""
    return f"CANCELLED {item}"

# ============================================================================
# ALL TOOLS COLLECTION
# ============================================================================

ALL_TOOLS = []

# Add real database tools if Neo4j is available
if NEO4J_AVAILABLE:
    ALL_TOOLS.extend([search_real_hotels, get_top_hotels])

# Add all other tools
ALL_TOOLS.extend([
    # Hotel tools (mock)
    search_hotels, search_hotel_reviews, get_hotel_details, get_hotel_pricing, 
    check_hotel_availability, book_hotel, check_hotel_availability_dates, compare_hotel_prices,
    
    # Flight tools
    search_flights, search_flight_prices, get_flight_details, get_flight_status, 
    check_flight_availability, book_flight,
    
    # Weather tools
    get_weather, get_weather_forecast, get_weather_alerts,
    
    # Payment tools
    process_payment, check_payment, refund_payment,
    
    # Travel utilities
    get_currency_exchange, get_travel_documents,
    
    # Generic/ambiguous tools
    search, check, get_details, get_status, get_info, book, cancel
])
