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
    from graph_tool import query_hotel_knowledge_graph
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    # Create a mock version if import fails
    def query_hotel_knowledge_graph(cypher_query: str) -> str:
        return f"Mock: Would execute Cypher query: {cypher_query[:100]}..."

# ============================================================================
# HOTEL TOOLS (Real Database + Mock)
# ============================================================================

@tool
def search_real_hotels(country: str, min_rating: float = 0.0) -> str:
    """Search actual verified hotels from real hotel database by country. Queries knowledge graph for authentic hotel data in France, Spain, Italy, Germany, Japan, etc. Returns real hotel names, addresses, verified ratings. Use when user specifically says 'real hotels' or wants database-verified properties."""
    if not NEO4J_AVAILABLE:
        return f"Mock: Hotels in {country} with rating >= {min_rating}"
    try:
        # Generate Cypher query with correct snake_case property names
        query = f"""
        MATCH (h:Hotel)
        WHERE h.address CONTAINS '{country}' OR h.name CONTAINS '{country}'
        AND coalesce(h.guest_rating, 0) >= {min_rating}
        RETURN h.name AS name, h.address AS address, h.guest_rating AS rating, h.total_rooms AS rooms
        ORDER BY h.guest_rating DESC
        LIMIT 10
        """
        results = query_hotel_knowledge_graph(query)
        return results
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_top_hotels(limit: int = 5) -> str:
    """Get highest-rated best hotels globally from database. Shows top luxury hotels, best reviewed properties, highest guest satisfaction ratings worldwide. Use for 'best hotels' or 'top rated hotels' queries."""
    if not NEO4J_AVAILABLE:
        return f"Mock: Top {limit} hotels"
    try:
        # Generate Cypher query with correct snake_case property names
        query = f"""
        MATCH (h:Hotel)
        WHERE h.guest_rating IS NOT NULL
        RETURN h.name AS name, h.address AS address, h.guest_rating AS rating, h.total_rooms AS rooms
        ORDER BY h.guest_rating DESC
        LIMIT {limit}
        """
        results = query_hotel_knowledge_graph(query)
        return results
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def search_hotels(query: str) -> str:
    """Find hotels by city location or destination. Use this for general hotel searches when you need to browse accommodation options in a specific place like Barcelona, Paris, London, or Tokyo."""
    return f"Hotels found for: {query}"

@tool
def search_hotel_reviews(hotel: str) -> str:
    """Read guest reviews, customer feedback, ratings and experiences about a specific hotel property. Use when checking quality, service ratings, or what previous guests said."""
    return f"Reviews for {hotel}: 4.5 stars"

@tool
def get_hotel_details(hotel: str) -> str:
    """View hotel amenities, facilities, room types, and services available. Shows what the hotel offers: pool, spa, gym, wifi, parking, restaurants. Does NOT show prices."""
    return f"{hotel}: Pool, Spa, $200/night"

@tool
def get_hotel_pricing(hotel: str) -> str:
    """Check hotel room rates, nightly prices, and cost information. Shows price ranges for different room categories. Use when asking 'how much does it cost'."""
    return f"{hotel}: $200-400/night"

@tool
def check_hotel_availability(hotel: str, date: str) -> str:
    """Check hotel room availability for a single specific date. Use for questions like 'is the hotel available tomorrow' or 'rooms available on July 15'. For date ranges use check_hotel_availability_dates."""
    return f"{hotel} available on {date}"

@tool
def book_hotel(hotel: str, guest: str) -> str:
    """Reserve and book a hotel room. Complete hotel reservation process. Creates a confirmed booking for guest accommodation."""
    return f"BOOKED {hotel} for {guest}"

@tool
def check_hotel_availability_dates(hotel_name: str, check_in: str, check_out: str) -> str:
    """Check hotel availability for date ranges with check-in and check-out dates. Use for multi-day stays like 'March 15 to 18' or 'weekend availability'."""
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
    """Compare and contrast prices across multiple different hotels in the same city. Shows side-by-side price comparison for budget planning."""
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
    """Find flights between two cities or airports. Browse flight options, departure times, airlines, routes. Use for 'find flights to Tokyo' or 'flights from New York to London'."""
    return f"Flights {origin}-{dest}: $300-500"

@tool
def search_flight_prices(origin: str, dest: str) -> str:
    """Compare flight costs, airfare prices, ticket rates between destinations. Shows how much flights cost. Use for 'how much do flights cost' or 'cheapest airfare'."""
    return f"Prices {origin}-{dest}: $300-500"

@tool
def get_flight_details(flight: str) -> str:
    """View specific flight information: aircraft model, flight duration, route details, airline carrier. Use for 'what plane' or 'how long is the flight'."""
    return f"Flight {flight}: Boeing 737, 3h"

@tool
def get_flight_status(flight: str) -> str:
    """Check if flight is on-time, delayed, cancelled. Shows gate number, departure/arrival status. Use for 'is flight AA123 on time' or 'flight delay status'."""
    return f"Flight {flight}: On time, Gate B4"

@tool
def check_flight_availability(flight: str) -> str:
    """Check remaining seats on a specific flight. See how many open seats available. Use for group travel planning or seat availability."""
    return f"Flight {flight}: 23 seats left"

@tool
def book_flight(flight: str, passenger: str) -> str:
    """Reserve flight ticket, complete airline booking, purchase airfare for passenger. Creates confirmed flight reservation."""
    return f"BOOKED {flight} for {passenger}"

# ============================================================================
# WEATHER TOOLS (Mock)
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """Check current weather conditions right now: temperature, sunny, rainy, cloudy. Shows today's weather only."""
    return f"{city}: 22°C, Sunny"

@tool
def get_weather_forecast(city: str) -> str:
    """View upcoming weather prediction for next few days. Shows tomorrow's weather, week ahead forecast, future conditions."""
    return f"{city} forecast: 22°C today, 20°C tomorrow"

@tool
def get_weather_alerts(city: str) -> str:
    """Check severe weather warnings, storm alerts, extreme weather notifications, travel advisories due to weather."""
    return f"{city}: No alerts"

# ============================================================================
# PAYMENT TOOLS (Mock)
# ============================================================================

@tool
def process_payment(amount: float) -> str:
    """Complete payment transaction, charge credit card, finalize purchase. Executes money transfer for booking."""
    return f"PAID ${amount}"

@tool
def check_payment(transaction_id: str) -> str:
    """Verify payment went through, check transaction status, confirm payment completed successfully."""
    return f"Transaction {transaction_id}: Complete"

@tool
def refund_payment(transaction_id: str) -> str:
    """Return money, process refund, reverse payment, cancel charge. Get money back for cancelled booking."""
    return f"REFUNDED {transaction_id}"

# ============================================================================
# TRAVEL UTILITY TOOLS
# ============================================================================

@tool
def get_currency_exchange(from_currency: str, to_currency: str, amount: float) -> str:
    """Convert money between currencies. Calculate exchange rates for international travel: USD to EUR, GBP to USD, etc. Use for 'convert 500 USD to EUR' or 'exchange rate' questions."""
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
    """Check visa requirements, passport rules, entry documentation needed for international travel. Shows if you need visa for Spain, France, Japan, etc. Use for 'do I need visa' questions."""
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
    """Generic broad search when category unknown. Only use if you cannot determine whether user wants hotels, flights, or other specific service. Last resort fallback."""
    return f"Results for: {query}"

@tool
def check(item: str) -> str:
    """Generic check function when type ambiguous. Only use if unclear whether checking availability, status, or something else. Last resort fallback."""
    return f"Checked: {item}"

@tool
def get_details(item: str) -> str:
    """Generic details function when unclear what information needed. Only use if user request is too vague for specific tools. Last resort fallback."""
    return f"Details: {item}"

@tool
def get_status(item: str) -> str:
    """Generic status check when unclear what status to check. Only use if cannot determine if user wants flight status, booking status, or other. Last resort fallback."""
    return f"Status: {item} OK"

@tool
def get_info(item: str) -> str:
    """Generic information retrieval when request too vague. Only use if user query doesn't match any specific tool category. Last resort fallback."""
    return f"Info: {item}"

@tool
def book(item: str, name: str) -> str:
    """Generic booking when type unclear. Only use if cannot determine whether booking hotel, flight, or other service. Last resort fallback."""
    return f"BOOKED {item} for {name}"

@tool
def cancel(item: str) -> str:
    """Generic cancellation when reservation type unclear. Only use if cannot determine what type of booking to cancel. Last resort fallback."""
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
