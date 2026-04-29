# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Travel booking tools for multi-agent validation demo
"""
from strands import tool

# Simulated database
HOTELS = {
    "anycompany_lisbon": {"name": "AnyCompany Lisbon Resort", "price": 95, "max_guests": 4, "available": True},
    "anycompany_paris": {"name": "AnyCompany Paris City Hotel", "price": 110, "max_guests": 3, "available": True},
    "anycompany_rome": {"name": "AnyCompany Rome City Hotel", "price": 115, "max_guests": 2, "available": False},
}

BOOKINGS = {}

@tool
def search_hotels(location: str, guests: int = 1) -> str:
    """Search available hotels in a location."""
    available = [f"{k}: ${v['price']}/night, max {v['max_guests']} guests" 
                 for k, v in HOTELS.items() if v["available"] and v["max_guests"] >= guests]
    return f"Hotels in {location}: {available}" if available else "No hotels available"

@tool
def book_hotel(hotel_id: str, guest_name: str, nights: int = 1) -> str:
    """Book a hotel room."""
    if hotel_id not in HOTELS:
        return f"ERROR: Hotel '{hotel_id}' not found"
    if not HOTELS[hotel_id]["available"]:
        return f"ERROR: {hotel_id} is not available"
    
    total = HOTELS[hotel_id]["price"] * nights
    booking_id = f"BK{len(BOOKINGS)+1:03d}"
    BOOKINGS[booking_id] = {"hotel": hotel_id, "guest": guest_name, "nights": nights, "total": total}
    return f"SUCCESS: Booking {booking_id} confirmed. {hotel_id} for {nights} nights. Total: ${total}"

@tool
def get_booking(booking_id: str) -> str:
    """Get booking details."""
    if booking_id not in BOOKINGS:
        return f"ERROR: Booking '{booking_id}' not found"
    b = BOOKINGS[booking_id]
    return f"Booking {booking_id}: {b['hotel']} for {b['guest']}, {b['nights']} nights, ${b['total']}"

ALL_TOOLS = [search_hotels, book_hotel, get_booking]
