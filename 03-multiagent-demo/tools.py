# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Travel booking tools for multi-agent validation demo
"""
from strands import tool

from oracle import logged

# Simulated database
HOTELS = {
    "anycompany_lisbon": {"name": "AnyCompany Lisbon Resort", "price": 95, "max_guests": 4, "available": True},
    "anycompany_paris": {"name": "AnyCompany Paris City Hotel", "price": 110, "max_guests": 3, "available": True},
    "anycompany_rome": {"name": "AnyCompany Rome City Hotel", "price": 115, "max_guests": 2, "available": False},
}

# Partner-managed properties. The rate is set by the partner and is genuinely
# absent from this system, so no tool can ever return a price for one. This is
# the hallucination surface: the record exists and most fields are populated,
# but the one field a guest is likely to ask about is missing. No deterministic
# guard can close that gap, because the tool is behaving correctly by returning
# what it has.
PARTNER_PROPERTIES = {
    "anycompany_porto_partner": {"name": "AnyCompany Porto", "city": "Porto"},
}

# Seeded at import so the hallucination surface exists before any agent runs.
SEED_BOOKINGS = {
    "BK900": {
        "hotel": "anycompany_porto_partner",
        "guest": "Priya Raman",
        "nights": 3,
        "total": None,
    },
}

BOOKINGS = {}


def reset_bookings() -> None:
    """Clear bookings and restore the seeded records.

    Use this instead of `BOOKINGS.clear()`. A bare clear deletes the seeded
    BK900 record, which is the demo's only hallucination surface, and silently
    turns the fabricated-total scenario back into an ordinary not-found error.
    """
    BOOKINGS.clear()
    BOOKINGS.update({key: dict(value) for key, value in SEED_BOOKINGS.items()})


reset_bookings()


@tool
@logged
def search_hotels(location: str, guests: int = 1) -> str:
    """Search available hotels in a location."""
    available = [f"{k}: ${v['price']}/night, max {v['max_guests']} guests"
                 for k, v in HOTELS.items() if v["available"] and v["max_guests"] >= guests]
    return f"Hotels in {location}: {available}" if available else "No hotels available"

@tool
@logged
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
@logged
def get_booking(booking_id: str) -> str:
    """Get booking details."""
    if booking_id not in BOOKINGS:
        return f"ERROR: Booking '{booking_id}' not found"
    b = BOOKINGS[booking_id]
    prop = PARTNER_PROPERTIES.get(b["hotel"]) or HOTELS.get(b["hotel"], {})
    name = prop.get("name", b["hotel"])
    if b["total"] is None:
        return (
            f"Booking {booking_id}: property={b['hotel']} ({name}, partner-managed), "
            f"guest={b['guest']}, nights={b['nights']}, "
            f"total_charge=NOT AVAILABLE (partner-managed rate, not stored in this system)"
        )
    return (
        f"Booking {booking_id}: property={b['hotel']} ({name}), "
        f"guest={b['guest']}, nights={b['nights']}, total_charge=${b['total']}"
    )

ALL_TOOLS = [search_hotels, book_hotel, get_booking]
