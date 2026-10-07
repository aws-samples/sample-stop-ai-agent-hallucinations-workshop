# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Try native steering yourself — book a party too big for one room.

Same pieces the notebook explains:
  * BookingStore.book_hotel  — writes reservations to a local JSON store.
  * HotelCheck.hotel_exists  — read-only Neo4j: confirm the hotel exists.
  * MaxGuestsSteering        — native Strands SteeringHandler. If you ask to
    book more than 10 guests, it does NOT fail — it guides the agent to split
    the party into rooms (ceil(guests/10)) and book each one.

Try: "Book 15 guests at Cliffside Resort for Jordan Lee" → two rooms (10 + 5).

Prerequisites: a running Neo4j with the hotel graph (see demo 01's build_graph.py).

Run:
    AWS_PROFILE=<profile> AWS_REGION=us-east-1 python chat.py
"""
import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from dotenv import load_dotenv
from strands import Agent

from booking_store import BookingStore
from steering_tools import MaxGuestsSteering, HotelCheck

load_dotenv()


def main():
    store = BookingStore()
    hotels = HotelCheck()

    # Strands uses Amazon Bedrock (Claude Sonnet 4) by default — no model config.
    agent = Agent(
        system_prompt=(
            "You are a travel agent that helps people with hotels. Book the whole party "
            "together; if you receive guidance about room limits, follow it. Keep answers concise."
        ),
        tools=[hotels.hotel_exists, store.book_hotel],
        plugins=[MaxGuestsSteering()],
    )

    print("Hotel booking agent ready. Steering splits oversize parties into rooms.")
    print('Try: "Book 15 guests at Cliffside Resort for Jordan Lee" — Ctrl-C to exit.\n')
    try:
        while True:
            q = input("you> ").strip()
            if not q:
                continue
            agent(q)
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
    finally:
        hotels.close()


if __name__ == "__main__":
    main()
