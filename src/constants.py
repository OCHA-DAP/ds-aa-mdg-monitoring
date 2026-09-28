import os

ISO3 = "MDG"

RAIN_THRESH = 300

NAUTICAL_MILE_TO_KM = 1.852

LISTMONK_LIST_ID = 109
# Test-mode list. Default 103 ("Pauline"); override with LISTMONK_TEST_LIST_ID
# to route a test send elsewhere (e.g. 5 = "Tristan only").
LISTMONK_LIST_ID_TEST = int(os.getenv("LISTMONK_TEST_LIST_ID", "103"))
