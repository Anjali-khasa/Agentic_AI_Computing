import pandas as pd
import os

# -------------------------------------------------
# Step 1: Loading the restaurant dataset
# -------------------------------------------------
current_folder = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_folder, "restaurant_data.csv")
data = pd.read_csv(csv_path)

print("Restaurant Dataset Loaded Successfully")
print("Number of restaurants in dataset:", len(data))
print("-" * 50)

# -------------------------------------------------
# Step 2: Storing the user query
# -------------------------------------------------
user_query = (
    "Find a Turkish restaurant in Downtown Baltimore, MD for two people "
    "to have dinner under $65 on Thursday night at 7:30 PM with a table "
    "for two near a window with a view of the garden or the street."
)

print("User Query:")
print(user_query)
print("-" * 50)

# -------------------------------------------------
# Step 3: Extracting the conditions from the query
# -------------------------------------------------
constraints = {
    "cuisine": None,
    "location": None,
    "budget": None,
    "day": None,
    "time": None,
    "party_size": None,
    "table_for_two": None,
    "window_view": None,
    "view_type": []
}

query_lower = user_query.lower()

if "turkish" in query_lower:
    constraints["cuisine"] = "Turkish"

if "downtown baltimore" in query_lower:
    constraints["location"] = "Downtown Baltimore"

if "under $65" in query_lower or "under 65" in query_lower:
    constraints["budget"] = 65

if "thursday" in query_lower:
    constraints["day"] = "Thursday"

if "7:30 pm" in query_lower or "7.30 pm" in query_lower:
    constraints["time"] = "19:30"

if "for two" in query_lower or "two people" in query_lower:
    constraints["party_size"] = 2
    constraints["table_for_two"] = "Yes"

if "near a window" in query_lower or "window" in query_lower:
    constraints["window_view"] = "Yes"

if "garden" in query_lower:
    constraints["view_type"].append("Garden")

if "street" in query_lower:
    constraints["view_type"].append("Street")

print("Extracted Constraints:")
for key, value in constraints.items():
    print(f"{key}: {value}")

print("-" * 50)
print("AGENT REASONING LOGS")
print("-" * 50)

# -------------------------------------------------
# Step 4: Filtering restaurants
# -------------------------------------------------
filtered_data = data.copy()
print(f"[Log 1] Initial restaurants loaded: {len(filtered_data)}")

if constraints["cuisine"] is not None:
    filtered_data = filtered_data[
        filtered_data["cuisine"].str.lower() == constraints["cuisine"].lower()
    ]
    print(f"[Log 2] After cuisine filter ({constraints['cuisine']}): {len(filtered_data)} restaurants remain")

if constraints["location"] is not None:
    filtered_data = filtered_data[
        filtered_data["location"].str.lower() == constraints["location"].lower()
    ]
    print(f"[Log 3] After location filter ({constraints['location']}): {len(filtered_data)} restaurants remain")

if constraints["budget"] is not None:
    filtered_data = filtered_data[
        filtered_data["avg_cost_for_two"] <= constraints["budget"]
    ]
    print(f"[Log 4] After budget filter (under ${constraints['budget']}): {len(filtered_data)} restaurants remain")

if constraints["day"] is not None:
    filtered_data = filtered_data[
        filtered_data["open_day"].str.lower() == constraints["day"].lower()
    ]
    print(f"[Log 5] After day filter ({constraints['day']}): {len(filtered_data)} restaurants remain")

if constraints["time"] is not None:
    filtered_data = filtered_data[
        (filtered_data["open_time"] <= constraints["time"]) &
        (filtered_data["close_time"] >= constraints["time"])
    ]
    print(f"[Log 6] After time filter ({constraints['time']}): {len(filtered_data)} restaurants remain")

if constraints["table_for_two"] is not None:
    filtered_data = filtered_data[
        filtered_data["table_for_two_available"].str.lower() == constraints["table_for_two"].lower()
    ]
    print(f"[Log 7] After table-for-two filter ({constraints['table_for_two']}): {len(filtered_data)} restaurants remain")

if constraints["window_view"] is not None:
    filtered_data = filtered_data[
        filtered_data["window_view_option"].str.lower() == constraints["window_view"].lower()
    ]
    print(f"[Log 8] After window-view filter ({constraints['window_view']}): {len(filtered_data)} restaurants remain")

if constraints["view_type"]:
    filtered_data = filtered_data[
        filtered_data["view_type"].isin(constraints["view_type"])
    ]
    print(f"[Log 9] After view-type filter ({constraints['view_type']}): {len(filtered_data)} restaurants remain")

print("-" * 50)
print("FILTERED RESULTS")
print("-" * 50)

if filtered_data.empty:
    print("No restaurants matched all constraints.")
else:
    print(filtered_data[[
        "name",
        "cuisine",
        "location",
        "avg_cost_for_two",
        "open_time",
        "close_time",
        "view_type",
        "rating"
    ]].to_string(index=False))

# -------------------------------------------------
# Step 5: Ranking the filtered restaurants
# -------------------------------------------------
print("-" * 50)
print("RANKING RESULTS")
print("-" * 50)

if not filtered_data.empty:
    ranked_data = filtered_data.copy()

    # Simple ranking:
    # higher rating is better
    # lower cost under the budget is also better
    ranked_data["score"] = (ranked_data["rating"] * 20) + (
        constraints["budget"] - ranked_data["avg_cost_for_two"]
    )

    ranked_data = ranked_data.sort_values(
        by=["score", "rating", "avg_cost_for_two"],
        ascending=[False, False, True]
    )

    print(ranked_data[[
        "name",
        "cuisine",
        "location",
        "avg_cost_for_two",
        "view_type",
        "rating",
        "score"
    ]].to_string(index=False))

    print("-" * 50)
    print("TOP 5 RANKED RESTAURANTS")
    print("-" * 50)
    print(ranked_data[[
        "name",
        "avg_cost_for_two",
        "view_type",
        "rating",
        "score"
    ]].head(5).to_string(index=False))
else:
    print("No restaurants available to rank.")

# -------------------------------------------------
# Step 6: Explanation of the top results
# -------------------------------------------------
print("-" * 50)
print("EXPLANATION OF TOP RANKED RESULTS")
print("-" * 50)

if not filtered_data.empty:
    top_results = ranked_data.head(5)

    for i, (_, row) in enumerate(top_results.iterrows(), start=1):
        print(
            f"Rank {i}: {row['name']} matches the query because it is a "
            f"{row['cuisine']} restaurant in {row['location']}, has an estimated "
            f"dinner cost for two of ${row['avg_cost_for_two']}, is open during "
            f"the requested Thursday dinner time, offers a table for two, supports "
            f"window seating, and has a {row['view_type'].lower()} view. "
            f"It has a rating of {row['rating']}, which contributed to its final score of {row['score']:.1f}."
        )
        print()
else:
    print("No explanations available because no restaurants matched the query.")

# -------------------------------------------------
# Step 7: The final top 5 recommendations
# -------------------------------------------------
print("-" * 50)
print("FINAL TOP 5 RECOMMENDATIONS")
print("-" * 50)

if not filtered_data.empty:
    final_output = ranked_data[[
        "name",
        "cuisine",
        "location",
        "avg_cost_for_two",
        "open_time",
        "close_time",
        "view_type",
        "rating",
        "score"
    ]].head(5)

    print(final_output.to_string(index=False))
else:
    print("No final recommendations available.")