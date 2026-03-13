Assignment 3: Restaurant Recommendation Agent

1. Project Overview

In this project, I built a simple restaurant recommendation agent that takes a user query in natural language and finds matching restaurants based on the given conditions. The main idea was to make the program behave like a small intelligent agent that can understand the request, search through available options, remove the restaurants that do not match, rank the remaining choices, and explain the final result.

The user scenario for this assignment is:
“Find a Turkish restaurant in Downtown Baltimore, MD for two people to have dinner under $65 on Thursday night at 7:30 PM with a table for two near a window with a view of the garden or the street.”

The goal of the agent is to return the best restaurant options that satisfy this request as closely as possible.

--------------------------------------------------

2. Goal of the Agent

The goal of this agent is to:
- understand the user’s restaurant request
- extract important conditions from the sentence
- search restaurant data
- apply all required constraints
- rank the valid restaurants
- explain why the top recommendations match the query

In simple words, the agent starts with a broad set of restaurant options and narrows them down step by step until it finds the best possible matches.

--------------------------------------------------

3. States of the Agent

I used the idea of states to show how the agent moves through the problem from start to finish.

State 1: Query Received
The agent first receives the user’s request as a sentence.

State 2: Constraints Extracted
The agent reads the query and identifies important details such as cuisine, location, budget, time, and seating preference.

State 3: Restaurant Data Retrieved
The agent loads restaurant data from the CSV file.

State 4: Restaurants Filtered
The agent removes the restaurants that do not satisfy the required conditions.

State 5: Restaurants Ranked
The remaining restaurants are scored and sorted from best match to lower match.

State 6: Final Recommendation Generated
The agent prints the top results and explains why they match the user’s request.

--------------------------------------------------

4. Actions Performed by the Agent

The agent performs the following actions in sequence:

1. Read the user query
2. Extract the important conditions from the query
3. Load restaurant data from the CSV file
4. Filter by cuisine
5. Filter by location
6. Filter by budget
7. Filter by day
8. Filter by time
9. Filter by table availability
10. Filter by window seating preference
11. Filter by preferred view type
12. Rank the remaining restaurants
13. Print the top 5 recommendations
14. Explain why the top restaurants match the query

--------------------------------------------------

5. How the Agent Uses a Pipeline Approach

I implemented the solution as a simple pipeline, where each step passes its output to the next step.

Step 1: User Query Input
The program starts by storing the user’s restaurant request.

Step 2: Query Understanding
I used a simple keyword-based method to extract important conditions from the sentence. For example, the program checks whether words like “Turkish,” “Downtown Baltimore,” “Thursday,” “under $65,” “7:30 PM,” “window,” “garden,” and “street” are present.

Step 3: Data Retrieval
The agent reads a custom CSV dataset that contains restaurant information such as cuisine, location, average cost for two, open and close times, view type, and rating.

Step 4: Filtering
The agent filters the data step by step using the extracted conditions. At each stage, restaurants that do not match are removed.

Step 5: Ranking
After filtering, the agent ranks the remaining restaurants using a score based on rating and price.

Step 6: Final Output
The agent prints the ranked results, explains the top recommendations, and shows the final top 5 restaurant choices.

This pipeline makes the program easy to understand because each part has a clear purpose.

--------------------------------------------------

6. Constraints Extracted from the Query

From the given user query, the agent extracts the following conditions:

- Cuisine = Turkish
- Location = Downtown Baltimore
- Budget = under $65
- Day = Thursday
- Time = 7:30 PM
- Party size = 2
- Table for two = Yes
- Window seating preference = Yes
- View type = Garden or Street

These extracted conditions are then used during filtering.

--------------------------------------------------

7. Filtering and Reasoning with Constraints

The main reasoning part of this project happens during filtering.

The agent starts with all restaurants in the dataset and then removes restaurants that do not match the user’s conditions. For example:

- restaurants that are not Turkish are removed
- restaurants outside Downtown Baltimore are removed
- restaurants above the budget are removed
- restaurants not open on Thursday at 7:30 PM are removed
- restaurants that do not offer a table for two are removed
- restaurants without window seating are removed
- restaurants without a garden or street view are removed

I also added reasoning logs to show how many restaurants remain after each step. This makes it easy to see how the agent is making decisions.

--------------------------------------------------

8. Ranking Logic

Once the filtering is complete, the agent ranks the remaining restaurants.

I used a simple scoring formula that gives importance to:
- higher restaurant rating
- lower cost within the budget

This means restaurants with better ratings and more affordable pricing tend to appear higher in the final results.

This ranking is simple, but it works well for this assignment because it gives a clear way to compare the remaining choices.

--------------------------------------------------

9. Final Output

The program prints:
- the extracted constraints
- reasoning logs after each filter
- the filtered restaurant list
- the ranked restaurant list
- explanations for the top ranked restaurants
- the final top 5 recommendations in a clean table

This makes the result easy to read and also shows how the agent reached its decision.


