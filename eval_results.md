# Evaluation Results

## Retrieval mode comparison (gold section in top-3?)

| # | Question | hybrid | semantic | bm25 |
|---|----------|--------|----------|------|
| 1 | Which shortest-path algorithm handles graphs with negative edge weights, and what special structure can it detect? | hit | hit | hit |
| 2 | In the greedy interval scheduling pattern, what should you sort the intervals by before selecting them? | hit | hit | hit |
| 3 | What are the four key components of the universal backtracking template? | hit | miss | hit |
| 4 | In the lowest common ancestor function, what is returned when both the left and right recursive calls return a non-null node? | hit | hit | hit |
| 5 | What is the time complexity of Dijkstra's algorithm when implemented with a min-heap? | hit | miss | hit |

**Gold-section hit rate:** hybrid=5/5, semantic=3/5, bm25=5/5

## End-to-end results (hybrid mode)

### Q1. Which shortest-path algorithm handles graphs with negative edge weights, and what special structure can it detect?
- **Expected:** Bellman-Ford; it is used when edges may be negative and it detects negative cycles.
- **Retrieved chunks:**
    - Graphs_Study_Guide.pdf :: 3. Shortest Path Algorithms > 3.1 Dijkstra’s Algorithm
    - Graphs_Study_Guide.pdf :: 3. Shortest Path Algorithms > 3.2 Bellman-Ford
    - Graphs_Study_Guide.pdf :: 3. Shortest Path Algorithms > 3.4 BFS for Unweighted Graphs
- **Retrieval quality:** Relevant (gold section retrieved)
- **System response:** The Bellman-Ford algorithm handles graphs with negative edge weights [2], and it can detect negative cycles [2].

### Q2. In the greedy interval scheduling pattern, what should you sort the intervals by before selecting them?
- **Expected:** Sort by end time (finish earliest) and choose an interval if it doesn't overlap the last chosen one.
- **Retrieved chunks:**
    - Greedy_Algorithms_Study_Guide.pdf :: 3. Core Greedy Patterns > 3.1 Interval Scheduling
    - Greedy_Algorithms_Study_Guide.pdf :: 3. Core Greedy Patterns > 3.2 Interval Merging
    - Greedy_Algorithms_Study_Guide.pdf :: 3. Core Greedy Patterns > 3.3 Minimum Number of Arrows / Points
- **Retrieval quality:** Relevant (gold section retrieved)
- **System response:** In the greedy interval scheduling pattern, you should sort the intervals by their end time [1].

### Q3. What are the four key components of the universal backtracking template?
- **Expected:** State, Choices, Constraints, and Goal.
- **Retrieved chunks:**
    - Backtracking_Study_Guide.pdf :: 3. Backtracking Template
    - Backtracking_Study_Guide.pdf :: 9. Key Takeaways
    - Greedy_Algorithms_Study_Guide.pdf :: 8. Key Takeaways
- **Retrieval quality:** Relevant (gold section retrieved)
- **System response:** The four key components of the universal backtracking template are: 
1. State: partial solution 
2. Choices: what you can try next 
3. Constraints: rules that invalidate choices 
4. Goal: when a solution is complete [1].

### Q4. In the lowest common ancestor function, what is returned when both the left and right recursive calls return a non-null node?
- **Expected:** The current node (root) is returned as the LCA.
- **Retrieved chunks:**
    - Trees_Study_Guide.pdf :: 3. Essential Tree Techniques > 3.3 Lowest Common Ancestor (LCA)
    - Trees_Study_Guide.pdf :: 3. Essential Tree Techniques > 3.1 Recursion Templates
    - Trees_Study_Guide.pdf :: 3. Essential Tree Techniques > 3.4 Serialization / Deserialization
- **Retrieval quality:** Relevant (gold section retrieved)
- **System response:** In the lowest common ancestor function, when both the left and right recursive calls return a non-null node, the function returns `root` [1]. This is indicated by the line `if left and right: return root` in the `lca` function.

### Q5. What is the time complexity of Dijkstra's algorithm when implemented with a min-heap?
- **Expected:** Not stated in the documents -> the system should say it doesn't have enough information (true answer is O((V+E) log V)).
- **Retrieved chunks:**
    - Graphs_Study_Guide.pdf :: 3. Shortest Path Algorithms > 3.1 Dijkstra’s Algorithm
    - Graphs_Study_Guide.pdf :: 7. Practice Roadmap
    - Backtracking_Study_Guide.pdf :: 3. Backtracking Template
- **Retrieval quality:** Relevant (gold section retrieved)
- **System response:** I can only help answer questions regarding these documents or help with the documents. The time complexity of Dijkstra's algorithm is not mentioned in the provided CONTEXT passages [1], [2], or [3].
