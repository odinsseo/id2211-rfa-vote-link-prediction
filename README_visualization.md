# Visualizing the Wikipedia Vote Graph

The vote graph can be visualized using Gephi, a free and open-source network visualization tool.

## Installation

1. Download and install Gephi from https://gephi.org/

## Loading the Graph

1. Open Gephi
2. Go to File > Open and select `data/vote_graph.graphml`
3. In the import settings:
   - Select "Directed" for graph type
   - Keep all attributes selected
   - Click "OK"

## Basic Visualization

1. Go to the "Overview" tab
2. In the Statistics panel (right side):
   - Run "Average Degree"
   - Run "Modularity" for community detection
   
3. In the Appearance panel (left side):
   - Color nodes by Modularity Class
   - Size nodes by Degree
   
4. In the Layout panel (left side):
   - Run "ForceAtlas 2" layout
   - Check "Prevent Overlap"
   - Adjust "Scaling" to spread out the network
   - Let it run until the network stabilizes
   
5. For edge colors:
   - Select Edges tab in Appearance
   - Color by "sign" attribute
   - Use red for negative (-1) and green for positive (+1)

## Temporal Analysis

1. In the Timeline panel (bottom):
   - Enable it by clicking the clock icon
   - Use the timestamp attribute
   - You can play through time to see how the network evolved

## Advanced Features

1. Filter panel (right side):
   - Filter by time period
   - Filter by edge sign
   - Filter by degree to focus on active users

2. Statistics:
   - Calculate centrality metrics
   - Identify important nodes
   - Analyze community structure

## Export Visualization

1. Go to the "Preview" tab
2. Adjust settings for optimal appearance
3. Export as:
   - PDF for vector graphics
   - PNG for images
   - SVG for web use

## Tips

- The graph is large (7,117 nodes), so:
  - Use filters to focus on specific time periods
  - Consider filtering to show only high-degree nodes
  - Adjust layout parameters for better visualization
  
- Edge colors:
  - Green edges = support votes (+1)
  - Red edges = oppose votes (-1)
  
- Node sizes can represent:
  - Number of votes received
  - Number of votes cast
  - Overall influence in the network
