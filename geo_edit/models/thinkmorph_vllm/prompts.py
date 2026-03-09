# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# ThinkMorph-specific prompt templates that encourage visual thinking

"""
Prompt templates optimized for ThinkMorph's visual thinking capabilities.
"""

# ThinkMorph-specific template for Spatial Route Navigation (SRN)
# Encourages the model to draw the route on the map
CARTOMAPQA_SRN_VISUAL_TEMPLATE = """You are provided with a cartographic map sourced from OpenStreetMap. Two colored map markers are shown: the blue marker indicates the starting location, and the red one marks the destination.

Your task is to find the shortest drivable route from the blue marker to the red marker.

To solve this problem, please use visual thinking:
1. Draw the route on the map - use a colored line (green or yellow) to trace the path from blue to red marker
2. You can generate multiple images if needed to refine your route or show different segments
3. Mark key turn points along your route
4. Based on your visualization, provide the driving directions

Use <image_start> </image_end> to generate your route visualization.

After your visual analysis, output the final answer:
<answer>Answer: [blue, <action_1>, road_1, <action_2>, road_2, ..., <action_N>, road_N, red]</answer>

Actions allowed:
- "make a U-turn and continue straight"
- "continue straight"
- "turn left"
- "turn right"

The user starts at the blue marker, facing toward the top of the map.
Use exact road names from the map."""
