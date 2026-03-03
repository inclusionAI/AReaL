MATHVISION_INPUT_TEMPLATE = """
Please solve the problem with provided tools. After you confirm the final answer, put your answer in one '<answer>\\boxed{{}}</answer>'. If it is a multiple choice question, only one letter is allowed in the '<answer>\\boxed{{}}</answer>'.\n{question}\n{options}
"""

MATHVISION_NOTOOL_INPUT_TEMPLATE = """
Please solve the problem step by step. After you confirm the final answer, put your answer in one '<answer>\\boxed{{}}</answer>'. If it is a multiple choice question, only one letter is allowed in the '<answer>\\boxed{{}}</answer>'.\n{question}\n{options}
"""


CARTOMAPQA_INPUT_TEMPLATE = f"""You are a highly skilled cartography assistant with expertise in interpreting cartographic maps. \
You can identify and analyze visual elements, such as paths, labels, symbols, and features, and extract structured information such as object counts, names, types, and measurements. \
You respond clearly, accurately, and concisely, following the specific output format requested in the user’s prompt."""

CARTOMAPQA_SRN_INPUT_TEMPLATE = """You are provided with a cartographic map sourced from OpenStreetMap. Two colored map markers are shown: the blue marker indicates the starting location, and the red one marks the destination.\n\
Assume the user is located at the blue marker, seated in a vehicle, and initially facing toward the top of the map. Your task is to determine the shortest drivable route from the blue marker to the red marker, and describe all required driving actions to reach the destination.\n\
Output your answer using the following strict format: <answer>Answer: [blue, <action_1>, road_1, <action_2>, road_2, ..., <action_N>, road_N, red]</answer>\n\
Each <action> must be one of the following:\n\
- "make a U-turn and continue straight"\n\
- "continue straight"\n\
- "turn left"\n\
- "turn right"\n\
Use the exact road names as shown on the map.\n\
For example: If the route involves a u-turn at the start, followed by driving on "Main Street", a left turn onto "2nd Avenue", and a right turn onto "Elm Street" to reach the destination, the response should be:\
<answer>Answer: [blue, make a U-turn and continue straight, Main Street, turn left, 2nd Avenue, turn right, Elm Street, red]</answer>\n\
Strictly follow this format. Do not include any explanation, justification, or additional text—only return the final answer list with the format as shown above."""

CARTOMAPQA_STMF_PRESENCE_TEMPLATE = """You are provided with a cartographic map sourced from OpenStreetMap.\n\
Your task is to determine whether the map contains any visual elements that indicate the presence of a point of interest (POI) of the following type: {mf_type}.\n\
Respond with "Yes" or "No" on the first line.\n\
On the next line, provide a concise explanation describing the evidence or absence of evidence that supports your answer.\n\
Base your reasoning on map features such as labels, icons, shapes, or patterns commonly associated with this type of POI."""

CARTOMAPQA_STMF_COUNTING_TEMPLATE = """You are provided with a cartographic map sourced from OpenStreetMap.\n\
Your task is to identify and count all visual elements that indicate the presence of a point of interest (POI) of the following type: {mf_type}.\n\
On the first line of your reponse, provide the total number of such elements found.\n\
On the next line, explain your reasoning by describing the evidence used to support your count.\n\
Example of a response: \
5\n
[Explanation]"""

CARTOMAPQA_STMF_NAME_LISTING_TEMPLATE = """You are provided with a cartographic map sourced from OpenStreetMap. \n\
Your task is to identify all visual elements that indicate the presence of a point of interest (POI) of the following type: {mf_type}, and list those that have an associated name.\n\
Respond with one name per line, using the exact name as it appears on the map.\n\
If a visual element does not have a name, represent it with an empty string ("").\n\
If all relevant elements lack associated names, leave the output completely blank.\n\
Example format (when all names are present):\n\
Name_1\n\
Name_2\n\
...\n\
Example format (when some names are present):\n\
Name_1\n\
""\n\
Name_3\n\
...\n\
Example format (when no names are available):\n\
[leave the output blank]\n\
Strictly follow this format. Do not include any explanation, description, code, or additional text."""
