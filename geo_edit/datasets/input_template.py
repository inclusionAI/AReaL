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

VISWORLD_EVAL_INPUT_TEMPLATE = """\
You are a visual reasoning AI assistant capable of understanding complex visual scenarios and dynamics.

{prompt}

If you need to analyze the image in detail, you can use the available tools. Otherwise, provide your answer directly.

Please provide your final answer in <answer></answer> tags.
"""

VISWORLD_EVAL_NOTOOL_INPUT_TEMPLATE = """\
You are a visual reasoning AI assistant capable of understanding complex visual scenarios and dynamics.

{prompt}

Please analyze the image carefully and provide your final answer in <answer></answer> tags.
"""

BABYVISION_INPUT_TEMPLATE = """\
Please solve the visual reasoning problem with provided tools. After you confirm the final answer, put your answer in '<answer></answer>' tags.

{question}
{options_text}
"""

BABYVISION_NOTOOL_INPUT_TEMPLATE = """\
Please solve the visual reasoning problem step by step. After you confirm the final answer, put your answer in '<answer></answer>' tags.

{question}
{options_text}
"""

MAPEVAL_VISUAL_INPUT_TEMPLATE = """\
You are a map understanding assistant specialized in analyzing map images and answering location-based questions.

Look at the map image and answer the following question by selecting the correct option.

Question: {question}

Options:
{options_text}

If you need to analyze the image in detail, you can use the available tools.

Select the best option by choosing its number. If none of the options are correct or the question cannot be answered, respond with 0.

Please provide your final answer as a single number in <answer></answer> tags.
"""

MAPEVAL_VISUAL_NOTOOL_INPUT_TEMPLATE = """\
You are a map understanding assistant specialized in analyzing map images and answering location-based questions.

Look at the map image carefully and answer the following question by selecting the correct option.

Question: {question}

Options:
{options_text}

Select the best option by choosing its number. If none of the options are correct or the question cannot be answered, respond with 0.

Please provide your final answer as a single number in <answer></answer> tags.
"""

# Separated version: question only, no role/answer format (for separated reasoning mode)
MAPEVAL_VISUAL_SEPARATED_TEMPLATE = """\
Question: {question}

Options:
{options_text}

Select the best option by choosing its number. If none of the options are correct or the question cannot be answered, respond with 0.
"""

MAPEVAL_VISUAL_ANSWER_FORMAT = "Please provide your final answer as a single number in <answer></answer> tags."

CHARTQA_INPUT_TEMPLATE = """\
You are a chart understanding assistant specialized in analyzing charts and graphs.

Look at the chart image and answer the following question. Provide a precise and concise answer.

Question: {question}

If you need to analyze the chart in detail, you can use the available tools.

Please provide your final answer in <answer></answer> tags.
"""

CHARTQA_NOTOOL_INPUT_TEMPLATE = """\
You are a chart understanding assistant specialized in analyzing charts and graphs.

Look at the chart image carefully and answer the following question. Provide a precise and concise answer.

Question: {question}

Please analyze the chart and provide your final answer in <answer></answer> tags.
"""

# Separated version: question only, no role/answer format (for separated reasoning mode)
CHARTQA_SEPARATED_TEMPLATE = """\
Question: {question}

Provide a precise and concise answer based on the chart.
"""

CHARTQA_ANSWER_FORMAT = "Please provide your final answer in <answer></answer> tags."

CHARTQAPRO_INPUT_TEMPLATE = """\
You are a chart understanding assistant specialized in analyzing charts and graphs.

Look at the chart image and answer the following question. The question may be factoid, hypothetical, fact-checking, conversational, or multiple choice.

Question: {question}

If you need to analyze the chart in detail, you can use the available tools.

Please provide your final answer in <answer></answer> tags. For multiple choice questions, provide only the letter (e.g., A, B, C, D).
"""

CHARTQAPRO_NOTOOL_INPUT_TEMPLATE = """\
You are a chart understanding assistant specialized in analyzing charts and graphs.

Look at the chart image carefully and answer the following question. The question may be factoid, hypothetical, fact-checking, conversational, or multiple choice.

Question: {question}

Please analyze the chart and provide your final answer in <answer></answer> tags. For multiple choice questions, provide only the letter (e.g., A, B, C, D).
"""

# Separated version: question only, no role/answer format (for separated reasoning mode)
CHARTQAPRO_SEPARATED_TEMPLATE = """\
Question: {question}

Provide a precise and concise answer based on the chart. For multiple choice questions, provide only the letter.
"""

CHARTQAPRO_ANSWER_FORMAT = "Please provide your final answer in <answer></answer> tags."

FIGUREQA_INPUT_TEMPLATE = """\
You are a chart understanding assistant specialized in analyzing figures and charts.

Look at the figure image and answer the following Yes/No question about the data shown in the chart.

Question: {question}

If you need to analyze the image in detail, you can use the available tools.

Please provide your final answer (Yes or No) in <answer></answer> tags.
"""

FIGUREQA_NOTOOL_INPUT_TEMPLATE = """\
You are a chart understanding assistant specialized in analyzing figures and charts.

Look at the figure image carefully and answer the following Yes/No question about the data shown in the chart.

Question: {question}

Please analyze the chart and provide your final answer (Yes or No) in <answer></answer> tags.
"""

REASONMAP_INPUT_TEMPLATE = """\
Look at the subway map image and answer the following question.

{question}

If you need to analyze the image in detail, you can use the available tools.

Please provide your final answer in <answer></answer> tags. For multiple choice questions, provide only the letter (A, B, C, or D). For yes/no questions, answer Yes or No. For counting questions, provide the number.
"""

REASONMAP_NOTOOL_INPUT_TEMPLATE = """\
Look at the subway map image carefully and answer the following question.

{question}

Please provide your final answer in <answer></answer> tags. For multiple choice questions, provide only the letter (A, B, C, or D). For yes/no questions, answer Yes or No. For counting questions, provide the number.
"""
