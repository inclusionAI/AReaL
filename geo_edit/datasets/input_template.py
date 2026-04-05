
MATHVISION_NOTOOL_INPUT_TEMPLATE = """
Please solve the problem step by step. After you confirm the final answer, put your answer in one '<answer>\\boxed{{}}</answer>'. If it is a multiple choice question, only one letter is allowed in the '<answer>\\boxed{{}}</answer>'.\n{question}\n{options}
"""


CARTOMAPQA_INPUT_TEMPLATE = """You are a highly skilled cartography assistant with expertise in interpreting cartographic maps. \
You can identify and analyze visual elements, such as paths, labels, symbols, and features, and extract structured information such as object counts, names, types, and measurements. \
You respond clearly, accurately, and concisely, following the specific output format requested in the user’s prompt."""

# Unified template: role description + HF dataset `question` field (which already
# contains full task instructions, format examples, and answer structure).
# The <answer> tag instruction comes from the pipeline system prompt, not here.
CARTOMAPQA_UNIFIED_TEMPLATE = CARTOMAPQA_INPUT_TEMPLATE + "\n\n{question}"

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

{question}

Options:
{options_text}

If you need to analyze the image in detail, you can use the available tools.

Select the best option by choosing its number. If none of the options are correct or the question cannot be answered, respond with 0.

Please provide your final answer as a single number in <answer></answer> tags.
"""

MAPEVAL_VISUAL_NOTOOL_INPUT_TEMPLATE = """\
You are a map understanding assistant specialized in analyzing map images and answering location-based questions.

Look at the map image carefully and answer the following question by selecting the correct option.

{question}

Options:
{options_text}

Select the best option by choosing its number. If none of the options are correct or the question cannot be answered, respond with 0.

Please provide your final answer as a single number in <answer></answer> tags.
"""

# Separated version: question only, no role/answer format (for separated reasoning mode)
MAPEVAL_VISUAL_SEPARATED_TEMPLATE = """\
{question}

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

# Separated version: question only, no role/answer format (for separated reasoning mode)
REASONMAP_SEPARATED_TEMPLATE = """\
{question}

For multiple choice questions, provide only the letter (A, B, C, or D). For yes/no questions, answer Yes or No. For counting questions, provide the number.
"""
