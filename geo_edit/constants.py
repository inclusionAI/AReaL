API_KEY = ""

SYSTEM_PROMPT = '''
You are an advanced AI agent capable of complex
reasoning and tool usage. You must strictly adhere
to the following protocol for every interaction:
1. ALWAYS call the appropriate tool first;
2. NEVER provide answers without tool results;
3. Call appropriate tools based on the task;
4. If you call multiple tools in one action, only the final result will be returned;
5. Reasoning Before Action: before selecting a tool,
you must analyze the user's request and determine
the necessary steps. Output your internal monologue
and logic inside <think> and </think> tags.
6. Reasoning After Action: Once you receive the
output from a tool, you must analyze the results to
determine if further actions are needed or if the task
is complete. Output this analysis inside <think> and
</think> tags;
7. Final Output: When you have formulated your
conclusion, you must wrap your final answer in
<answer> and </answer> tags.
'''

MATHVISION_INPUT_TEMPLATE = '''
Please solve the problem with provided tools. After you confirm the final answer, put your answer in one '<answer>\\boxed{{}}</answer>'. If it is a multiple choice question, only one letter is allowed in the '<answer>\\boxed{{}}</answer>'.\n{question}\n{options}
'''

NOTOOL_INPUT_TEMPLATE = '''
Please solve the problem step by step. After you confirm the final answer, put your answer in one '<answer>\\boxed{{}}</answer>'. If it is a multiple choice question, only one letter is allowed in the '<answer>\\boxed{{}}</answer>'.\n{question}\n{options}
'''

TOOL_EXECUTION_SUCCESS_PROMPT="All your tool calls were executed successfully. Now you can check the tool execution results and decide your next action."

TOOL_EXECUTION_FAILURE_PROMPT="Some of your tool calls failed. Please carefully check the tool execution results, identify the failed tool calls, and decide your next action accordingly."

EVAL_SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs "
    "for question-answer pairs.\n"
    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully.\n"
    "------\n"
    "##INSTRUCTIONS:\n"
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)

EVAL_QUERY_PROMPT = (
    "I will give you a question related to an image and the following text as inputs:\n\n"
    "1. **Question Related to the Image**: {question}\n"
    "2. **Ground Truth Answer**: {ground_truth}\n"
    "3. **Model Predicted Answer**: {prediction}\n\n"
    "Your task is to evaluate the model's predicted answer against the ground truth answer, "
    "based on the context provided by the question related to the image. Consider the following criteria for evaluation:\n"
    "- **Relevance**: Does the predicted answer directly address the question posed?\n"
    "- **Accuracy**:\n"
    "(1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information "
    "given in the ground truth without introducing factual inaccuracies.\n"
    "(2) If the ground truth answer is a definitive answer, strictly compare the model's prediction to the actual answer. "
    "Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model's "
    "prediction should be deemed correct.\n\n"
    "**Output Format**:\n"
    'Your response should include an integer score indicating the correctness of the prediction: 1 for correct and 0 for incorrect.\n'
    'The format should be "Score: 0 or 1"'
)

MAX_TOOL_CALLS = 8

SUDOKU_TOOL_CALL_INPUT_TEMPLATE = '''
You are a professional Sudoku puzzle solver. Please solve the following Sudoku variant.

## Format Explanation ##
Coordinates:
- We will use r{{x}}c{{y}} coordinates. For example, r1c1 is the top-left cell at row 1 column 1, r1c2 is the cell to the right at row 1 column 2, r2c1 is the cell below at row 2 column 1, and so on.

Visual Elements:
- Any visual elements will be described in text using rxcy coordinates.
- Please note the visual elements will be described as-is. If a thermo or arrow appears on the board, the location of the circle or bulb will be listed, and the line or arrow will be listed as a separate object. But you can infer they are part of the same object by their coordinates.
- If a visual element is described as "between" two cells, it means the visual element appears on the edge between the two cells.
- In some puzzles there may be visual elements outside of the grid and these will be described using the same coordinate system. For example an arrow in r0c1 pointing to the lower right means there is an arrow above r1c1 that points in the direction of the diagonal: r1c2, r2c3, etc.

## Image Input ##
We provide an image input as the initial board state and visual elements. You can use the provided tool to help you read and edit the image to solve the puzzle.

## Tips ##
- In solving the puzzle it often helps to understand that there exists a unique solution.
- It therefore helps to focus on what values must be forced given the puzzle constraints, and given the fact that the solution is unique.
- All information is provided and is sufficient to solve the puzzle.

## Size ## 
{rows}x{cols}

## Rules ##
{rules}

## Visual Elements ##
{visual_elements}

## Initial Sudoku Board ##
{initial_board}

## Answer Format ##
Thinking step by step with proper tool usage, please provide your answer at the end of your response. Put your answer within tags <ANSWER></ANSWER>. Your answer will be a sequence of {rows}x{cols} = {total_cells} digits.

For example, the format should look like
<ANSWER>
1234567...
</ANSWER>
'''


SUDOKU_TEXT_INPUT_TEMPLATE='''
You are a professional Sudoku puzzle solver. Please solve the following Sudoku variant.

## Format Explanation ##
Coordinates:
- We will use r{{x}}c{{y}} coordinates. For example, r1c1 is the top-left cell at row 1 column 1, r1c2 is the cell to the right at row 1 column 2, r2c1 is the cell below at row 2 column 1, and so on.

Visual Elements:
- Any visual elements will be described in text using rxcy coordinates.
- Please note the visual elements will be described as-is. If a thermo or arrow appears on the board, the location of the circle or bulb will be listed, and the line or arrow will be listed as a separate object. But you can infer they are part of the same object by their coordinates.
- If a visual element is described as "between" two cells, it means the visual element appears on the edge between the two cells.
- In some puzzles there may be visual elements outside of the grid and these will be described using the same coordinate system. For example an arrow in r0c1 pointing to the lower right means there is an arrow above r1c1 that points in the direction of the diagonal: r1c2, r2c3, etc.

## Tips ##
- In solving the puzzle it often helps to understand that there exists a unique solution.
- It therefore helps to focus on what values must be forced given the puzzle constraints, and given the fact that the solution is unique.
- All information is provided and is sufficient to solve the puzzle.

## Size ## 
{rows}x{cols}

## Rules ##
{rules}

## Visual Elements ##
{visual_elements}

## Initial Sudoku Board ##
{initial_board}

## Answer Format ##
Thinking step by step with proper tool usage, please provide your answer at the end of your response. Put your answer within tags <ANSWER></ANSWER>. Your answer will be a sequence of {rows}x{cols} = {total_cells} digits.

For example, the format should look like
<ANSWER>
1234567...
</ANSWER>
'''