TOOL_CROP_SYSTEM_PROMPT="""You are a helpful assistant. Answer the user's question based on the image provided. Output your thinking process within the <think> and </think> tags. Whenever you find anything unclear, you can zoom in a specific region in the given image to see more clearly by outputing <grounding>{\"bbox_2d\": [x0, y0, x1, y1], \"source\": \"original_image\"}</grounding>, where (x0, y0) and (x1, y1) are the top-left and bottom-right coordinates of the region that you want to zoom in, respectively (suppose the width and height of the image are 1.0), and 'source' refers to the image that you zoom in and could be either 'original_image' or 'observation_i'. Once the final answer is confirmed, put it within <answer> and </answer>."""

SYSTEM_PROMPT_MAP={
    "tool_crop": TOOL_CROP_SYSTEM_PROMPT,
}

TOOL_CALL_CROP_MULTI_TRUN_PROMPT="After the above Action {action_turn}, here is the the zoom-in image (Observation {observation_turn}):\n<|vision_start|><|image_pad|><|vision_end|>.\nContinue your reasoning process inside <think> and </think>. If needed, you can continue to zoom in on the original image or any of the observations, by outputting <grounding> and </grounding> as before. If the final answer is confirmed, put your final answer inside <answer> and </answer>."

ERROR_INFO_MULTI_TURN_PROMPT="Please analyze the error information obtained from the function tool and adjust your response. Countinue your reasoning process inside <think> and </think>."

INVALID_PROMPT="Note that when the Assistant finds the question is invalid, e.g. no answer could match all information in the question, the Assistant replies with '<answer> the question is invalid. </answer>'."
VALID_PROMPT="You should try to find the most likely answer. "