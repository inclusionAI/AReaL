from typing import List


# Use the same projection as appworld, replace <code>...</code> with <action>...</action>
def tau2_projection(actions: List[str]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    """
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                extracted_action = actions[i][-100:]
                valids[i] = 0
                actions[i] = extracted_action
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag) : end_idx]

            actions[i] = extracted_action
            valids[i] = 1

        except:
            extracted_action = actions[i][-100:]
            valids[i] = 0
            actions[i] = extracted_action

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

    return actions, valids
