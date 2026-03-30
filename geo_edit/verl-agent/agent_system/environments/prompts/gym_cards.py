# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- Gym Cards --------------------- #
GYM_CARDS_NUMBERLINE_TEMPLATE = """
<image>You are playing a game called number line. You will see a target number and a current number in the image. And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number.

Your response should be a valid json file in the following format: 
{{
"current number": "x",
"target number": "x",
"thoughts": "first read out the current and target number, then think carefully about which action to choose",
"action": "-" or "+" 
}}
"""

GYM_CARDS_BLACKJACK_TEMPLATE = """
<image>You are an expert blackjack player helping to decide the optimal action based on the current game state displayed in the image. 

From the image, you can see:
- Dealer (top): one face-up card and one face-down card.  
- Player (bottom): every card in your hand, laid left-to-right (wraps after five).

Game Rules:
- Your goal is to get as close to 21 as possible without going over.
- Number cards (2–10) = their value. Face cards (J, Q, K) = 10. Aces (\"A\") = 1 or 11 (an ace is always counted as 11 (usable) unless it causes a bust).
- After you choose "stand", the dealer reveals the hidden card and draws until the hand total is 17 or higher.
- If a hand exceeds 21, it busts and loses immediately.
- A natural blackjack (Ace+10) can get an extra reward.
- The deck is infinite (with replacement).

Admissible Actions:
- "hit": take another card.
- "stand": stop and let the dealer play.

Your response should be a valid json file in the following format:
{{
"thoughts": "Analyze the image to identify your cards and the dealer's visible card. Then, reason step-by-step based on optimal blackjack strategy under the above rules.",
"action": "an admissible action"
}}
"""

GYM_CARDS_EZPOINTS_TEMPLATE = """
<image>You are an expert card game player helping to build a math formula that evaluates to **12**, using only the two numbers shown on the playing cards in the image. You may choose from the following actions: one of the two available card numbers, the operators '+' or '*', or the equals sign '='. You must build the formula step by step, adding only one character at a time to the end of the current formula.

Card Rules:
1. The image shows exactly two playing cards.
2. Each card represents one number:
   - Number cards (2–10) equal their face value.
   - Face cards ('J', 'Q', 'K') are treated as the number 10.
   - Ace ('A') is treated as 1.
3. You can only use these two card numbers — no other numbers are allowed.
4. Each number can be used only once in the formula.

Formula Rules:
1. At each step, you append only one character: a number (from the two card values), an operator ('+' or '*'), or '='.
2. Once you add '=', the formula is evaluated and the game ends.

Rewards:
+10: if the formula evaluates exactly to 12.
0: otherwise (e.g., using a number not shown on the cards, reusing a number, incorrect syntax, not evaluating to 12).

Now, you are given two card numbers as shown in the image, and the current formula is: '{text_formula}'
It's your turn to append the next character.

Your response MUST be a valid JSON object in the following format:
{{
  "thoughts": "Start by identifying the two card numbers shown in the image. Then review the current formula. Based on the remaining available characters and the goal of reaching 12, reason step-by-step to choose the next valid character to add.",
  "action": "next character (number, '+', '*', or '=') to append"
}}
"""

GYM_CARDS_POINTS24_TEMPLATE = """
<image>You are an expert at solving the classic "24 Game", where you build an formula that evaluates exactly to **24**, using only the four numbers shown on the playing cards in the image. You may choose from the following actions: one of the four available card numbers, the operators ('+', '-', '*', '/'), the parentheses '(', ')', and the equals sign '='.

Card Rules:
1. The image shows exactly four playing cards.
2. Each card represents one number:
   - Number cards (2–10) equal their face value.
   - Face cards ('J', 'Q', 'K') are treated as 10.
   - Ace ('A') is treated as 1.
3. You can only use these four card numbers — no other numbers are allowed.
4. Each card can be used only once in the formula.

Formula Rules:
1. At each step, you append only one character: a number (from the four card values), an operator ('+', '-', '*', '/'), a parenthes ('(', ')'), or '='.
2. Once you add '=', the formula is evaluated and the game ends.

Rewards:
+10: if the final formula is valid and equals 24, and all four card numbers are used.
0: otherwise (e.g., using a number not shown on the cards, reusing a number, incorrect syntax, not evaluating to 24).

----
Here are three examples.

Example 1:
The current formula is: '8*'  
Card numbers: '8', '3', '2', '1'  
Since '8*3*(2-1)=24', the correct action is: '3'

Example 2:
The current formula is: '(10+2)*(4-2)'
Card numbers are: '4', '10', '2', '2'
Since '(10+2)*(4-2)=24', the correct action is: '='

Example 3:
The current formula is: ''
Card numbers are: '6', '8', '1', '1'
Since '6*8/(1+1)=24', the correct action is: '6'
----

Now, you are given four card numbers as shown in the image, and the current formula is: '{text_formula}'
It's your turn to append the next character.

Your response MUST be a valid JSON object in the following format:
{{
  "thoughts": "Start by identifying the four card numbers shown in the image. Then analyze the current formula. Determine what characters (card numbers, operators, or parentheses) are still available. Reason step-by-step how to build a valid formula that equals 24 and figure out the next character to append.",
  "action": "next character (number, '+', '-', '*', '/', '(', ')', or '=') to append"
}}
"""
