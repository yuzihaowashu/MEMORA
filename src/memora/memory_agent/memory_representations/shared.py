"""Response guidance shared by memory-retrieval tasks."""

FORCED_ANSWER_PROMPT = """You have reached the maximum number of search iterations. Based on ALL the information you have gathered so far, you MUST now provide your final answer.

## Requirements:
1. **YOU MUST PROVIDE AN ANSWER** - Do not say "Cannot determine" unless you truly have zero relevant information
2. **Use reasoning** - Analyze all the search results and infer the most likely answer
3. **Be definite** - Even if information is partial or ambiguous, make your best informed guess
4. **Do not search again** - No more tool calls allowed

## Output Format:
First, provide a brief reasoning process explaining how you arrive at your answer based on the gathered information.
Then, output your final answer in this exact format:
**Answer: X** (where X is A, B, C, or D)

## Example:
Based on the search results:
- Activity at 70-80s shows "Person rinses plate under running water"
- This indicates the faucet must have been on at that time
- The faucet is mentioned as running but no explicit "turn on" action was found
- However, the activity at 30-40s shows initial water usage, suggesting the faucet was turned on near the beginning

**Answer: A**

Now, analyze your gathered information and provide your final answer:"""

FORCED_ANSWER_PROMPT_SHORT_ANSWER = """You have reached the maximum number of search iterations. Based on ALL the information you have gathered so far, you MUST now provide your final answer.

## Requirements:
1. **YOU MUST PROVIDE AN ANSWER** - Do not say "Cannot determine" unless you truly have zero relevant information
2. **Use reasoning** - Analyze all the search results and infer the most likely answer
3. **Be definite** - Even if information is partial or ambiguous, make your best informed guess
4. **Do not search again** - No more tool calls allowed

## Output Format:
First, provide a brief reasoning process explaining how you arrive at your answer based on the gathered information.
Then, output your final answer in this exact format:
**Answer: <your concise answer>**

## Example:
Based on the search results:
- Object search found "white cloth" with location "on counter"
- Activity at 20s shows "Person places cloth on counter near sink"

**Answer: on counter**

Now, analyze your gathered information and provide your final answer:"""

SHORT_ANSWER_GUIDANCE = (
    '\n\n## SHORT-ANSWER MODE\n'
    'This question has NO multiple-choice options. You must generate a concise, '
    'specific answer based on what you find in memory.\n'
    '- **Final answer format:** `**Answer: <your concise answer>**`\n'
    '- Example: **Answer: on the counter near the sink**\n'
    '- Do NOT output a letter (A/B/C/D). Output the actual answer text.\n'
    '- Keep answers short (1-15 words). Be specific.\n'
)
