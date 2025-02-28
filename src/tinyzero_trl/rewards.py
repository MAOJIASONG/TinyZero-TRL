# import debugpy; debugpy.connect(('localhost', 9501))
import os
import random
import re
import colorlog
import datetime
import json
import math
import tiktoken
import string

from typing import Dict
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from openai import OpenAI
from utils.system_prompts import ORM_PROMPT
from utils import is_e2b_available

if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import Sandbox

    load_dotenv()


global Reward_LLM, Reward_Tokenizer
Reward_LLM, Reward_Tokenizer = None, None


logger = colorlog.getLogger(__name__)


def get_reward_functions(training_args):
    """
    Retrieves and initializes reward functions based on training arguments.

    Args:
        training_args (object): A configuration object containing reward function settings.

    Returns:
        list: A list of reward function callables.
    """
    
    reward_funcs = [
        REWARD_FUNCS_REGISTRY[func] or REWARD_FUNCS_REGISTRY_WITH_ARGS[func](training_args)
        for func in getattr(training_args, "reward_funcs", [])
        if func in REWARD_FUNCS_REGISTRY or func in REWARD_FUNCS_REGISTRY_WITH_ARGS
    ]

    if not reward_funcs:
        logger.warning("No valid reward functions found. Using default 'accuracy'.")
        reward_funcs = [REWARD_FUNCS_REGISTRY["accuracy"]]
    
    return reward_funcs


def load_reward_model():
    """Loading Reward LLM model (lazy loading)"""
    global Reward_LLM, Reward_Tokenizer
    if Reward_LLM is None:
        logger.info("Loading Reward Model...")
        Reward_LLM = OpenAI(api_key="", base_url="")
        Reward_Tokenizer = tiktoken.encoding_for_model("gpt-4o")
        
        
def get_log_progress(logging_file: str):
    """
    Randomly save prompts and completions during training.
    """
    def log_progress(prompts, completions, **kwargs):
        if random.random() <= 0.1:
            os.makedirs("rollout_logs", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_content = "\n".join(
                [f"\n{'='*50}\n[{timestamp}]\nPrompt:\n{p}\n{'-'*50}\nCompletion:\n{c}" for p, c in zip(prompts, completions)]
            )
            with open(os.path.join("rollout_logs", logging_file), "a") as f:
                f.write(log_content)
        
        return [0.0] * len(prompts)
    
    return log_progress



def extract_answer(text):
    """Extract content between <answer> tags."""
    if text is None:
        return ""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
    


def evaluate_answer_similarity(completions, solution, **kwargs):
    """Reward function that evaluates the similarity between completions and ground truth using GPT-4o."""
    
    def normalize_answer(s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    load_reward_model()
        
    contents = [c[0]["content"] for c in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        try:
            response = Reward_LLM.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": ORM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"Student answer: {completions}\nCorrect solution: {solution}\nOutput only 1.0 or 0.0:"
                    }
                ],
                temperature=0.0
            )
            result = response.choices[0].message.content.strip()
            rewards.append(float(result))
        except Exception as e:
            print(f"Error in GPT evaluation: {e}")
            # If API call fails, fall back to simple text matching
            rewards.append(1.0 if normalize_answer(content) == normalize_answer(sol) else 0.0)

    return rewards


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [c[0]["content"] for c in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if not gold_parsed:
            logger.warning(f"Failed to parse gold solution: {solution}")
            rewards.append(1.0)
            continue

        try:
            answer_parsed = parse(content, extraction_config=[LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False, malformed_operators=False, basic_latex=True, equations=True, boxed="all", units=True),
                boxed_match_priority=0, try_extract_without_anchor=False
            )], extraction_mode="first_match")

            rewards.append(float(verify(answer_parsed, gold_parsed)))
        except Exception as e:
            logger.error(f"Verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
            rewards.append(0.0)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [c[0]["content"] for c in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [c[0]["content"] for c in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [c[0]["content"] for c in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [c[0]["content"] for c in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [c[0]["content"] for c in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def code_reward(completions, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    def extract_code(completion: str) -> str:
        pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ""
        return extracted_answer
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    rewards = []
    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    try:
        """Returns a reward function that evaluates code snippets in a sandbox."""
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
        verification_info = kwargs["verification_info"]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"]))
            )
            for code, info in zip(code_snippets, verification_info)
        ]
        with Sandbox(timeout=30, request_timeout=3) as sbx:
            for script in scripts:
                execution = sbx.run_code(script, language=verification_info["language"])
                try:
                    output = float(execution.text)
                except (TypeError, ValueError):
                    output = 0.0
                rewards.append(output)
    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)
    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def countdown_format_reward(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion, gt in zip(completions, target):

      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        if random.random() < 0.1:  # 1% chance to write samples into a file
          os.makedirs("completion_samples", exist_ok=True)
          log_file = os.path.join("completion_samples", "completion_samples.txt")
          with open(log_file, "a") as f:
            f.write(f"\n\n==============\n")
            f.write(completion)
        
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

        match = re.search(regex, completion, re.DOTALL) 
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
      except Exception:
        rewards.append(0.0)
        
    return rewards

def countdown_equation_reward(completions, target, nums, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            rewards.append(0.0)
            continue
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers):
            rewards.append(0.0)
            continue
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
           rewards.append(0.0)
           continue
        
        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(gt)) < 1e-5:
            rewards.append(1.0)
            if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)
        else:
            rewards.append(0.0)
      except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards



REWARD_FUNCS_REGISTRY = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "reasoning_steps": reasoning_steps_reward,
    "length": len_reward,
    "code": code_reward,
    "tag_count": tag_count_reward,
    "judge": evaluate_answer_similarity,
    "countdown_format": countdown_format_reward,
    "countdown_equation": countdown_equation_reward,
}

REWARD_FUNCS_REGISTRY_WITH_ARGS = {
    "cosine": lambda args: get_cosine_scaled_reward(
        min_value_wrong=args.cosine_min_value_wrong,
        max_value_wrong=args.cosine_max_value_wrong,
        min_value_correct=args.cosine_min_value_correct,
        max_value_correct=args.cosine_max_value_correct,
        max_len=args.cosine_max_len,
    ),
    "repetition_penalty": lambda args: get_repetition_penalty_reward(
        ngram_size=args.repetition_n_grams,
        max_penalty=args.repetition_max_penalty,
    ),
    "code_format": lambda args: get_code_format_reward(language=args.code_language),
    "print": lambda args: get_log_progress(logging_file=args.output_dir.split("/")[-1] + ".txt"),
}



if __name__ == "__main__":
    # Test reward functions
    def test_accuracy_reward_correct_answer():
        """Test accuracy_reward with a correct answer."""
        completion = [[{"content": r"\boxed{\frac{63}{400}}"}]]
        solution = [r"\frac{63}{400}"]

        rewards = accuracy_reward(completion, solution)
        assert rewards[0] ==1.0, f"Expected 1.0, got {rewards[0]}"
        print("Accuracy reward test passed for correct answer.")
        
    def test_accuracy_reward_incorrect_answer():
        """Test accuracy_reward with an incorrect answer."""
        completion = [[{"content": r"\boxed{\frac{63}{400}}"}]]
        solution = [r"\frac{64}{400}"]

        rewards = accuracy_reward(completion, solution)
        assert rewards[0] == 0.0, f"Expected 0.0, got {rewards[0]}"
        print("Accuracy reward test passed for incorrect answer.")
    
    def test_format_reward_correct():
        """Test format_reward with correct format."""
        completion = [[{"content": "<think>\nSome reasoning\n</think>\n<answer>\nSome answer\n</answer>"}]]

        rewards = format_reward(completion)
        assert rewards[0] == 1.0, f"Expected 1.0, got {rewards[0]}"
        print("Format reward test passed for correct answer.")
        
    def test_format_reward_incorrect():
        """Test format_reward with incorrect format."""
        incorrect_formats = [
            "<think>Only thinking</think>",
            "<answer>Only answer</answer>",
            "No tags at all",
            "<think>Missing closing</think><answer>Missing closing",
            "<think>Wrong order</answer><answer>Wrong order</think>",
        ]

        for fmt in incorrect_formats:
            completion = [[{"content": fmt}]]
            rewards = format_reward(completion)
            assert rewards[0] == 0.0, f"Expected 0.0, got {rewards[0]}"
        print("Format reward test passed for incorrect answer.")
        
    def test_reasoning_steps_reward():
        """Test reasoning_steps_reward with various formats."""
        test_cases = [
            # Full credit cases (3 or more steps)
            ("Step 1: First step\nStep 2: Second step\nStep 3: Third step", 1.0),
            ("First, we do this.\nSecond, we do that.\nFinally, we conclude.", 1.0),
            # Partial credit cases (less than 3 steps)
            ("Step 1: Only step", 1 / 3),
            ("First, we do this.\nFinally, we conclude.", 2 / 3),
            # No credit case
            ("Just plain text without any clear steps", 0.0),
        ]

        for content, expected_reward in test_cases:
            completion = [[{"content": content}]]
            rewards = reasoning_steps_reward(completion)
            assert rewards[0]-expected_reward < 0.01, f"Expected AlmostEqual, got {rewards[0]}"
        print("Reasoning steps reward test passed.")
    
    def test_multiple_completions():
        """Test handling multiple completions at once."""
        completions = [[{"content": r"\boxed{\frac{63}{400}}"}], [{"content": r"\boxed{\frac{64}{400}}"}]]
        solutions = [r"\frac{63}{400}", r"\frac{63}{400}"]

        rewards = accuracy_reward(completions, solutions)
        assert len(rewards)==2, f"Expected 2 rewards, got {len(rewards)}"
        assert rewards[0] == 1.0, f"Expected 1.0, got {rewards[0]}"
        assert rewards[1] == 0.0, f"Expected 0.0, got {rewards[1]}"
        print("Multiple completions test passed.")
    
    test_accuracy_reward_incorrect_answer()
    