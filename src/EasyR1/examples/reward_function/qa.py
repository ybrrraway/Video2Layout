# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
from typing import Any, Dict, List
import numpy as np

def extract_final_answer(response: str) -> str | None:
    """
    Extracts the content from within the <answer>...</answer> tag.
    Returns the extracted string or None if the tag is not found.
    """
    # Use re.DOTALL so that '.' matches any character, including newlines.
    # Use non-greedy '.*?' to find the shortest possible match.
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        # .strip() removes leading/trailing whitespace and newlines from the answer.
        return match.group(1).strip()
    return None

import re

def format_reward(response: str) -> float:
    cleaned_response = response.strip()

    # 1. 优先检查完整格式
    full_pattern = re.compile(
        r"<map>.*?</map>\s*<think>.*?</think>\s*<answer>.*?</answer>",
        re.DOTALL
    )
    if re.fullmatch(full_pattern, cleaned_response):
        return 1.0

    # # 2. 如果完整格式不匹配，则检查精简格式
    # answer_only_pattern = re.compile(
    #     r"<map>.*?</map>\s*<answer>.*?</answer>",
    #     re.DOTALL
    # )
    # if re.fullmatch(answer_only_pattern, cleaned_response):
    #     return 1.0
    
    # answer_only_pattern = re.compile(
    #     r"<think>.*?</think>\s*<answer>.*?</answer>",
    #     re.DOTALL
    # )
    # if re.fullmatch(answer_only_pattern, cleaned_response):
    #     return 1.0
    
    # answer_only_pattern = re.compile(
    #     r"<answer>.*?</answer>",
    #     re.DOTALL
    # )
    # if re.fullmatch(answer_only_pattern, cleaned_response):
    #     return 1.0

    # 3. 两种格式都不匹配
    return 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Calculates the accuracy score based on the ground truth type.
    - For string types, it performs a case-insensitive exact match.
    - For numeric types, it uses Multi-threshold Relative Accuracy (MRA) scoring.
    """
    answer = extract_final_answer(response)
    if answer is None:
        return 0.0  # No answer found, so accuracy is 0.

    # Try to convert ground_truth and answer to floats to see if it's a numeric question.
    try:
        gt_num = float(ground_truth)
        pred_num = float(answer)

        # --- This is a NUMERICAL question ---
        # Define the 10 thresholds for relative error, from 50% down to 5%.
        mra_error_thresholds = np.linspace(0.50, 0.05, 10)

        # Calculate the error. Handle the case where the ground truth is 0.
        if gt_num == 0:
            # If ground truth is 0, relative error is undefined. We use absolute error.
            error = abs(pred_num - gt_num)
        else:
            # Otherwise, calculate the standard relative error.
            error = abs(pred_num - gt_num) / abs(gt_num)

        # Count how many thresholds the error successfully passes (i.e., is less than or equal to).
        passed_count = np.sum(error <= mra_error_thresholds)

        # The final score is the fraction of thresholds passed.
        return float(passed_count) / len(mra_error_thresholds)

    except (ValueError, TypeError):
        # --- This is a STRING question ---
        # If conversion to float fails, we treat it as a string comparison.
        # .lower() makes the comparison case-insensitive.
        return 1.0 if answer.lower() == str(ground_truth).lower() else 0.0

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    """
    Computes the overall, format, and accuracy scores for a batch of responses.
    The overall score is a weighted average of the format and accuracy scores.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this math reward function.")

    scores = []
    for reward_input in reward_inputs:
        
        # breakpoint()
        
        # This preprocessing step from your original code removes whitespace around tag characters.
        # It can be useful if the model output has inconsistent spacing like "< / analyse >".
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        ground_truth = reward_input["ground_truth"]

        # Calculate the two components of the score.
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, ground_truth)

        # Combine them into a final weighted score.
        overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score

        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
