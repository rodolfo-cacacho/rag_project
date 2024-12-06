import os
import sys

# Correct computation of project root to align with the actual directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
print("Computed project root:", project_root)

# Ensure the correct project root is added to `sys.path`
if project_root not in sys.path:
    sys.path.append(project_root)

from config import (CONFIG_SQL_DB,DB_NAME,
                    SQL_EVAL_QAS_TABLE,TEST_GEN_ANSWERS_SCHEMA,
                    TEST_GEN_ANSWERS_TABLE,TEST_RESULTS_TABLE,
                    SQL_PROMPTS_TABLE
                    )
from utils.MySQLDB_manager import MySQLDB
from testing.modules.testing_modules import evaluate_questions

sql_con = MySQLDB(CONFIG_SQL_DB,DB_NAME)

rubric = """
1: The response is irrelevant, misaligned with the Expected Answer, or off-topic.
   - Includes hallucinated content or fabricated details that contradict the Expected Answer.
   - Does not address the question or provides misleading information.
   Example: An answer that fabricates regulatory clauses or misrepresents key terms.

2: The response admits it cannot provide an answer due to a lack of information or context.
   - While it does not address the question, it is honest and avoids misleading information or hallucinations.
   Example: "I do not have sufficient context to answer this question."

3: The response is relevant to the question but contains inaccuracies, incomplete explanations, or slight deviations from the Expected Answer.
   - Some parts of the response align with the Expected Answer, but discrepancies reduce reliability.
   - A vague question may lead to a partially relevant answer.
   Example: An answer that explains key terms but omits critical regulatory details or misinterprets context.

4: The response is accurate, relevant, and aligned with the Expected Answer.
   - It provides sufficient information to address the question but lacks depth or supplementary details.
   - The response is acceptable for practical use but does not exceed expectations.
   Example: An answer that correctly summarizes a regulation but does not provide specific clause numbers or examples.

5: The response is fully accurate, aligns with the Expected Answer, and is comprehensive.
   - It provides additional insights, supplementary details, or context beyond the minimum required.
   - Demonstrates a thorough understanding of the question and Expected Answer.
   Example: An answer that explains the regulation, provides examples, and contextualizes its impact on energy efficiency projects.
"""

results = evaluate_questions(sql_con=sql_con,
                   qas_table=SQL_EVAL_QAS_TABLE,
                   test_results_table=TEST_RESULTS_TABLE,
                   test_answers_table=TEST_GEN_ANSWERS_TABLE,
                   test_answers_schema=TEST_GEN_ANSWERS_SCHEMA,
                   prompts_table=SQL_PROMPTS_TABLE,
                   rubric=rubric,
                   overwrite=False)
