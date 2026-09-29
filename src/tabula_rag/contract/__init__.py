"""Contract mode: the exact shape the Mini-Challenge 3 harness invokes and grades.

The service in the rest of this package answers in sentences with chunk-level citations, which
is right for people and wrong for the grader. This subpackage keeps the same ideas (grounding as
a veto, abstention over guessing, revision awareness) and changes their shape to the challenge:
value-only answers, file-level citation sets, a persisted index, and a fresh process per
question.
"""

from tabula_rag.contract.answer import QueryResult, answer_query
from tabula_rag.contract.config import ContractSettings
from tabula_rag.contract.index import ContractIndex

__all__ = ["ContractIndex", "ContractSettings", "QueryResult", "answer_query"]
