"""TABULA RAG: grounded question answering over proprietary documents.

Public surface:

>>> from tabula_rag import RAGService, Settings
"""

from tabula_rag.config import Settings, get_settings
from tabula_rag.models import AnswerResult, Claim, ClaimStatus
from tabula_rag.pipeline import RAGService

__version__ = "1.0.0"

__all__ = [
    "AnswerResult",
    "Claim",
    "ClaimStatus",
    "RAGService",
    "Settings",
    "__version__",
    "get_settings",
]
