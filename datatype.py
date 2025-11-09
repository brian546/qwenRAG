from types import Dict, Any
from dataclasses import dataclass

@dataclass
class Document:
    """Class to represent a document chunk."""
    text: str
    metadata: Dict[str, Any]