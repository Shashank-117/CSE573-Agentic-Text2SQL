from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Text2SQLExample:
    """
    Common representation for Text2SQL benchmark examples.
    """

    example_id: str
    dataset: str
    split: str
    db_id: str
    question: str

    gold_sql: Optional[str] = None
    dialect: str = "sqlite"
    external_knowledge: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselinePrediction:
    """
    Prediction produced by the classical Text2SQL baseline.
    """

    example_id: str
    db_id: str
    question: str
    gold_sql: Optional[str]

    predicted_intent: str
    predicted_sql: Optional[str]

    supported: bool

    matched_table: Optional[str] = None
    matched_column: Optional[str] = None
    nearest_question: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)