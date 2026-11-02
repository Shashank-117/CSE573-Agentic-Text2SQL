import re
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from .intent import sql_to_intent
from .models import BaselinePrediction, Text2SQLExample


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def lexical_schema_match(question: str, schema: Dict) -> Tuple[Optional[str], Optional[str]]:
    q = _tokens(question)
    best_table = None
    best_column = None
    best_score = 0

    for table in schema.get("tables", []):
        table_tokens = _tokens(table["name"].replace("_", " "))
        table_score = len(q & table_tokens)
        if table_score > best_score:
            best_score = table_score
            best_table = table["name"]
            best_column = None

        for column in table.get("columns", []):
            col_tokens = _tokens(column["name"].replace("_", " "))
            score = table_score + (2 * len(q & col_tokens))
            if score > best_score:
                best_score = score
                best_table = table["name"]
                best_column = column["name"]

    if best_table is None and schema.get("tables"):
        best_table = schema["tables"][0]["name"]
    return best_table, best_column


def render_template(intent: str, table: Optional[str], column: Optional[str]) -> Optional[str]:
    if not table:
        return None
    table_q = f'"{table}"'
    column_q = f'"{column}"' if column else "*"
    if intent == "COUNT":
        return f"SELECT COUNT(*) FROM {table_q};"
    if intent == "SUM" and column:
        return f"SELECT SUM({column_q}) FROM {table_q};"
    if intent == "SELECT":
        return f"SELECT {column_q} FROM {table_q};"
    # FILTER variants require value/condition extraction, deliberately unsupported in v1.
    return None


class ClassicalBaseline:
    def __init__(self, n_neighbors: int = 3):
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", metric="cosine")
        self.train_examples: List[Text2SQLExample] = []
        self.train_vectors = None

    def fit(self, examples: List[Text2SQLExample]) -> "ClassicalBaseline":
        if not examples:
            raise ValueError("Training examples cannot be empty")
        self.train_examples = examples
        questions = [x.question for x in examples]
        labels = [sql_to_intent(x.gold_sql) for x in examples]
        self.train_vectors = self.vectorizer.fit_transform(questions)
        k = min(self.classifier.n_neighbors, len(examples))
        self.classifier.set_params(n_neighbors=k)
        self.classifier.fit(self.train_vectors, labels)
        return self

    def predict(self, example: Text2SQLExample, schema: Dict) -> BaselinePrediction:
        vector = self.vectorizer.transform([example.question])
        intent = str(self.classifier.predict(vector)[0])
        distances, indices = self.classifier.kneighbors(vector, n_neighbors=1)
        nearest = self.train_examples[int(indices[0][0])].question
        table, column = lexical_schema_match(example.question, schema)
        sql = render_template(intent, table, column)
        return BaselinePrediction(
            example_id=example.example_id,
            db_id=example.db_id,
            question=example.question,
            gold_sql=example.gold_sql,
            predicted_intent=intent,
            predicted_sql=sql,
            supported=sql is not None,
            matched_table=table,
            matched_column=column,
            nearest_question=nearest,
        )
