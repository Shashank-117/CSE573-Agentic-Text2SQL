import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from text2sql_agents import SingleAgentConfig, SingleAgentSystem, extract_sql
from text2sql_eval.contracts import Budget, EvidenceDoc, QuestionInput, Status
from text2sql_llm import GenerationParams, LLMError, Message, OpenAICompatibleClient, ScriptedLLM
from text2sql_retrieval import build_table_docs, rank_tables, select_tables

GOOD = ("```sql\nSELECT d.dept_name, COUNT(*) FROM staff s JOIN dept d USING (dept_id) "
        "WHERE s.status = 'A' GROUP BY d.dept_name\n```")
NO_GPU = SingleAgentConfig(measure_gpu=False)


def _question(db, evidence=()):
    return QuestionInput("q1", "test", "company", str(db),
                         "How many active employees are in each department?", evidence=evidence)


# --------------------------------------------------------- SQL extraction

@pytest.mark.parametrize(
    "text,expected",
    [
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("Here:\n```sql\nSELECT 1\n```\nfixed:\n```sql\nSELECT 2\n```", "SELECT 2"),
        ("```\nSELECT 3\n```", "SELECT 3"),
        ("```sql\nSELECT 4 FROM t WHERE", "SELECT 4 FROM t WHERE"),  # cut off at max tokens
        ("SELECT 5", "SELECT 5"),
        ("The answer is SELECT 6 FROM t", "SELECT 6 FROM t"),
        ("I cannot answer that.", None),
        ("", None),
    ],
)
def test_extract_sql(text, expected):
    assert extract_sql(text) == expected


# ----------------------------------------------------- schema / retrieval

def test_table_docs_render_ddl_keys_and_samples(company_db):
    docs = {d.table: d for d in build_table_docs(str(company_db), "company_t1")}
    staff = docs["staff"]
    assert '"staff_id" INTEGER PRIMARY KEY' in staff.ddl
    assert 'FOREIGN KEY ("dept_id") REFERENCES "dept"("dept_id")' in staff.ddl
    assert "sample rows" in staff.sample and "Sales" not in staff.sample
    assert staff.references == ("dept",)
    assert "has no rows" in docs["audit_log"].sample


def test_bm25_ranks_relevant_table_first(company_db):
    docs = build_table_docs(str(company_db), "company_t2")
    ranked = rank_tables("which department name has the most staff", docs)
    assert ranked[-1][0].table == "audit_log"


def test_selection_priority_relevant_then_fk_then_filler(company_db):
    docs = build_table_docs(str(company_db), "company_t3")  # schema order: audit_log, dept, staff
    # "staff" is the only lexical hit; with room for two tables the FK target
    # (dept) must beat the higher-ranked-by-order filler table (audit_log).
    selection = select_tables("staff status", docs, token_budget=10**6, estimate_tokens=len, top_k=2)
    assert [t.table for t in selection.tables] == ["dept", "staff"]  # returned in schema order
    assert not selection.complete
    assert selection.ranked_ids[0] == "company_t3::staff"


def test_selection_top_k_is_hard_cap_and_budget_respected(company_db):
    docs = build_table_docs(str(company_db), "company_t4")
    only_one = select_tables("staff status", docs, token_budget=10**6, estimate_tokens=len, top_k=1)
    assert [t.table for t in only_one.tables] == ["staff"]
    staff_cost = len(next(d for d in docs if d.table == "staff").prompt_text)
    tight = select_tables("staff status", docs, token_budget=staff_cost, estimate_tokens=len)
    assert [t.table for t in tight.tables] == ["staff"] and tight.estimated_tokens <= staff_cost
    assert select_tables("staff", docs, token_budget=10**6, estimate_tokens=len).complete


# ------------------------------------------------------------ agent loop

def test_first_try_success(company_db):
    llm = ScriptedLLM([GOOD])
    result = SingleAgentSystem(llm, NO_GPU).predict(_question(company_db), Budget())
    assert result.status == "success"
    assert "GROUP BY" in result.generated_sql
    assert result.model_calls == 1 and result.retry_count == 0
    assert result.intermediate_artifacts["stop_reason"] == "ok"
    assert result.input_tokens > 0 and result.output_tokens > 0
    assert set(result.referenced_tables) == {"staff", "dept"}
    assert result.peak_gpu_memory_mb is None  # not measured -> None, never 0

    prompt = llm.calls[0][1].content
    assert 'CREATE TABLE "staff"' in prompt and "How many active employees" in prompt
    assert "sqlite" in llm.calls[0][0].content.lower()


def test_repairs_after_execution_error(company_db):
    llm = ScriptedLLM(["```sql\nSELECT nope FROM staff\n```", GOOD])
    result = SingleAgentSystem(llm, NO_GPU).predict(_question(company_db), Budget())
    assert result.model_calls == 2 and result.retry_count == 1
    repair = llm.calls[1]
    assert [m.role for m in repair] == ["system", "user", "assistant", "user"]
    assert "no such column: nope" in repair[-1].content
    attempts = result.intermediate_artifacts["attempts"]
    assert attempts[0]["problem"].startswith("Database error") and attempts[1]["problem"] is None


def test_repairs_unsafe_and_missing_sql(company_db):
    llm = ScriptedLLM(["I think you should delete things", "```sql\nDELETE FROM staff\n```", GOOD])
    result = SingleAgentSystem(llm, NO_GPU).predict(_question(company_db), Budget())
    problems = [a["problem"] for a in result.intermediate_artifacts["attempts"]]
    assert "not a single read-only SELECT" in problems[1]
    assert problems[2] is None and result.model_calls == 3


def test_stops_after_max_retries_and_returns_last_sql(company_db):
    llm = ScriptedLLM(["```sql\nSELECT bad1 FROM staff\n```"] * 10)
    result = SingleAgentSystem(llm, NO_GPU).predict(_question(company_db), Budget(max_retries=3))
    assert result.model_calls == 4  # 1 + 3 retries
    assert result.status == "success"  # harness will score it as execution_error
    assert result.generated_sql == "SELECT bad1 FROM staff"
    assert result.intermediate_artifacts["stop_reason"] == "max_retries"
    assert "no such column" in result.error_message


def test_parse_error_when_no_sql_ever(company_db):
    llm = ScriptedLLM(["no idea"] * 4)
    result = SingleAgentSystem(llm, NO_GPU).predict(_question(company_db), Budget())
    assert result.status == Status.PARSE_ERROR and result.generated_sql is None


def test_respects_model_call_and_token_budgets(company_db):
    llm = ScriptedLLM(["```sql\nSELECT bad FROM staff\n```"] * 10)
    capped = SingleAgentSystem(llm, NO_GPU).predict(_question(company_db), Budget(max_model_calls=2))
    assert capped.model_calls == 2 and capped.intermediate_artifacts["stop_reason"] == "budget_exhausted"

    tiny = SingleAgentSystem(ScriptedLLM([GOOD]), NO_GPU).predict(
        _question(company_db), Budget(max_total_input_tokens=10))
    assert tiny.model_calls == 0 and tiny.status == Status.PARSE_ERROR


def test_llm_failure_is_system_error(company_db):
    class Down:
        model = "down"

        def describe(self):
            return {}

        def generate(self, messages, params):
            raise LLMError("connection refused")

    result = SingleAgentSystem(Down(), NO_GPU).predict(_question(company_db), Budget())
    assert result.status == Status.SYSTEM_ERROR and "connection refused" in result.error_message


def test_evidence_included_and_truncated(company_db):
    evidence = (EvidenceDoc("rule.md", "Active means status = 'A'. " + "x" * 50_000),)
    llm = ScriptedLLM([GOOD])
    config = SingleAgentConfig(measure_gpu=False, evidence_token_budget=100)
    result = SingleAgentSystem(llm, config).predict(_question(company_db, evidence), Budget())
    prompt = llm.calls[0][1].content
    assert "[rule.md]" in prompt and "Active means status = 'A'" in prompt and "[... truncated]" in prompt
    assert "rule.md" in result.retrieved_document_ids


def test_retry_on_empty_result_option(company_db):
    empty = "```sql\nSELECT * FROM staff WHERE status = 'active'\n```"
    off = SingleAgentSystem(ScriptedLLM([empty]), NO_GPU).predict(_question(company_db), Budget())
    assert off.model_calls == 1
    config = SingleAgentConfig(measure_gpu=False, retry_on_empty_result=True)
    on = SingleAgentSystem(ScriptedLLM([empty, GOOD]), config).predict(_question(company_db), Budget())
    assert on.model_calls == 2


# ------------------------------------------------------- HTTP LLM client

class _Handler(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):  # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.received.append(body)
        reply = {"model": "m", "choices": [{"message": {"content": "```sql\nSELECT 1\n```"},
                                            "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 42, "completion_tokens": 7}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_openai_compatible_client_round_trip():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OpenAICompatibleClient(f"http://127.0.0.1:{server.server_port}/v1", "qwen")
        gen = client.generate([Message("user", "hi")], GenerationParams(seed=7, max_new_tokens=32))
    finally:
        server.shutdown()
    assert (gen.text, gen.input_tokens, gen.output_tokens, gen.finish_reason) == ("```sql\nSELECT 1\n```", 42, 7, "stop")
    sent = _Handler.received[-1]
    assert sent["seed"] == 7 and sent["max_tokens"] == 32 and sent["temperature"] == 0.0


def test_openai_client_unreachable_raises_llm_error():
    client = OpenAICompatibleClient("http://127.0.0.1:9/v1", "qwen", timeout_s=2)
    with pytest.raises(LLMError):
        client.generate([Message("user", "hi")], GenerationParams())
