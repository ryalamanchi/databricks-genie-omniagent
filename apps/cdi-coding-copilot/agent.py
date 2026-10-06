"""CDI Coding Agent — grounded ICD-10/CPT coding + CDI gap detection.

Takes a signed, PII-masked clinical note and produces coder-ready output:

  1. extract clinical entities (diagnosis / symptom / medication / procedure)
  2. assign codes using ONLY the Unity Catalog codebook — hallucinated codes are
     rejected in code, never surfaced to a coder
  3. locate every evidence quote verbatim in the note -> char offsets, so the UI
     can highlight the exact justification; unlocatable quotes cap confidence
  4. compute an explainable Coding Risk Score (0-100) in Python, not by the LLM
  5. surface CDI documentation gaps with compliant, NON-LEADING physician queries

Used by the Dash app (live "sign note" review) and re-usable from a batch job.
All persistence is parameterised — no SQL string interpolation.
"""
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.sdk.service.sql import StatementParameterListItem

try:
    import mlflow
    from mlflow.entities import SpanType
except ImportError:  # tracing is optional, agent must still run
    mlflow = None
    SpanType = None

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ENDPOINT = os.environ.get("CDI_AGENT_ENDPOINT", "databricks-claude-sonnet-5")
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "22998c886e21ee6c")
CATALOG = os.environ.get("CDI_CATALOG", "`external-ai-build-day`")
SCHEMA_CDI = f"{CATALOG}.cdi_copilot"
SCHEMA_CC = f"{CATALOG}.clinical_coding"

PROMPT_VERSION = "cdi-coder-v2"
UNVERIFIED_CONFIDENCE_CAP = 0.5   # quote not found in note -> cannot trust the code
MAX_TOKENS = 4000

SYSTEM_PROMPT = """You are an expert inpatient/outpatient clinical coder and CDI (Clinical Documentation Integrity) specialist.
You review a signed clinical note whose PHI has already been masked (tokens such as [PATIENT], [CLINICIAN], [MRN], <PERSON>).
Never attempt to re-identify the patient and never invent PHI.

Tasks:
1. Extract the clinical entities actually documented: diagnoses, symptoms, medications, procedures.
2. Assign codes using ONLY codes present in the CODEBOOK provided below. Never invent or infer a code
   that is not in the codebook.
   - Code only what the clinician documented. Do not code from suspicion, history or ruled-out findings.
   - Prefer the most specific code the documentation actually supports.
   - Include CPT codes for documented procedures and E&M services when they appear in the codebook.
   - Mark exactly one code "principal": true when the note supports a principal diagnosis.
3. For EVERY code supply "evidence_quote": an EXACT, verbatim, contiguous substring copied character-for-character
   from the note (roughly 5-25 words) that justifies the code. Do not paraphrase, do not stitch fragments,
   do not add ellipses. If you cannot quote the note, do not emit the code.
4. Supply "confidence" 0.0-1.0 that the code is correct AND fully supported by the documentation.
5. Identify CDI documentation gaps: vague, missing or conflicting documentation that blocks a more specific or
   higher-acuity code, or that creates payer denial risk (e.g. sepsis without documented organ dysfunction,
   heart failure without type/acuity, pneumonia without organism, debridement without depth, no linking
   statement between diabetes and its manifestation).
   For each gap write a compliant physician query in AHIMA/ACDIS style: state the clinical indicators found in the
   note, ask an open or multiple-choice question, ALWAYS include "other" and "clinically undetermined" as options,
   and never suggest which answer increases reimbursement. Non-leading queries only.

Respond with JSON only, no prose, no markdown fence, matching exactly:
{
  "entities": [{"type": "diagnosis|symptom|medication|procedure", "text": "..."}],
  "codes": [{"code": "...", "code_system": "ICD-10|CPT", "evidence_quote": "...",
             "rationale": "one sentence", "confidence": 0.0, "principal": false}],
  "cdi_gaps": [{"gap": "...", "impact": "coding / reimbursement / denial impact",
                "evidence_quote": "...", "physician_query": "..."}]
}"""


# ── Tracing (optional) ──────────────────────────────────────────────────────
_tracing_ready = False


def init_tracing() -> bool:
    """Enable MLflow tracing when MLFLOW_EXPERIMENT_ID is bound; silently no-op otherwise."""
    global _tracing_ready
    if _tracing_ready or mlflow is None or not os.environ.get("MLFLOW_EXPERIMENT_ID"):
        return _tracing_ready
    try:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(experiment_id=os.environ["MLFLOW_EXPERIMENT_ID"])
        mlflow.config.enable_async_logging(True)
        _tracing_ready = True
    except Exception as e:  # tracing must never break the coder workflow
        print(f"[agent] MLflow tracing disabled: {e}")
    return _tracing_ready


class _NoSpan:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def set_inputs(self, *a, **k):
        pass

    def set_outputs(self, *a, **k):
        pass

    def set_attributes(self, *a, **k):
        pass


def _span(name, span_type=None):
    if _tracing_ready:
        return mlflow.start_span(name=name, span_type=span_type)
    return _NoSpan()


# ── SQL helpers (Statement Execution API, parameterised) ────────────────────
def _exec(w: WorkspaceClient, stmt: str, params=None):
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=stmt,
        parameters=params,
        wait_timeout="50s",
    )
    if resp.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"SQL failed: {resp.status.error or 'unknown error'}")
    return resp


def _rows(resp):
    cols = [c.name for c in resp.manifest.schema.columns]
    data = resp.result.data_array if resp.result and resp.result.data_array else []
    return [dict(zip(cols, r)) for r in data]


def load_codebook(w: WorkspaceClient) -> list:
    """The allow-list of billable codes. Agent output is constrained to exactly these."""
    resp = _exec(w, f"""
        SELECT code, code_system, description, avg_reimbursement
        FROM {SCHEMA_CC}.icd10_cpt_ref
        ORDER BY code_system, code
    """)
    return [{
        "code": r["code"],
        "code_system": r["code_system"],
        "description": r["description"],
        "avg_reimbursement": int(float(r["avg_reimbursement"] or 0)),
    } for r in _rows(resp)]


def load_denial_priors(w: WorkspaceClient, payer: str) -> dict:
    """{code: {denial_rate, top_denial_reason}} historical priors for this encounter's payer."""
    if not payer:
        return {}
    resp = _exec(w, f"""
        SELECT code, denial_rate, top_denial_reason
        FROM {SCHEMA_CDI}.gold_code_denial_priors
        WHERE payer = :payer
    """, [StatementParameterListItem(name="payer", value=str(payer))])
    return {
        r["code"]: {
            "denial_rate": float(r["denial_rate"] or 0),
            "top_denial_reason": r["top_denial_reason"] or "",
        } for r in _rows(resp)
    }


def load_note(w: WorkspaceClient, encounter_id: str) -> dict:
    """Fetch the PII-masked note for an encounter. Raw text never leaves the silver layer."""
    resp = _exec(w, f"""
        SELECT note_id, encounter_id, masked_text, payer, service_line, encounter_type
        FROM {SCHEMA_CDI}.silver_notes_masked
        WHERE encounter_id = :enc
        LIMIT 1
    """, [StatementParameterListItem(name="enc", value=str(encounter_id))])
    rows = _rows(resp)
    return rows[0] if rows else {}


# ── Grounding ───────────────────────────────────────────────────────────────
def locate_quote(quote: str, text: str):
    """Return (start, end) of `quote` in `text`, tolerant of case/whitespace drift; None if absent."""
    if not quote or not text:
        return None
    q = quote.strip().strip('"').strip("'").strip()
    if not q:
        return None
    i = text.find(q)
    if i >= 0:
        return i, i + len(q)
    words = [re.escape(t) for t in q.split()]
    if not words:
        return None
    m = re.search(r"\s+".join(words), text, flags=re.IGNORECASE)
    if m:
        return m.start(), m.end()
    # models often drop trailing punctuation or a leading article - retry the inner run
    if len(words) > 6:
        m = re.search(r"\s+".join(words[1:-1]), text, flags=re.IGNORECASE)
        if m:
            return m.start(), m.end()
    return None


def _parse_json(content: str) -> dict:
    content = (content or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL)
    if fence:
        content = fence.group(1)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in model output")
    return json.loads(content[start:end + 1])


def coding_risk_score(codes, gaps, denial_priors=None):
    """Explainable 0-100 score — higher means this chart needs a coder first.

    Reimbursement-weighted so a shaky $12k code outranks a shaky $80 code:
      low_confidence        0-40  weighted mean confidence across suggested codes
      cdi_gaps              0-25  open documentation gaps (capped at 3)
      payer_denial_prior    0-25  worst historical denial rate for this payer+code
      unverified_evidence   0-10  share of codes whose quote was not found in the note
    """
    denial_priors = denial_priors or {}
    if codes:
        weights = [max(c.get("avg_reimbursement") or 0, 1) for c in codes]
        conf = sum(wt * c["confidence"] for wt, c in zip(weights, codes)) / sum(weights)
        denial = max(
            float((denial_priors.get(c["code"]) or {}).get("denial_rate", 0) or 0)
            for c in codes
        )
        unverified = sum(not c["evidence_verified"] for c in codes) / len(codes)
    else:
        conf, denial, unverified = 0.0, 0.0, 1.0
    components = {
        "low_confidence": round((1 - conf) * 40, 1),
        "cdi_gaps": round(min(len(gaps), 3) / 3 * 25, 1),
        "payer_denial_prior": round(min(denial / 0.5, 1) * 25, 1),
        "unverified_evidence": round(unverified * 10, 1),
    }
    return round(sum(components.values()), 1), components


def risk_tier(score: float) -> str:
    if score >= 60:
        return "HIGH RISK"
    if score >= 35:
        return "MEDIUM RISK"
    return "LOW RISK"


# ── LLM call ────────────────────────────────────────────────────────────────
def _call_llm(w: WorkspaceClient, note_text: str, codebook: list):
    codebook_txt = "\n".join(
        f"{c['code']} | {c['code_system']} | {c['description']}" for c in codebook
    )
    user = (
        f"CODEBOOK (code | system | description):\n{codebook_txt}\n\n"
        f"SIGNED CLINICAL NOTE (PHI masked):\n<<<\n{note_text}\n>>>"
    )
    with _span("llm_extract_and_code", SpanType.CHAT_MODEL if mlflow else None) as span:
        span.set_inputs({
            "endpoint": MODEL_ENDPOINT,
            "prompt_version": PROMPT_VERSION,
            "codebook_size": len(codebook),
            "note_chars": len(note_text),
        })
        resp = w.serving_endpoints.query(
            name=MODEL_ENDPOINT,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=SYSTEM_PROMPT),
                ChatMessage(role=ChatMessageRole.USER, content=user),
            ],
            max_tokens=MAX_TOKENS,
        )
        content = resp.choices[0].message.content
        if isinstance(content, list):  # reasoning models return content blocks
            content = "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        usage = resp.usage.as_dict() if resp.usage else {}
        span.set_outputs({"content": content, "usage": usage})
    return content, usage


# ── Agent entry point ───────────────────────────────────────────────────────
def code_note(w: WorkspaceClient, note_id: str, encounter_id: str, note_text: str,
              codebook: list, denial_priors: dict = None, payer: str = None) -> dict:
    """Run the coding agent over one masked note and return grounded, scored output."""
    init_tracing()
    denial_priors = denial_priors or {}
    book = {c["code"]: c for c in codebook}
    t0 = time.time()

    with _span("cdi_code_note", SpanType.AGENT if mlflow else None) as root:
        root.set_inputs({"note_id": note_id, "encounter_id": encounter_id, "payer": payer})
        content, usage = _call_llm(w, note_text, codebook)
        try:
            raw = _parse_json(content)
        except Exception as e:
            raise RuntimeError(f"Agent returned non-JSON output ({e}): {str(content)[:300]}")

        with _span("ground_and_score", SpanType.PARSER if mlflow else None) as gspan:
            codes, rejected, seen = [], [], set()
            for c in raw.get("codes", []):
                code = str(c.get("code", "")).strip().upper()
                if code not in book:          # hallucination guard
                    rejected.append(code)
                    continue
                if code in seen:
                    continue
                seen.add(code)
                pos = locate_quote(c.get("evidence_quote", ""), note_text)
                conf = max(0.0, min(1.0, float(c.get("confidence", 0) or 0)))
                if pos is None:
                    conf = min(conf, UNVERIFIED_CONFIDENCE_CAP)
                prior = denial_priors.get(code) or {}
                codes.append({
                    "code": code,
                    "code_system": book[code]["code_system"],
                    "code_description": book[code]["description"],
                    "avg_reimbursement": int(book[code]["avg_reimbursement"] or 0),
                    "confidence": round(conf, 3),
                    "principal": bool(c.get("principal", False)),
                    "rationale": c.get("rationale", ""),
                    "evidence_quote": note_text[pos[0]:pos[1]] if pos else c.get("evidence_quote", ""),
                    "evidence_start": pos[0] if pos else None,
                    "evidence_end": pos[1] if pos else None,
                    "evidence_verified": pos is not None,
                    "denial_rate": prior.get("denial_rate"),
                    "top_denial_reason": prior.get("top_denial_reason"),
                })
            codes.sort(key=lambda c: (not c["principal"], -c["avg_reimbursement"]))

            gaps = []
            for g in raw.get("cdi_gaps", []):
                pos = locate_quote(g.get("evidence_quote", ""), note_text)
                gaps.append({
                    "gap": g.get("gap", ""),
                    "impact": g.get("impact", ""),
                    "physician_query": g.get("physician_query", ""),
                    "evidence_quote": g.get("evidence_quote", ""),
                    "evidence_start": pos[0] if pos else None,
                    "evidence_end": pos[1] if pos else None,
                })

            score, components = coding_risk_score(codes, gaps, denial_priors)
            gspan.set_outputs({
                "n_codes": len(codes),
                "rejected_codes": rejected,
                "risk_score": score,
            })

        result = {
            "run_id": str(uuid.uuid4()),
            "note_id": note_id,
            "encounter_id": encounter_id,
            "payer": payer,
            "codes": codes,
            "cdi_gaps": gaps,
            "entities": raw.get("entities", []),
            "rejected_codes": rejected,
            "coding_risk_score": score,
            "risk_components": components,
            "risk_tier": risk_tier(score),
            "n_codes": len(codes),
            "n_evidence_verified": sum(1 for c in codes if c["evidence_verified"]),
            "total_reimbursement": sum(c["avg_reimbursement"] for c in codes),
            "model_endpoint": MODEL_ENDPOINT,
            "prompt_version": PROMPT_VERSION,
            "latency_ms": int((time.time() - t0) * 1000),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "trace_id": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if _tracing_ready:
            active = mlflow.get_active_span()
            result["trace_id"] = active.trace_id if active else None
        root.set_outputs({k: result[k] for k in
                          ("coding_risk_score", "risk_components", "rejected_codes", "latency_ms")})
    return result


def run_for_encounter(w: WorkspaceClient, encounter_id: str, codebook: list = None) -> dict:
    """Convenience path used by the app: load masked note + payer priors, then code it."""
    note = load_note(w, encounter_id)
    if not note:
        raise RuntimeError(f"No masked note found for encounter {encounter_id}")
    book = codebook or load_codebook(w)
    priors = load_denial_priors(w, note.get("payer"))
    return code_note(
        w,
        note_id=note["note_id"],
        encounter_id=note["encounter_id"],
        note_text=note["masked_text"],
        codebook=book,
        denial_priors=priors,
        payer=note.get("payer"),
    )


# ── Persistence (parameterised inserts) ─────────────────────────────────────
def _p(name, value, typ="STRING"):
    return StatementParameterListItem(
        name=name,
        value=None if value is None else str(value),
        type=typ,
    )


def persist_run(w: WorkspaceClient, result: dict, triggered_by: str = "app") -> None:
    """Write the run header + one row per grounded suggestion. Fully parameterised."""
    _exec(w, f"""
        INSERT INTO {SCHEMA_CDI}.agent_runs (
            run_id, note_id, encounter_id, coding_risk_score, risk_components, cdi_gaps,
            entities, rejected_codes, n_codes, n_evidence_verified, model_endpoint,
            prompt_version, latency_ms, input_tokens, output_tokens, trace_id,
            triggered_by, created_at
        ) VALUES (
            :run_id, :note_id, :encounter_id, :score, :components, :gaps,
            :entities, :rejected, :n_codes, :n_verified, :endpoint,
            :prompt_version, :latency, :in_tok, :out_tok, :trace_id,
            :triggered_by, CAST(:created_at AS TIMESTAMP)
        )
    """, [
        _p("run_id", result["run_id"]),
        _p("note_id", result["note_id"]),
        _p("encounter_id", result["encounter_id"]),
        _p("score", result["coding_risk_score"], "DOUBLE"),
        _p("components", json.dumps(result["risk_components"])),
        _p("gaps", json.dumps(result["cdi_gaps"])),
        _p("entities", json.dumps(result["entities"])),
        _p("rejected", json.dumps(result["rejected_codes"])),
        _p("n_codes", result["n_codes"], "INT"),
        _p("n_verified", result["n_evidence_verified"], "INT"),
        _p("endpoint", result["model_endpoint"]),
        _p("prompt_version", result["prompt_version"]),
        _p("latency", result["latency_ms"], "BIGINT"),
        _p("in_tok", result.get("input_tokens"), "BIGINT"),
        _p("out_tok", result.get("output_tokens"), "BIGINT"),
        _p("trace_id", result.get("trace_id")),
        _p("triggered_by", triggered_by),
        _p("created_at", result["created_at"]),
    ])

    for c in result["codes"]:
        _exec(w, f"""
            INSERT INTO {SCHEMA_CDI}.agent_code_suggestions (
                suggestion_id, run_id, note_id, encounter_id, code, code_system,
                code_description, avg_reimbursement, confidence, principal, rationale,
                evidence_quote, evidence_start, evidence_end, evidence_verified,
                payer, denial_rate, top_denial_reason, created_at
            ) VALUES (
                :sid, :run_id, :note_id, :encounter_id, :code, :system,
                :descr, :reimb, :conf, :principal, :rationale,
                :quote, :e_start, :e_end, :verified,
                :payer, :denial_rate, :denial_reason, CAST(:created_at AS TIMESTAMP)
            )
        """, [
            _p("sid", str(uuid.uuid4())),
            _p("run_id", result["run_id"]),
            _p("note_id", result["note_id"]),
            _p("encounter_id", result["encounter_id"]),
            _p("code", c["code"]),
            _p("system", c["code_system"]),
            _p("descr", c["code_description"]),
            _p("reimb", c["avg_reimbursement"], "BIGINT"),
            _p("conf", c["confidence"], "DOUBLE"),
            _p("principal", str(c["principal"]).lower(), "BOOLEAN"),
            _p("rationale", c["rationale"]),
            _p("quote", str(c["evidence_quote"])[:2000]),
            _p("e_start", c["evidence_start"], "INT"),
            _p("e_end", c["evidence_end"], "INT"),
            _p("verified", str(c["evidence_verified"]).lower(), "BOOLEAN"),
            _p("payer", result.get("payer")),
            _p("denial_rate", c.get("denial_rate"), "DOUBLE"),
            _p("denial_reason", c.get("top_denial_reason")),
            _p("created_at", result["created_at"]),
        ])


def persist_decision(w: WorkspaceClient, result: dict, code: dict, decision: str,
                     coder_id: str, override_code: str = None,
                     override_reason: str = None, cdi_query_text: str = None) -> None:
    """Record a coder accept / override / reject against an agent suggestion."""
    _exec(w, f"""
        INSERT INTO {SCHEMA_CDI}.coder_decisions (
            decision_id, encounter_id, note_id, ai_suggested_code, ai_code_system,
            ai_code_description, ai_confidence_score, ai_confidence_tier, ai_risk_flags,
            ai_evidence_snippet, ai_avg_reimbursement, coder_decision, override_code,
            override_code_description, override_reason, cdi_query_text, coder_id,
            decision_timestamp, override_delta
        ) VALUES (
            :did, :encounter_id, :note_id, :code, :system,
            :descr, CAST(:conf AS DECIMAL(10,4)), :tier, :flags,
            :snippet, :reimb, :decision, :ovr_code,
            :ovr_descr, :ovr_reason, :cdi_text, :coder_id,
            CURRENT_TIMESTAMP(), :delta
        )
    """, [
        _p("did", str(uuid.uuid4())[:12]),
        _p("encounter_id", result["encounter_id"]),
        _p("note_id", result.get("note_id")),
        _p("code", code["code"]),
        _p("system", code["code_system"]),
        _p("descr", code["code_description"]),
        _p("conf", code["confidence"], "DOUBLE"),
        _p("tier", risk_tier(result["coding_risk_score"])),
        _p("flags", "evidence_unverified" if not code["evidence_verified"] else ""),
        _p("snippet", str(code["evidence_quote"])[:500]),
        _p("reimb", code["avg_reimbursement"], "BIGINT"),
        _p("decision", decision),
        _p("ovr_code", override_code),
        _p("ovr_descr", None),
        _p("ovr_reason", override_reason),
        _p("cdi_text", cdi_query_text),
        _p("coder_id", coder_id),
        _p("delta", "" if not override_code else f"{code['code']}->{override_code}"),
    ])
