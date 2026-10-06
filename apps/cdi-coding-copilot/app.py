import os, json, uuid
from datetime import datetime, timezone

import dash
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import pandas as pd
from databricks.sdk import WorkspaceClient

import agent  # live CDI coding agent (grounded codes + risk score + CDI gaps)

# ── Config ──────────────────────────────────────────────────────────────────
CATALOG   = "`external-ai-build-day`"
SCHEMA_CC = f"{CATALOG}.clinical_coding"
SCHEMA_CDI = f"{CATALOG}.cdi_copilot"
WAREHOUSE = os.environ.get("DATABRICKS_WAREHOUSE_ID", "22998c886e21ee6c")

# SDK WorkspaceClient auto-authenticates as the App service principal
w = WorkspaceClient()

# ── Lakebase Postgres (OLTP audit trail for coder decisions) ────────────────
import psycopg2, psycopg2.extras

def _get_pg_conn():
    """Return a Lakebase Postgres connection using auto-injected env vars."""
    return psycopg2.connect(
        host=os.environ.get("PGHOST", ""),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "databricks_postgres"),
        user=os.environ.get("PGUSER", ""),
        sslmode=os.environ.get("PGSSLMODE", "require"),
    )

def _init_lakebase_schema():
    """Create the coder_decisions table in Lakebase if it doesn't exist."""
    try:
        conn = _get_pg_conn()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS cdi_data")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cdi_data.coder_decisions (
                    decision_id    TEXT PRIMARY KEY,
                    encounter_id   TEXT NOT NULL,
                    ai_suggested   TEXT,
                    ai_code_system TEXT,
                    ai_description TEXT,
                    ai_confidence  DECIMAL(5,1),
                    ai_tier        TEXT,
                    ai_risk_flags  TEXT,
                    ai_evidence    TEXT,
                    ai_reimburse   BIGINT,
                    coder_decision TEXT NOT NULL,
                    committed_code TEXT,
                    override_reason TEXT,
                    cdi_query_text TEXT,
                    coder_id       TEXT,
                    decided_at     TIMESTAMPTZ DEFAULT NOW(),
                    override_delta JSONB
                )
            """)
        conn.close()
        print("[Lakebase] Schema cdi_data.coder_decisions ready")
    except Exception as e:
        print(f"[Lakebase] Schema init skipped (non-fatal): {e}")

def persist_to_lakebase(row: dict) -> None:
    """Write a single coder decision to Lakebase Postgres."""
    try:
        conn = _get_pg_conn()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO cdi_data.coder_decisions
                    (decision_id, encounter_id, ai_suggested, ai_code_system,
                     ai_description, ai_confidence, ai_tier, ai_risk_flags,
                     ai_evidence, ai_reimburse, coder_decision, committed_code,
                     override_reason, cdi_query_text, coder_id, decided_at,
                     override_delta)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (decision_id) DO NOTHING
            """, (
                row["decision_id"], row["encounter_id"],
                row["ai_suggested"], row["ai_code_system"],
                row["ai_description"], row["ai_confidence"],
                row["ai_tier"], row["ai_risk_flags"],
                row["ai_evidence"][:500], row["ai_reimburse"],
                row["coder_decision"], row["committed_code"],
                row["override_reason"], row["cdi_query_text"],
                row["coder_id"], row["decided_at"],
                psycopg2.extras.Json(row["override_delta"]),
            ))
        conn.close()
    except Exception as e:
        print(f"[Lakebase] Persist skipped (non-fatal): {e}")

# Init Lakebase schema at startup (non-blocking)
_init_lakebase_schema()

def run_query(query: str) -> pd.DataFrame:
    """Execute SQL via Statement Execution API (App SP auth)."""
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE,
        statement=query,
        wait_timeout="50s",
    )
    if resp.status.state.value != "SUCCEEDED":
        err = resp.status.error if resp.status.error else "Unknown"
        raise Exception(f"Query failed: {err}")
    cols = [c.name for c in resp.manifest.schema.columns]
    data = resp.result.data_array if resp.result and resp.result.data_array else []
    return pd.DataFrame(data, columns=cols)

def run_stmt(stmt: str) -> None:
    """Execute a write statement via Statement Execution API."""
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE,
        statement=stmt,
        wait_timeout="50s",
    )
    if resp.status.state.value != "SUCCEEDED":
        err = resp.status.error if resp.status.error else "Unknown"
        raise Exception(f"Write failed: {err}")

# ── Data loaders (cached at startup) ────────────────────────────────────────
def load_note_queue():
    return run_query(f"""
        SELECT
            ecr.encounter_id,
            FIRST(s.patient_id_masked)       AS patient_id,
            FIRST(s.service_line)            AS service_line,
            FIRST(s.encounter_type)          AS encounter_type,
            FIRST(s.payer)                   AS payer,
            FIRST(s.signature_timestamp)     AS signed_at,
            FIRST(s.baseline_days_to_final_code) AS days_to_code,
            COUNT(*)                         AS num_codes,
            ROUND(SUM(ecr.avg_reimbursement),0)  AS total_reimbursement,
            MAX(ecr.confidence_score)        AS max_confidence,
            CASE
              WHEN MIN(ecr.confidence_tier) = 'LOW' THEN 'HIGH RISK'
              WHEN MIN(ecr.confidence_tier) = 'MEDIUM' THEN 'MEDIUM RISK'
              ELSE 'LOW RISK'
            END AS risk_level
        FROM {SCHEMA_CC}.encounter_coding_review ecr
        JOIN {SCHEMA_CDI}.silver_notes_masked s
          ON ecr.encounter_id = s.encounter_id
        GROUP BY ecr.encounter_id
        ORDER BY
            CASE WHEN MIN(ecr.confidence_tier)='LOW' THEN 1
                 WHEN MIN(ecr.confidence_tier)='MEDIUM' THEN 2 ELSE 3 END,
            SUM(ecr.avg_reimbursement) DESC
    """)

def load_encounter_detail(enc_id: str):
    return run_query(f"""
        SELECT ecr.*, s.masked_text, s.payer,
               dp.denial_rate, dp.top_denial_reason
        FROM {SCHEMA_CC}.encounter_coding_review ecr
        JOIN {SCHEMA_CDI}.silver_notes_masked s
          ON ecr.encounter_id = s.encounter_id
        LEFT JOIN {SCHEMA_CDI}.gold_code_denial_priors dp
          ON ecr.code = dp.code AND s.payer = dp.payer
        WHERE ecr.encounter_id = '{enc_id}'
        ORDER BY ecr.avg_reimbursement DESC
    """)

def load_kpis():
    return run_query(f"""
        SELECT
            COUNT(DISTINCT e.encounter_id) AS total_encounters,
            ROUND(AVG(e.days_to_final_code),1) AS avg_days_to_code,
            SUM(CASE WHEN cd.status='denied' THEN 1 ELSE 0 END) AS total_denials,
            COUNT(cd.claim_id) AS total_claims,
            ROUND(SUM(CASE WHEN cd.status='denied' THEN 1 ELSE 0 END)*100.0
                / NULLIF(COUNT(cd.claim_id),0),1) AS denial_rate_pct,
            ROUND(SUM(CASE WHEN cd.status='denied' THEN cd.billed_amount ELSE 0 END),0)
                AS total_denied_dollars
        FROM {SCHEMA_CC}.encounters e
        LEFT JOIN {SCHEMA_CC}.claims_denials cd ON e.encounter_id = cd.encounter_id
    """)

def load_codebook():
    return run_query(f"SELECT * FROM {SCHEMA_CC}.icd10_cpt_ref ORDER BY code")

def load_cdi_gap(enc_id: str):
    return run_query(f"""
        SELECT gold_cdi_gap FROM {SCHEMA_CDI}.note_labels
        WHERE encounter_id = '{enc_id}'
    """)

def load_decisions():
    return run_query(f"SELECT * FROM {SCHEMA_CDI}.coder_decisions ORDER BY decision_timestamp DESC LIMIT 200")

# ── Dash app ────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="TeamOne Hospital Scoring App",
)
server = app.server

# ── Colour helpers ──────────────────────────────────────────────────────────
RISK_COLORS = {"HIGH RISK": "danger", "MEDIUM RISK": "warning", "LOW RISK": "success"}
TIER_COLORS = {"LOW": "danger", "MEDIUM": "warning", "HIGH": "success"}

def kpi_card(title, value, sub="", color="primary"):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted mb-1", style={"fontSize": "0.75rem"}),
            html.H3(value, className=f"text-{color} mb-0", style={"fontWeight": "700"}),
            html.Small(sub, className="text-muted") if sub else None,
        ], className="py-2 px-3")
    ], className="shadow-sm h-100")

# ── Layout ──────────────────────────────────────────────────────────────────
app.layout = dbc.Container([
    dcc.Store(id="selected-encounter", data=None),
    dcc.Store(id="decisions-store", data=[]),
    dcc.Store(id="refresh-trigger", data=0),
    dcc.Store(id="agent-result", data=None),
    dcc.Store(id="codebook-cache", data=None),

    # Header
    dbc.Navbar(
        dbc.Container([
            html.Span("\U0001F3E5", style={"fontSize": "1.5rem", "marginRight": "10px"}),
            dbc.NavbarBrand("TeamOne Hospital Scoring App",
                           className="fw-bold", style={"fontSize": "1.2rem"}),
            html.Span("AI-Assisted Code Review",
                      className="text-light ms-3", style={"fontSize": "0.85rem", "opacity": 0.8}),
        ], fluid=True),
        color="dark", dark=True, className="mb-3"
    ),

    # KPI bar
    dbc.Row(id="kpi-bar", className="mb-3 g-2"),

    # Main 3-panel layout
    dbc.Row([
        # Left: Note queue
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H6("\U0001F4CB Note Queue", className="mb-0 fw-bold"),
                    html.Small("Sorted by coding risk", className="text-muted")
                ]),
                dbc.CardBody(id="note-queue-body", style={
                    "maxHeight": "70vh", "overflowY": "auto", "padding": "0"
                })
            ], className="shadow-sm")
        ], width=3),

        # Center: Code review
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("\U0001F50D Code Review", className="mb-0 fw-bold")),
                dbc.CardBody([
                    dbc.Button("Run AI coding agent on this note",
                               id="run-agent-btn", color="primary", size="sm",
                               className="fw-bold w-100 mb-2"),
                    dbc.Spinner(html.Div(id="agent-output"), size="sm", color="primary"),
                ], className="pb-0"),
                dbc.CardBody(id="code-review-body", style={"maxHeight": "45vh", "overflowY": "auto"})
            ], className="shadow-sm")
        ], width=6),

        # Right: CDI Query + Commit
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("\u26A0\uFE0F CDI & Commit", className="mb-0 fw-bold")),
                dbc.CardBody(id="cdi-panel-body", style={"maxHeight": "70vh", "overflowY": "auto"})
            ], className="shadow-sm")
        ], width=3),
    ]),

    # Toast for commit feedback
    dbc.Toast(id="commit-toast", header="Decision Recorded", is_open=False,
              duration=4000, icon="success",
              style={"position": "fixed", "top": 10, "right": 10, "zIndex": 9999}),

], fluid=True, style={"backgroundColor": "#f8f9fa", "minHeight": "100vh"})

# ── Callbacks ───────────────────────────────────────────────────────────────

@callback(Output("kpi-bar", "children"), Input("refresh-trigger", "data"))
def render_kpis(_):
    try:
        kdf = load_kpis()
        k = kdf.iloc[0]
        return [
            dbc.Col(kpi_card("Avg Days to Code",
                             f"{k['avg_days_to_code']}d",
                             "vs <1 day target", "danger"), width=3),
            dbc.Col(kpi_card("Denial Rate",
                             f"{k['denial_rate_pct']}%",
                             f"{int(float(k['total_denials']))}/{int(float(k['total_claims']))} claims", "warning"), width=3),
            dbc.Col(kpi_card("$ Denied (Revenue at Risk)",
                             f"${int(float(k['total_denied_dollars'])):,}",
                             "recoverable with CDI", "danger"), width=3),
            dbc.Col(kpi_card("Encounters to Review",
                             str(int(float(k['total_encounters']))),
                             "awaiting coder action", "info"), width=3),
        ]
    except Exception as e:
        return [dbc.Col(dbc.Alert(f"KPI load error: {e}", color="danger"))]


@callback(Output("note-queue-body", "children"), Input("refresh-trigger", "data"))
def render_note_queue(_):
    try:
        nq = load_note_queue()
        items = []
        for _, r in nq.iterrows():
            risk = r["risk_level"]
            color = RISK_COLORS.get(risk, "secondary")
            items.append(
                dbc.ListGroupItem([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                dbc.Badge(risk, color=color, className="me-2",
                                          style={"fontSize": "0.65rem"}),
                                html.Strong(r["encounter_id"], style={"fontSize": "0.85rem"}),
                            ]),
                            html.Div([
                                html.Small(f"{r['service_line']} | {r['encounter_type']}",
                                           className="text-muted"),
                            ]),
                            html.Div([
                                html.Small(f"{r['payer']} | {int(float(r['num_codes']))} codes",
                                           className="text-muted"),
                            ]),
                        ], width=8),
                        dbc.Col([
                            html.Div(f"${int(float(r['total_reimbursement'])):,}",
                                     className="fw-bold text-end",
                                     style={"fontSize": "0.85rem"}),
                            html.Small(f"{r['days_to_code']}d wait",
                                       className="text-muted d-block text-end"),
                        ], width=4),
                    ])
                ], id={"type": "queue-item", "index": r["encounter_id"]},
                   action=True, style={"cursor": "pointer", "padding": "8px 12px"})
            )
        return dbc.ListGroup(items, flush=True)
    except Exception as e:
        return dbc.Alert(f"Queue load error: {e}", color="danger")


@callback(
    Output("selected-encounter", "data"),
    Input({"type": "queue-item", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_encounter(clicks):
    if not ctx.triggered_id:
        return no_update
    return ctx.triggered_id["index"]


@callback(
    Output("code-review-body", "children"),
    Input("selected-encounter", "data"),
)
def render_code_review(enc_id):
    if not enc_id:
        return html.Div([
            html.I(className="bi bi-arrow-left-circle", style={"fontSize": "2rem"}),
            html.P("Select an encounter from the queue to begin review.",
                   className="text-muted mt-2")
        ], className="text-center py-5")

    try:
        df = load_encounter_detail(enc_id)
        if df.empty:
            return dbc.Alert("No coding data found.", color="warning")

        row0 = df.iloc[0]

        # Clinical note display
        note_section = dbc.Card([
            dbc.CardHeader(html.Small("Masked Clinical Note", className="fw-bold")),
            dbc.CardBody(
                html.Pre(row0["masked_text"][:2000],
                         style={"whiteSpace": "pre-wrap", "fontSize": "0.78rem",
                                "maxHeight": "200px", "overflowY": "auto",
                                "backgroundColor": "#f8f9fa", "padding": "10px",
                                "borderRadius": "4px"})
            )
        ], className="mb-3", style={"border": "1px solid #dee2e6"})

        # Code suggestions table
        code_rows = []
        for i, r in df.iterrows():
            tier = str(r.get("confidence_tier", "MEDIUM"))
            tier_color = TIER_COLORS.get(tier, "secondary")
            denial_rate = r.get("denial_rate")
            denial_str = f"{float(denial_rate)*100:.0f}%" if denial_rate and pd.notna(denial_rate) else "N/A"
            denial_reason = r.get("top_denial_reason", "") or ""

            snippet = str(r.get("evidence_snippet", "") or "")[:200]
            risk_flags = str(r.get("risk_flags", "") or "")

            code_rows.append(
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Code(r["code"], style={"fontSize": "1rem", "fontWeight": "bold"}),
                                    html.Span(f" ({r['code_system']})", className="text-muted ms-1",
                                              style={"fontSize": "0.75rem"}),
                                    dbc.Badge(f"Confidence: {tier}", color=tier_color,
                                              className="ms-2", style={"fontSize": "0.65rem"}),
                                ]),
                                html.Div(r.get("code_description", ""),
                                         style={"fontSize": "0.8rem"}, className="mt-1"),
                            ], width=6),
                            dbc.Col([
                                html.Div(f"${int(float(r['avg_reimbursement'])):,}",
                                         className="fw-bold", style={"fontSize": "0.9rem"}),
                                html.Small(f"Denial risk: {denial_str}",
                                           className="text-danger" if denial_str != "N/A" and float(denial_rate or 0) > 0.2 else "text-muted"),
                                html.Br(),
                                html.Small(denial_reason.replace("_"," ").title() if denial_reason else "",
                                           className="text-muted"),
                            ], width=3, className="text-end"),
                            dbc.Col([
                                dbc.ButtonGroup([
                                    dbc.Button("\u2713", color="success", size="sm",
                                               id={"type": "accept-btn", "index": f"{enc_id}|{r['code']}"},
                                               title="Accept"),
                                    dbc.Button("\u270E", color="warning", size="sm",
                                               id={"type": "override-btn", "index": f"{enc_id}|{r['code']}"},
                                               title="Override"),
                                    dbc.Button("\u2717", color="danger", size="sm",
                                               id={"type": "reject-btn", "index": f"{enc_id}|{r['code']}"},
                                               title="Reject"),
                                ], size="sm")
                            ], width=3, className="text-end"),
                        ]),
                        # Risk flags
                        html.Div([
                            dbc.Badge(flag.strip(), color="dark", className="me-1",
                                      style={"fontSize": "0.6rem"})
                            for flag in risk_flags.split("|") if flag.strip()
                        ], className="mt-1") if risk_flags else None,
                        # Evidence snippet
                        html.Div([
                            html.Small("Evidence: ", className="fw-bold text-muted"),
                            html.Small(snippet, className="text-muted fst-italic"),
                        ], className="mt-1",
                           style={"backgroundColor": "#fff3cd", "padding": "4px 8px",
                                  "borderRadius": "4px", "fontSize": "0.75rem"}) if snippet else None,
                    ], className="py-2")
                ], className="mb-2", style={"border": "1px solid #dee2e6"})
            )

        return html.Div([
            html.Div([
                html.H5(enc_id, className="mb-0 fw-bold"),
                html.Small(f"{row0.get('service_line','')} | {row0.get('encounter_type','')} | {row0.get('payer','')}",
                           className="text-muted"),
            ], className="mb-3"),
            note_section,
            html.H6(f"Suggested Codes ({len(df)})", className="fw-bold mb-2"),
            html.Div(code_rows),
        ])
    except Exception as e:
        return dbc.Alert(f"Error loading encounter: {e}", color="danger")


@callback(
    Output("cdi-panel-body", "children"),
    Input("selected-encounter", "data"),
)
def render_cdi_panel(enc_id):
    if not enc_id:
        return html.P("Select an encounter to see CDI details.", className="text-muted")

    try:
        gap_df = load_cdi_gap(enc_id)
        cdi_text = ""
        if not gap_df.empty and gap_df.iloc[0]["gold_cdi_gap"]:
            cdi_text = str(gap_df.iloc[0]["gold_cdi_gap"])

        detail_df = load_encounter_detail(enc_id)
        risk_flags_all = ""
        if not detail_df.empty:
            risk_flags_all = " | ".join(
                set(f for r in detail_df["risk_flags"].dropna()
                    for f in str(r).split("|") if f.strip())
            )

        return html.Div([
            # CDI gap alert
            dbc.Alert([
                html.H6("CDI Documentation Gap", className="alert-heading fw-bold"),
                html.P(cdi_text if cdi_text else "No documentation gaps identified for this encounter.",
                       style={"fontSize": "0.85rem"}),
            ], color="warning" if cdi_text else "success", className="mb-3"),

            # Risk flags summary
            html.Div([
                html.H6("Risk Flags", className="fw-bold mb-2"),
                html.Div([
                    dbc.Badge(f.strip(), color="dark", className="me-1 mb-1")
                    for f in risk_flags_all.split("|") if f.strip()
                ]) if risk_flags_all else html.Small("None", className="text-muted"),
            ], className="mb-3"),

            html.Hr(),

            # CDI Query composer
            html.H6("Compose CDI Query to Physician", className="fw-bold mb-2"),
            dbc.Textarea(
                id="cdi-query-text",
                value=cdi_text if cdi_text else "",
                placeholder="Type a CDI query for the attending physician...",
                style={"fontSize": "0.8rem", "minHeight": "100px"},
                className="mb-2",
            ),

            html.Hr(),

            # Override section
            html.H6("Override Code (optional)", className="fw-bold mb-2"),
            dbc.Input(id="override-code", placeholder="e.g. R65.21",
                      size="sm", className="mb-1"),
            dbc.Input(id="override-reason", placeholder="Override reason...",
                      size="sm", className="mb-3"),

            # Commit button
            dbc.Button(
                "\u2705 Commit All Decisions",
                id="commit-btn", color="primary", size="lg",
                className="w-100 fw-bold",
            ),

            html.Hr(),

            # Recent decisions
            html.H6("Recent Decisions", className="fw-bold mb-2"),
            html.Div(id="recent-decisions"),
        ])
    except Exception as e:
        return dbc.Alert(f"CDI panel error: {e}", color="danger")


@callback(
    [Output("commit-toast", "is_open"), Output("commit-toast", "children"),
     Output("recent-decisions", "children")],
    Input("commit-btn", "n_clicks"),
    [State("selected-encounter", "data"),
     State("cdi-query-text", "value"),
     State("override-code", "value"),
     State("override-reason", "value")],
    prevent_initial_call=True,
)
def commit_decisions(n, enc_id, cdi_text, override_code, override_reason):
    if not enc_id:
        return False, "", no_update

    try:
        detail_df = load_encounter_detail(enc_id)
        decision_type = "override" if override_code else "accept"
        ts = datetime.now(timezone.utc).isoformat()
        coder_id = os.environ.get("DATABRICKS_USER", "demo-coder-01")

        for _, r in detail_df.iterrows():
            did = str(uuid.uuid4())[:12]
            ai_code = str(r['code'])
            final_code = override_code if override_code else ai_code
            esc = lambda s: str(s).replace("'", "''") if s else ""

            # Build override delta JSON — captures AI-suggested vs committed
            delta = json.dumps({
                "ai_suggested": ai_code,
                "committed": final_code,
                "changed": ai_code != final_code,
                "action": decision_type,
                "cdi_query_raised": bool(cdi_text),
            })

            # 1) Write to UC table (for Genie Space analytics)
            run_stmt(f"""
                INSERT INTO {SCHEMA_CDI}.coder_decisions VALUES (
                    '{did}', '{r['encounter_id']}', NULL,
                    '{esc(ai_code)}', '{esc(r['code_system'])}',
                    '{esc(r.get('code_description',''))}',
                    {r.get('confidence_score',0)}, '{esc(r.get('confidence_tier',''))}',
                    '{esc(r.get('risk_flags',''))}',
                    '{esc(str(r.get('evidence_snippet',''))[:500])}',
                    {int(float(r.get('avg_reimbursement',0)))},
                    '{decision_type}',
                    '{esc(final_code) if override_code else ''}',
                    '{esc(r.get('code_description',''))}',
                    '{esc(override_reason) if override_reason else ''}',
                    '{esc(cdi_text) if cdi_text else ''}',
                    '{coder_id}',
                    '{ts}',
                    '{esc(delta)}'
                )
            """)

            # 2) Dual-write to Lakebase Postgres (OLTP audit trail)
            persist_to_lakebase({
                "decision_id": did,
                "encounter_id": r["encounter_id"],
                "ai_suggested": ai_code,
                "ai_code_system": r.get("code_system", ""),
                "ai_description": r.get("code_description", ""),
                "ai_confidence": float(r.get("confidence_score", 0)),
                "ai_tier": r.get("confidence_tier", ""),
                "ai_risk_flags": r.get("risk_flags", ""),
                "ai_evidence": str(r.get("evidence_snippet", ""))[:500],
                "ai_reimburse": int(float(r.get("avg_reimbursement", 0))),
                "coder_decision": decision_type,
                "committed_code": final_code,
                "override_reason": override_reason or "",
                "cdi_query_text": cdi_text or "",
                "coder_id": coder_id,
                "decided_at": ts,
                "override_delta": json.loads(delta),
            })

        # Load recent decisions
        recent = load_decisions()
        rows = []
        for _, d in recent.head(5).iterrows():
            rows.append(html.Div([
                html.Small(f"{d['encounter_id']} | {d['ai_suggested_code']} | {d['coder_decision']}",
                           className="text-muted"),
            ], className="mb-1"))

        msg = f"Committed {len(detail_df)} code decisions for {enc_id}"
        if cdi_text:
            msg += " + CDI query sent"
        return True, msg, html.Div(rows) if rows else html.Small("None yet")
    except Exception as e:
        return True, f"Error: {e}", no_update


# Accept/Override/Reject individual button callbacks (pattern-matching)
@callback(
    Output("commit-toast", "is_open", allow_duplicate=True),
    Output("commit-toast", "children", allow_duplicate=True),
    Input({"type": "accept-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def handle_accept(clicks):
    if not any(clicks):
        return no_update, no_update
    triggered = ctx.triggered_id
    if triggered:
        parts = triggered["index"].split("|")
        enc_id, code = parts[0], parts[1]
        return True, f"Accepted {code} for {enc_id}"
    return no_update, no_update


# ── Live agent ──────────────────────────────────────────────────────────────
_codebook = None


def get_codebook():
    """Codebook is small and static — fetch once per app process."""
    global _codebook
    if _codebook is None:
        _codebook = agent.load_codebook(w)
    return _codebook


def highlight_note(text, spans):
    """Render the masked note with every verified evidence span highlighted."""
    spans = sorted([s for s in spans if s[0] is not None and s[1] is not None])
    merged, cursor, parts = [], 0, []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    for s, e in merged:
        if s > cursor:
            parts.append(text[cursor:s])
        parts.append(html.Mark(text[s:e], style={"backgroundColor": "#fff3cd",
                                                 "padding": "1px 2px"}))
        cursor = e
    parts.append(text[cursor:])
    return html.Pre(parts, style={"whiteSpace": "pre-wrap", "fontSize": "0.75rem",
                                  "maxHeight": "220px", "overflowY": "auto",
                                  "backgroundColor": "#fdfdfe", "padding": "10px",
                                  "border": "1px solid #dee2e6", "borderRadius": "4px"})


RISK_TIER_COLOR = {"HIGH RISK": "danger", "MEDIUM RISK": "warning", "LOW RISK": "success"}
COMPONENT_LABEL = {
    "low_confidence": "Low model confidence (max 40)",
    "cdi_gaps": "Open CDI documentation gaps (max 25)",
    "payer_denial_prior": "Payer denial history for these codes (max 25)",
    "unverified_evidence": "Evidence not found in note (max 10)",
}


def render_agent_result(res):
    tier = res["risk_tier"]
    color = RISK_TIER_COLOR.get(tier, "secondary")

    header = dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H6("Coding Risk Score", className="text-muted mb-1",
                        style={"fontSize": "0.75rem"}),
                html.Div([
                    html.H2(f"{res['coding_risk_score']:.0f}", className=f"text-{color} d-inline mb-0",
                            style={"fontWeight": "700"}),
                    html.Span("/100", className="text-muted ms-1"),
                    dbc.Badge(tier, color=color, className="ms-2"),
                ]),
            ], width=4),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Small(COMPONENT_LABEL.get(k, k), className="text-muted"),
                        dbc.Progress(value=v, max=40 if k == "low_confidence" else
                                     (10 if k == "unverified_evidence" else 25),
                                     color=color, style={"height": "6px"}, className="mb-1"),
                    ]) for k, v in res["risk_components"].items()
                ])
            ], width=8),
        ]),
        html.Hr(className="my-2"),
        html.Small(
            f"{res['n_codes']} codes · {res['n_evidence_verified']}/{res['n_codes']} evidence verified · "
            f"{len(res['cdi_gaps'])} CDI gaps · {len(res['rejected_codes'])} hallucinated codes blocked · "
            f"${res['total_reimbursement']:,} at stake · {res['latency_ms'] / 1000:.1f}s · "
            f"{res['model_endpoint']} ({res['prompt_version']})",
            className="text-muted"),
    ], className="py-2 px-3"), color="light", className="mb-2")

    note_text = res.get("_note_text", "")
    spans = [(c["evidence_start"], c["evidence_end"]) for c in res["codes"]]
    note_block = html.Div([
        html.Small("Masked note — highlighted text is the agent's cited evidence",
                   className="fw-bold text-muted"),
        highlight_note(note_text, spans),
    ], className="mb-2") if note_text else None

    cards = []
    for c in res["codes"]:
        dr = c.get("denial_rate")
        denial_badge = dbc.Badge(
            f"Denial risk {float(dr) * 100:.0f}% · {str(c.get('top_denial_reason') or '').replace('_', ' ')}",
            color="danger" if dr and float(dr) > 0.2 else "secondary",
            className="ms-1", style={"fontSize": "0.6rem"}) if dr else None
        cards.append(dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Code(c["code"], style={"fontSize": "0.95rem", "fontWeight": "bold"}),
                        html.Span(f" ({c['code_system']})", className="text-muted ms-1",
                                  style={"fontSize": "0.7rem"}),
                        dbc.Badge("PRINCIPAL", color="primary", className="ms-2",
                                  style={"fontSize": "0.6rem"}) if c["principal"] else None,
                        dbc.Badge(f"conf {c['confidence']:.2f}",
                                  color="success" if c["confidence"] >= 0.8 else
                                        ("warning" if c["confidence"] >= 0.6 else "danger"),
                                  className="ms-1", style={"fontSize": "0.6rem"}),
                        dbc.Badge("evidence verified" if c["evidence_verified"]
                                  else "EVIDENCE NOT FOUND",
                                  color="success" if c["evidence_verified"] else "danger",
                                  className="ms-1", style={"fontSize": "0.6rem"}),
                        denial_badge,
                    ]),
                    html.Div(c["code_description"], style={"fontSize": "0.8rem"}, className="mt-1"),
                    html.Div([
                        html.Small("Evidence: ", className="fw-bold text-muted"),
                        html.Small(f"\u201c{c['evidence_quote']}\u201d", className="fst-italic"),
                    ], className="mt-1"),
                    html.Div(html.Small(c["rationale"], className="text-muted"), className="mt-1"),
                ], width=8),
                dbc.Col([
                    html.Div(f"${c['avg_reimbursement']:,}", className="fw-bold text-end"),
                    dbc.ButtonGroup([
                        dbc.Button("Accept", color="success", size="sm",
                                   id={"type": "agent-accept", "index": c["code"]}),
                        dbc.Button("Reject", color="outline-danger", size="sm",
                                   id={"type": "agent-reject", "index": c["code"]}),
                    ], size="sm", className="mt-2"),
                ], width=4, className="text-end"),
            ]),
        ], className="py-2 px-3"), className="mb-2",
            style={"borderLeft": f"4px solid var(--bs-{'success' if c['evidence_verified'] and c['confidence'] >= 0.8 else 'warning'})"}))

    rejected = dbc.Alert(
        [html.Strong("Blocked as not in codebook: "), ", ".join(res["rejected_codes"])],
        color="secondary", className="py-1 px-2", style={"fontSize": "0.75rem"}
    ) if res["rejected_codes"] else None

    gaps = [dbc.Alert([
        html.Strong(g["gap"], style={"fontSize": "0.8rem"}),
        html.Div(html.Small(g["impact"], className="text-muted")),
        html.Div([html.Small("Suggested physician query: ", className="fw-bold"),
                  html.Small(g["physician_query"])], className="mt-1"),
    ], color="warning", className="py-2 px-2") for g in res["cdi_gaps"]]

    return html.Div([
        header,
        note_block,
        html.Div(cards),
        rejected,
        html.Div([html.H6("CDI documentation gaps", className="fw-bold mt-2 mb-2")] + gaps)
        if gaps else None,
    ])


@callback(
    [Output("agent-output", "children"), Output("agent-result", "data")],
    Input("run-agent-btn", "n_clicks"),
    State("selected-encounter", "data"),
    prevent_initial_call=True,
)
def run_agent(n, enc_id):
    if not enc_id:
        return dbc.Alert("Select an encounter from the queue first.", color="info",
                         className="py-1 px-2"), no_update
    try:
        note = agent.load_note(w, enc_id)
        if not note:
            return dbc.Alert(f"No masked note found for {enc_id}.", color="warning"), no_update
        priors = agent.load_denial_priors(w, note.get("payer"))
        res = agent.code_note(
            w,
            note_id=note["note_id"],
            encounter_id=note["encounter_id"],
            note_text=note["masked_text"],
            codebook=get_codebook(),
            denial_priors=priors,
            payer=note.get("payer"),
        )
        try:
            agent.persist_run(w, res, triggered_by="app")
        except Exception as pe:
            print(f"[app] persist_run failed: {pe}")
        res["_note_text"] = note["masked_text"]
        return render_agent_result(res), res
    except Exception as e:
        return dbc.Alert(f"Agent error: {e}", color="danger"), no_update


@callback(
    Output("cdi-query-text", "value", allow_duplicate=True),
    Input("agent-result", "data"),
    prevent_initial_call=True,
)
def prefill_cdi_query(res):
    if not res or not res.get("cdi_gaps"):
        return no_update
    return "\n\n".join(
        f"[{g['gap']}]\n{g['physician_query']}" for g in res["cdi_gaps"] if g.get("physician_query")
    )


@callback(
    [Output("commit-toast", "is_open", allow_duplicate=True),
     Output("commit-toast", "children", allow_duplicate=True)],
    [Input({"type": "agent-accept", "index": dash.ALL}, "n_clicks"),
     Input({"type": "agent-reject", "index": dash.ALL}, "n_clicks")],
    [State("agent-result", "data"), State("cdi-query-text", "value")],
    prevent_initial_call=True,
)
def record_agent_decision(acc, rej, res, cdi_text):
    if not ctx.triggered_id or not res:
        return no_update, no_update
    code_id = ctx.triggered_id["index"]
    decision = "accept" if ctx.triggered_id["type"] == "agent-accept" else "reject"
    code = next((c for c in res["codes"] if c["code"] == code_id), None)
    if not code:
        return no_update, no_update
    try:
        agent.persist_decision(
            w, res, code, decision,
            coder_id=os.environ.get("DATABRICKS_USER", "demo-coder-01"),
            cdi_query_text=cdi_text,
        )
        return True, f"{decision.title()}ed {code_id} for {res['encounter_id']} (persisted)"
    except Exception as e:
        return True, f"Could not record decision: {e}"


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
