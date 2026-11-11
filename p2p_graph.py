import operator
import sqlite3
from typing import Annotated, TypedDict, Optional, List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from database_and_rag import DB_PATH, init_erp_db, init_policy_vector_store

# Initialize DB and Vector Store
init_erp_db()
vector_store = init_policy_vector_store()
llm = ChatOllama(model="llama3.2", temperature=0)

# ==========================================
# 1. Pydantic Schemas for Structured State
# ==========================================
class ExtractedInvoice(BaseModel):
    invoice_id: str = Field(description="Invoice number, e.g., INV-8841")
    po_number: str = Field(description="Purchase Order reference, e.g., PO-9001")
    vendor_name: str = Field(description="Vendor name")
    item_code: str = Field(description="Item SKU or code")
    billed_qty: int = Field(description="Quantity billed by vendor")
    billed_unit_price: float = Field(description="Unit price billed by vendor")
    tax_rate_pct: float = Field(description="Tax percentage billed")

class PolicyEvaluation(BaseModel):
    applied_policy_id: str = Field(description="Policy ID applied, e.g., POL-101, POL-102, or POL-103")
    routing_decision: str = Field(description="Strictly 'AUTO_APPROVED' or 'ESCALATED_HITL'")
    recommended_credit_memo_usd: float = Field(description="Dollar amount to dispute/deduct, 0.0 if approved")
    compliance_rationale: str = Field(description="Concise explanation of why the policy applies")

# ==========================================
# 2. Multi-Type LangGraph State Definition
# ==========================================
class P2PGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    raw_invoice_text: str
    invoice_data: Optional[dict]
    match_metrics: Optional[dict]
    policy_decision: Optional[dict]
    human_override_action: Optional[str]  # "APPROVE_AS_IS" or "SEND_DISPUTE_MEMO"
    final_status: str
    audit_trail: Annotated[List[str], operator.add]

# ==========================================
# 3. Enterprise ERP & RAG Tools
# ==========================================
@tool
def run_erp_three_way_match(po_number: str, billed_qty: int, billed_unit_price: float) -> dict:
    """Query SQLite ERP for Purchase Order and Goods Receipt Note to compute 3-way match variances."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT p.vendor_name, p.item_code, p.ordered_qty, p.po_unit_price,
               g.grn_id, g.received_qty, g.accepted_qty, g.damaged_qty
        FROM purchase_orders p
        LEFT JOIN goods_receipts g ON p.po_number = g.po_number
        WHERE p.po_number = ?
        """,
        (po_number,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {"error": f"PO {po_number} not found in ERP."}

    vendor, item, ordered_qty, po_price, grn_id, rec_qty, acc_qty, dmg_qty = row
    qty_overbilled = billed_qty - acc_qty
    unit_price_diff = round(billed_unit_price - po_price, 2)
    price_variance_pct = round((unit_price_diff / po_price) * 100, 2)

    billed_subtotal = round(billed_qty * billed_unit_price, 2)
    entitled_subtotal = round(acc_qty * po_price, 2)
    total_financial_variance = round(billed_subtotal - entitled_subtotal, 2)

    return {
        "po_number": po_number,
        "grn_id": grn_id,
        "po_ordered_qty": ordered_qty,
        "grn_accepted_qty": acc_qty,
        "grn_damaged_qty": dmg_qty,
        "billed_qty": billed_qty,
        "qty_overbilled": qty_overbilled,
        "po_unit_price": po_price,
        "billed_unit_price": billed_unit_price,
        "price_variance_pct": price_variance_pct,
        "billed_subtotal": billed_subtotal,
        "entitled_subtotal": entitled_subtotal,
        "total_financial_variance": total_financial_variance,
    }

erp_tools = [run_erp_three_way_match]
reconciler_llm = llm.bind_tools(erp_tools)

# ==========================================
# 4. Agent Nodes
# ==========================================
def ingestion_agent(state: P2PGraphState):
    """Agent 1: Extracts structured invoice schema from unstructured text."""
    extractor = llm.with_structured_output(ExtractedInvoice)
    result: ExtractedInvoice = extractor.invoke(
        f"Extract all invoice fields accurately from this text:\n{state['raw_invoice_text']}"
    )
    data = result.model_dump()
    return {
        "invoice_data": data,
        "messages": [
            HumanMessage(
                content=(
                    f"Invoice {data['invoice_id']} extracted for {data['po_number']}: "
                    f"billed_qty={data['billed_qty']}, billed_unit_price=${data['billed_unit_price']}."
                )
            )
        ],
        "audit_trail": [f"INGESTED:{data['invoice_id']}"],
    }

def reconciliation_agent(state: P2PGraphState):
    """Agent 2: Invokes the ERP 3-way match tool and stores deterministic variance metrics."""
    inv = state["invoice_data"]
    sys_prompt = SystemMessage(
        content=(
            "You are the SynOps ERP Reconciliation Agent. "
            f"Call 'run_erp_three_way_match' with po_number='{inv['po_number']}', "
            f"billed_qty={inv['billed_qty']}, and billed_unit_price={inv['billed_unit_price']}. "
            "After receiving the tool output, summarize the variances in one sentence."
        )
    )
    response = reconciler_llm.invoke([sys_prompt] + state["messages"])
    return {"messages": [response], "audit_trail": ["RECONCILIATION_STEP"]}

def policy_compliance_agent(state: P2PGraphState):
    """Agent 3: Retrieves SLA policies from ChromaDB and issues a structured compliance ruling."""
    # Extract deterministic metrics directly via helper call to ensure state accuracy
    inv = state["invoice_data"]
    metrics = run_erp_three_way_match.invoke({
        "po_number": inv["po_number"],
        "billed_qty": inv["billed_qty"],
        "billed_unit_price": inv["billed_unit_price"],
    })

    # Query local ChromaDB RAG for relevant procurement policies
    query = (
        f"qty_overbilled={metrics.get('qty_overbilled')} "
        f"price_variance_pct={metrics.get('price_variance_pct')}% "
        f"total_financial_variance=${metrics.get('total_financial_variance')}"
    )
    docs = vector_store.similarity_search(query, k=3)
    policy_context = "\n".join([d.page_content for d in docs])

    evaluator = llm.with_structured_output(PolicyEvaluation)
    prompt = (
        f"3-Way Match Metrics:\n{metrics}\n\n"
        f"Procurement SLA Policies:\n{policy_context}\n\n"
        "Determine the exact policy_id, routing_decision ('AUTO_APPROVED' if qty_overbilled <= 0 "
        "and price_variance_pct <= 2.0 and total_financial_variance <= 150, otherwise 'ESCALATED_HITL'), "
        "and recommended_credit_memo_usd."
    )
    ruling: PolicyEvaluation = evaluator.invoke(prompt)

    # Enforce deterministic guardrail on top of LLM structured output
    if metrics.get("qty_overbilled", 0) > 0 or metrics.get("price_variance_pct", 0) > 2.0:
        ruling.routing_decision = "ESCALATED_HITL"
        ruling.applied_policy_id = "POL-103"
        ruling.recommended_credit_memo_usd = max(0.0, metrics.get("total_financial_variance", 0.0))
    else:
        ruling.routing_decision = "AUTO_APPROVED"
        ruling.recommended_credit_memo_usd = 0.0

    ruling_dict = ruling.model_dump()
    return {
        "match_metrics": metrics,
        "policy_decision": ruling_dict,
        "messages": [
            HumanMessage(
                content=f"Policy Ruling ({ruling.applied_policy_id}): {ruling.routing_decision} - {ruling.compliance_rationale}"
            )
        ],
        "audit_trail": [f"POLICY_EVAL:{ruling.applied_policy_id}:{ruling.routing_decision}"],
    }

def resolution_executor_agent(state: P2PGraphState):
    """Agent 4: Executes final ERP database logging and generates vendor dispute or payment notice."""
    inv = state["invoice_data"]
    metrics = state["match_metrics"]
    ruling = state["policy_decision"]
    human_action = state.get("human_override_action")

    if ruling["routing_decision"] == "AUTO_APPROVED":
        final_status = "RELEASED_FOR_PAYMENT"
    elif human_action == "APPROVE_AS_IS":
        final_status = "HUMAN_OVERRIDE_APPROVED"
    else:
        final_status = "DISPUTE_MEMO_DISPATCHED"

    sys_prompt = SystemMessage(
        content=(
            "You are the SynOps Resolution Executor. Write a concise 2-sentence operational resolution notice "
            f"for Invoice {inv['invoice_id']} (PO {inv['po_number']}) with final status '{final_status}'. "
            f"Reference the variance of ${metrics['total_financial_variance']} and policy {ruling['applied_policy_id']}."
        )
    )
    response = llm.invoke([sys_prompt] + state["messages"])

    # Persist to SQLite Audit Log
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO invoice_audit_log VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            inv["invoice_id"],
            inv["po_number"],
            metrics["billed_subtotal"],
            metrics["total_financial_variance"],
            ruling["applied_policy_id"],
            final_status,
            response.content,
        ),
    )
    conn.commit()
    conn.close()

    return {
        "messages": [response],
        "final_status": final_status,
        "audit_trail": [f"EXECUTED:{final_status}"],
    }

# ==========================================
# 5. Routing & Graph Compilation
# ==========================================
def route_after_reconciler(state: P2PGraphState):
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "erp_tools"
    return "policy_compliance"

def build_p2p_graph(enable_hitl_interrupt: bool = True):
    builder = StateGraph(P2PGraphState)

    builder.add_node("ingestion", ingestion_agent)
    builder.add_node("reconciler", reconciliation_agent)
    builder.add_node("erp_tools", ToolNode(erp_tools))
    builder.add_node("policy_compliance", policy_compliance_agent)
    builder.add_node("resolution_executor", resolution_executor_agent)

    builder.add_edge(START, "ingestion")
    builder.add_edge("ingestion", "reconciler")
    builder.add_conditional_edges(
        "reconciler",
        route_after_reconciler,
        {"erp_tools": "erp_tools", "policy_compliance": "policy_compliance"},
    )
    builder.add_edge("erp_tools", "reconciler")
    builder.add_edge("policy_compliance", "resolution_executor")
    builder.add_edge("resolution_executor", END)

    checkpointer = MemorySaver()
    interrupt_nodes = ["resolution_executor"] if enable_hitl_interrupt else []
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_nodes)