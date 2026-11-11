from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from p2p_graph import build_p2p_graph
from fastapi.responses import RedirectResponse



app = FastAPI(title="SynOps 2.0 Autonomous P2P Engine")
graph = build_p2p_graph(enable_hitl_interrupt=True)
from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    # Automatically redirects anyone hitting http://127.0.0.1:8000 to the Swagger UI
    return RedirectResponse(url="/docs")
class InvoiceSubmission(BaseModel):
    thread_id: str
    raw_invoice_text: str

class HumanReviewDecision(BaseModel):
    action: str  # "APPROVE_AS_IS" or "SEND_DISPUTE_MEMO"

@app.post("/invoices/process")
def process_invoice(payload: InvoiceSubmission):
    config = {"configurable": {"thread_id": payload.thread_id}}
    initial_state = {
        "messages": [],
        "raw_invoice_text": payload.raw_invoice_text,
        "invoice_data": None,
        "match_metrics": None,
        "policy_decision": None,
        "human_override_action": None,
        "final_status": "IN_PROGRESS",
        "audit_trail": [],
    }

    # Run until interrupt_before=["resolution_executor"]
    state = graph.invoke(initial_state, config=config)

    # If policy ruling is AUTO_APPROVED, automatically continue past the breakpoint
    if state["policy_decision"]["routing_decision"] == "AUTO_APPROVED":
        final_state = graph.invoke(None, config=config)
        return {
            "thread_id": payload.thread_id,
            "status": final_state["final_status"],
            "match_metrics": final_state["match_metrics"],
            "policy_decision": final_state["policy_decision"],
            "audit_trail": final_state["audit_trail"],
        }

    # Otherwise, return paused state awaiting Human-in-the-Loop review
    return {
        "thread_id": payload.thread_id,
        "status": "PAUSED_FOR_HITL_REVIEW",
        "match_metrics": state["match_metrics"],
        "policy_decision": state["policy_decision"],
        "audit_trail": state["audit_trail"],
        "next_step": "POST /invoices/{thread_id}/review with action='SEND_DISPUTE_MEMO' or 'APPROVE_AS_IS'",
    }

@app.post("/invoices/{thread_id}/review")
def resume_paused_invoice(thread_id: str, decision: HumanReviewDecision):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot.next:
        raise HTTPException(status_code=400, detail="Workflow is not paused or thread_id not found.")

    # Inject human decision into graph state and resume execution
    graph.update_state(config, {"human_override_action": decision.action})
    final_state = graph.invoke(None, config=config)

    return {
        "thread_id": thread_id,
        "status": final_state["final_status"],
        "resolution_notice": final_state["messages"][-1].content,
        "audit_trail": final_state["audit_trail"],
    }


if __name__ == "__main__":
    print("=== TEST 1: Clean Touchless 3-Way Match (PO-9001) ===")
    res1 = process_invoice(
        InvoiceSubmission(
            thread_id="tx-101",
            raw_invoice_text=(
                "INVOICE #INV-1001 | Vendor: Apex Industrial Components | PO: PO-9001 | "
                "SKU: VALVE-HX | Qty Billed: 100 | Unit Price: $50.00 | Tax: 18%"
            ),
        )
    )
    print(f"Status: {res1['status']} | Audit Trail: {res1['audit_trail']}\n")

    print("=== TEST 2: Damaged Goods & Price Overbilling Escalation (PO-9002) ===")
    res2 = process_invoice(
        InvoiceSubmission(
            thread_id="tx-102",
            raw_invoice_text=(
                "INVOICE #INV-8841 | Vendor: Nova Logistics Supply | PO: PO-9002 | "
                "SKU: SENSOR-T2 | Qty Billed: 200 | Unit Price: $28.00 | Tax: 18%"
            ),
        )
    )
    print(f"Paused Status: {res2['status']}")
    print(f"Detected Variance: {res2['match_metrics']}")
    print(f"Policy Ruling: {res2['policy_decision']}\n")

    print("=== RESUMING TEST 2 WITH HUMAN DECISION: SEND_DISPUTE_MEMO ===")
    res2_final = resume_paused_invoice("tx-102", HumanReviewDecision(action="SEND_DISPUTE_MEMO"))
    print(f"Final Status: {res2_final['status']}")
    print(f"Resolution Notice:\n{res2_final['resolution_notice']}")
    print(f"Complete Audit Trail: {res2_final['audit_trail']}")