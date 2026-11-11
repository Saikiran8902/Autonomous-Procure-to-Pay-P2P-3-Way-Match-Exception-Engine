from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from p2p_graph import build_p2p_graph

eval_llm = ChatOllama(model="llama3.2", temperature=0)

class AgentQualityScorecard(BaseModel):
    technical_depth: int = Field(description="Score 1-10 on precision of financial variance & ERP data usage")
    clarity_of_explanation: int = Field(description="Score 1-10 on clarity of resolution notice and policy citation")
    originality: int = Field(description="Score 1-10 on non-repetitive, actionable resolution phrasing")
    understanding_of_implementation: int = Field(description="Score 1-10 on adherence to 3-way match SLA rules")

TEST_SUITE = [
    {
        "test_id": "eval-1",
        "raw_invoice": (
            "INVOICE #INV-301 | Vendor: Apex Industrial Components | PO: PO-9001 | "
            "SKU: VALVE-HX | Qty Billed: 100 | Unit Price: $50.00 | Tax: 18%"
        ),
        "expected_decision": "AUTO_APPROVED",
        "expected_variance": 0.0,
    },
    {
        "test_id": "eval-2",
        "raw_invoice": (
            "INVOICE #INV-302 | Vendor: Zenith Precision Gears | PO: PO-9003 | "
            "SKU: GEAR-M8 | Qty Billed: 50 | Unit Price: $121.50 | Tax: 18%"
        ),
        "expected_decision": "AUTO_APPROVED",  # 1.25% variance ($75 total) -> within POL-102 tolerance
        "expected_variance": 75.0,
    },
    {
        "test_id": "eval-3",
        "raw_invoice": (
            "INVOICE #INV-303 | Vendor: Nova Logistics Supply | PO: PO-9002 | "
            "SKU: SENSOR-T2 | Qty Billed: 200 | Unit Price: $28.00 | Tax: 18%"
        ),
        "expected_decision": "ESCALATED_HITL",  # 20 damaged units billed + 12% price hike -> POL-103
        "expected_variance": 1100.0,
    },
]

def run_evaluation():
    graph = build_p2p_graph(enable_hitl_interrupt=False)
    judge = eval_llm.with_structured_output(AgentQualityScorecard)

    correct_routings = 0
    correct_variances = 0

    print("Running SynOps P2P Multi-Agent Evaluation Suite...\n")
    for case in TEST_SUITE:
        config = {"configurable": {"thread_id": case["test_id"]}}
        final_state = graph.invoke(
            {
                "messages": [],
                "raw_invoice_text": case["raw_invoice"],
                "invoice_data": None,
                "match_metrics": None,
                "policy_decision": None,
                "human_override_action": "SEND_DISPUTE_MEMO",
                "final_status": "PENDING",
                "audit_trail": [],
            },
            config=config,
        )

        actual_decision = final_state["policy_decision"]["routing_decision"]
        actual_variance = final_state["match_metrics"]["total_financial_variance"]
        resolution_text = final_state["messages"][-1].content

        if actual_decision == case["expected_decision"]:
            correct_routings += 1
        if abs(actual_variance - case["expected_variance"]) < 0.01:
            correct_variances += 1

        scorecard: AgentQualityScorecard = judge.invoke(
            f"Evaluate this autonomous P2P agent output.\n"
            f"Metrics: {final_state['match_metrics']}\n"
            f"Policy Ruling: {final_state['policy_decision']}\n"
            f"Resolution Notice: {resolution_text}"
        )

        print(f"[{case['test_id']}] Decision: {actual_decision} (Expected: {case['expected_decision']}) | Variance: ${actual_variance}")
        print(f"  Judge Scores -> Technical Depth: {scorecard.technical_depth}/10 | Clarity: {scorecard.clarity_of_explanation}/10 | Implementation: {scorecard.understanding_of_implementation}/10\n")

    total = len(TEST_SUITE)
    print("=== BENCHMARK SUMMARY ===")
    print(f"Routing Accuracy: {(correct_routings / total) * 100:.1f}% ({correct_routings}/{total})")
    print(f"3-Way Variance Precision: {(correct_variances / total) * 100:.1f}% ({correct_variances}/{total})")

if __name__ == "__main__":
    run_evaluation()