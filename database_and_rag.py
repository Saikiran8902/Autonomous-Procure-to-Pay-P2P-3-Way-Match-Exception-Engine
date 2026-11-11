import sqlite3
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DB_PATH = "synops_erp.db"

def init_erp_db():
    """Initialize SQLite ERP tables for Purchase Orders, Goods Receipts, and Audit Logs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS purchase_orders (
        po_number TEXT PRIMARY KEY,
        vendor_name TEXT,
        item_code TEXT,
        ordered_qty INTEGER,
        po_unit_price REAL,
        tax_rate_pct REAL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS goods_receipts (
        grn_id TEXT PRIMARY KEY,
        po_number TEXT,
        received_qty INTEGER,
        accepted_qty INTEGER,
        damaged_qty INTEGER,
        FOREIGN KEY (po_number) REFERENCES purchase_orders(po_number)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS invoice_audit_log (
        invoice_id TEXT PRIMARY KEY,
        po_number TEXT,
        billed_total REAL,
        variance_amount REAL,
        policy_applied TEXT,
        final_status TEXT,
        resolution_notes TEXT
    )
    """)

    # Seed Purchase Orders
    cursor.executemany(
        "INSERT OR REPLACE INTO purchase_orders VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("PO-9001", "Apex Industrial Components", "VALVE-HX", 100, 50.00, 18.0),
            ("PO-9002", "Nova Logistics Supply", "SENSOR-T2", 200, 25.00, 18.0),
            ("PO-9003", "Zenith Precision Gears", "GEAR-M8", 50, 120.00, 18.0),
        ],
    )

    # Seed Goods Receipt Notes (GRNs)
    cursor.executemany(
        "INSERT OR REPLACE INTO goods_receipts VALUES (?, ?, ?, ?, ?)",
        [
            ("GRN-501", "PO-9001", 100, 100, 0),   # Full clean delivery
            ("GRN-502", "PO-9002", 200, 180, 20),  # 20 damaged units rejected at dock
            ("GRN-503", "PO-9003", 50, 50, 0),     # Full clean delivery
        ],
    )

    conn.commit()
    conn.close()


def init_policy_vector_store() -> Chroma:
    """Build local ChromaDB vector store using OllamaEmbeddings for Procurement SLA policies."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    policies = [
        Document(
            page_content=(
                "Policy ID: POL-101 (Touchless Auto-Approval). An invoice qualifies for touchless "
                "AUTO_APPROVED status if billed_qty is less than or equal to GRN accepted_qty AND "
                "unit price variance is 0% (exact match)."
            ),
            metadata={"policy_id": "POL-101", "category": "auto_approval"},
        ),
        Document(
            page_content=(
                "Policy ID: POL-102 (Minor Price Tolerance). If billed_qty matches GRN accepted_qty "
                "and unit price variance is positive but less than or equal to 2.0% AND total variance "
                "is under $150.00, the invoice is AUTO_APPROVED under minor freight/fx tolerance."
            ),
            metadata={"policy_id": "POL-102", "category": "price_tolerance"},
        ),
        Document(
            page_content=(
                "Policy ID: POL-103 (Damaged Goods & Overbilling Escalation). Vendors may only bill "
                "for GRN accepted_qty. If billed_qty exceeds GRN accepted_qty (e.g., billing for damaged "
                "units) OR price variance exceeds 2.0%, status MUST be ESCALATED_HITL and a debit/credit "
                "memo requested."
            ),
            metadata={"policy_id": "POL-103", "category": "escalation"},
        ),
    ]

    vector_store = Chroma.from_documents(
        documents=policies,
        embedding=embeddings,
        collection_name="synops_procurement_policies",
    )
    return vector_store


if __name__ == "__main__":
    init_erp_db()
    init_policy_vector_store()
    print("ERP SQLite DB and Ollama Policy Vector Store initialized.")