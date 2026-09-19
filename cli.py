#!/usr/bin/env python3
"""
LexiRAG Command-Line Interface (CLI).
Enables querying, document batch ingestion, benchmarking, and database seeding directly from terminal.
"""

import argparse
import sys
import os
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from src.rag.rag_system import RAGSystem
from src.utils.evaluation_metrics import RAGSystemEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="⚖️ LexiRAG: Autonomous Legal Contract Intelligence Engine CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: query
    query_parser = subparsers.add_parser("query", help="Ask a question against indexed contracts")
    query_parser.add_argument("question", type=str, help="Legal question to evaluate")
    query_parser.add_argument("--k", type=int, default=4, help="Top-k clauses to retrieve")

    # Command: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a contract PDF or directory")
    ingest_parser.add_argument("path", type=str, help="Path to PDF file or contracts directory")

    # Command: eval
    eval_parser = subparsers.add_parser("eval", help="Run empirical benchmark on ground truth test set")

    # Command: seed
    seed_parser = subparsers.add_parser("seed", help="Seed benchmark contracts from embedded sqlite DB")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "seed":
        from scripts.seed_contracts import extract_and_seed
        extract_and_seed()
        sys.exit(0)

    rag = RAGSystem()

    if args.command == "query":
        print(f"\n🔍 Query: {args.question}")
        print("⏳ Executing Hybrid Retrieval (Dense Vector + BM25 RRF)...")
        result = rag.answer_query(args.question, top_k=args.k)
        
        print("\n" + "=" * 65)
        print("                        LEGAL ADVICE                        ")
        print("=" * 65)
        print(result["answer"])
        print("\n" + "-" * 65)
        print(f"📊 Citations ({len(result.get('citations', []))}):")
        for cite in result.get("citations", []):
            print(f"  • [Page {cite['page']}] {cite['source']}: \"{cite['snippet'][:100]}...\"")
        
        telemetry = result.get("telemetry", {})
        faithfulness = result.get("faithfulness", {})
        print(f"\n⚡ Telemetry: Total {telemetry.get('total_ms', 0)}ms (Retrieval: {telemetry.get('retrieval_ms', 0)}ms)")
        print(f"🛡️ Faithfulness Score: {faithfulness.get('faithfulness_score', 'N/A')} ({faithfulness.get('status', 'OK')})")
        print("=" * 65 + "\n")

    elif args.command == "ingest":
        target = Path(args.path)
        if not target.exists():
            print(f"❌ Target path does not exist: {target}")
            sys.exit(1)

        if target.is_file() and target.suffix.lower() == ".pdf":
            count = rag.add_pdf(str(target))
            print(f"✅ Successfully ingested {target.name} ({count} clauses).")
        elif target.is_dir():
            from src.core.pdf_loader import PDFLoader
            docs = PDFLoader.load_from_directory(str(target))
            if docs:
                count = rag.add_documents(docs)
                print(f"✅ Ingested {len(docs)} pages ({count} clauses) from {target}.")
            else:
                # Ingest text files if no PDFs
                txt_files = list(target.glob("*.txt"))
                total = 0
                for txt in txt_files:
                    with open(txt, "r", encoding="utf-8") as f:
                        total += rag.add_documents([f.read()], metadatas=[{"source": txt.name}])
                print(f"✅ Ingested {len(txt_files)} text contracts ({total} clauses).")

    elif args.command == "eval":
        evaluator = RAGSystemEvaluator(rag)
        test_set = [
            ("Who are the parties to the Agreement and what are their defined names?", 
             "Cloud Investments Ltd. (\"Company\") and Jack Robinson (\"Advisor\")."),
            ("What are the payments to the Advisor under the Agreement?", 
             "Fees of $9 per hour up to a monthly limit of $1,500, workspace expense of $100 per month."),
            ("Is there a non-compete obligation to the Advisor?", 
             "During the term of engagement with the Company and for a period of 12 months thereafter."),
            ("Whose consent is required for the assignment of the Agreement by the Buyer?", 
             "The consent of the Sellers or parties is required."),
            ("How much is the escrow amount?", 
             "The escrow amount is equal to $1,000,000.")
        ]
        summary = evaluator.evaluate_test_set(test_set)
        evaluator.print_report(summary)


if __name__ == "__main__":
    main()
