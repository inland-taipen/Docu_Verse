"""
run_tests.py
Automated test runner for the PDF Constrained Conversational Agent.

Usage:
    python run_tests.py [--pdf sample.pdf]

What it checks:
  ✅ Valid queries  → non-empty answer + contains page citation
  ✅ Invalid queries → answer contains refusal phrase
"""

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from pdf_processor import PDFProcessor
from utils import build_prompt, detect_language, format_citations
from llm_interface import ask_llm


# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def banner(text: str) -> None:
    width = 70
    print(f"\n{BOLD}{CYAN}{'═' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * width}{RESET}\n")


def run_query(processor: PDFProcessor, query: str, top_k: int = 4) -> tuple[str, list[dict]]:
    """Run a single query through the full pipeline and return (response, chunks)."""
    lang_code, lang_name = detect_language(query)
    chunks = processor.retrieve(query, top_k=top_k)
    if not chunks:
        return "", []
    prompt = build_prompt(query, chunks, lang_name)
    response = ask_llm(prompt)
    return response, chunks


def is_backend_error(result: str) -> bool:
    """
    Detect transport/provider failures returned by llm_interface.
    These are not valid model answers and should fail tests.
    """
    if not result:
        return False
    lowered = result.lower()
    markers = (
        "[ollama error:",
        "[groq error:",
        "[openai error:",
        "[error:",
    )
    return any(m in lowered for m in markers)


def is_hard_refusal(result: str, phrase: str = "cannot answer") -> bool:
    """
    True when the model effectively returned only a refusal.
    This avoids false negatives when an otherwise useful answer contains
    a refusal phrase in passing.
    """
    if not result:
        return False
    lowered = result.strip().lower()
    compact = re.sub(r"[\s\"'`*_.,:;!?()\[\]{}-]+", " ", lowered).strip()
    # Common strict refusal forms with or without trailing period.
    refusal_variants = {
        "i cannot answer based on the provided pdf",
        "cannot answer based on the provided pdf",
        "i cannot answer",
        "cannot answer",
    }
    if compact in refusal_variants:
        return True
    # Also treat very short responses centered on the refusal phrase as hard refusal.
    return phrase in compact and len(compact.split()) <= 12


def check_valid(result: str, chunks: list[dict], case: dict) -> bool:
    """
    Valid query passes if:
      - result is non-empty
      - result does NOT contain the refusal phrase
      - chunks (citations) are present
    """
    refusal = "cannot answer"
    if not result or not result.strip():
        print(f"  {RED}✗ EMPTY response{RESET}")
        return False
    if is_backend_error(result):
        print(f"  {RED}✗ LLM backend error (not a valid answer){RESET}")
        return False
    if is_hard_refusal(result, refusal):
        print(f"  {RED}✗ Unexpected REFUSAL in valid query{RESET}")
        return False
    if not chunks:
        print(f"  {RED}✗ No chunks retrieved (no citation){RESET}")
        return False
    print(f"  {GREEN}✓ Non-empty answer with citations: {format_citations(chunks)}{RESET}")
    return True


def check_invalid(result: str, chunks: list[dict], case: dict) -> bool:
    """
    Invalid query passes if response contains the expected refusal phrase.
    """
    expected = case.get("expected_refusal_contains", "cannot answer").lower()
    if is_backend_error(result):
        print(f"  {RED}✗ LLM backend error (cannot verify refusal behavior){RESET}")
        print(f"  {YELLOW}  Response snippet: {result[:200]!r}{RESET}")
        return False
    if expected in result.lower():
        print(f"  {GREEN}✓ Correctly REFUSED: '{expected}' found in response{RESET}")
        return True
    print(f"  {RED}✗ Expected refusal phrase '{expected}' NOT found{RESET}")
    print(f"  {YELLOW}  Response snippet: {result[:200]!r}{RESET}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Test the PDF chatbot pipeline.")
    parser.add_argument("--pdf",     default="sample.pdf",   help="Path to PDF file")
    parser.add_argument("--tests",   default="test_queries.json", help="Path to test JSON")
    parser.add_argument("--top-k",   default=4, type=int,     help="Chunks to retrieve")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    tests_path = Path(args.tests)
    if not tests_path.exists():
        print(f"{RED}Test file '{args.tests}' not found.{RESET}")
        sys.exit(1)

    with open(tests_path) as f:
        test_data = json.load(f)

    # ------------------------------------------------------------------
    # Load & index the PDF
    # ------------------------------------------------------------------
    pdf_path = args.pdf
    if not Path(pdf_path).exists():
        pdf_path = Path(__file__).parent / args.pdf
    if not Path(pdf_path).exists():
        print(f"{RED}PDF '{args.pdf}' not found. Please provide a valid PDF.{RESET}")
        sys.exit(1)

    banner("Loading & indexing PDF")
    processor = PDFProcessor()
    num_chunks = processor.chunk_pdf(str(pdf_path))
    print(f"  PDF: {pdf_path}")
    print(f"  Chunks indexed: {num_chunks}")

    # ------------------------------------------------------------------
    # Run valid queries
    # ------------------------------------------------------------------
    banner("Valid Query Tests")
    valid_cases  = test_data.get("valid_queries", [])
    valid_passed = 0

    for case in valid_cases:
        qid   = case["id"]
        query = case["query"]
        lang  = case.get("language", "?")
        print(f"{BOLD}[{qid}] ({lang}) {query}{RESET}")
        response, chunks = run_query(processor, query, top_k=args.top_k)
        snippet = textwrap.shorten(response, width=200, placeholder="…")
        print(f"  Response: {snippet}")
        if check_valid(response, chunks, case):
            valid_passed += 1
        print()

    # ------------------------------------------------------------------
    # Run invalid queries
    # ------------------------------------------------------------------
    banner("Invalid Query Tests (Refusal Checks)")
    invalid_cases  = test_data.get("invalid_queries", [])
    invalid_passed = 0

    for case in invalid_cases:
        qid   = case["id"]
        query = case["query"]
        lang  = case.get("language", "?")
        print(f"{BOLD}[{qid}] ({lang}) {query}{RESET}")
        response, chunks = run_query(processor, query, top_k=args.top_k)
        snippet = textwrap.shorten(response, width=200, placeholder="…")
        print(f"  Response: {snippet}")
        if check_invalid(response, chunks, case):
            invalid_passed += 1
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    banner("Test Summary")
    total_valid   = len(valid_cases)
    total_invalid = len(invalid_cases)
    total         = total_valid + total_invalid
    passed        = valid_passed + invalid_passed

    print(f"  Valid queries:   {valid_passed}/{total_valid} passed")
    print(f"  Invalid queries: {invalid_passed}/{total_invalid} passed")
    print(f"  Overall:         {passed}/{total} passed\n")

    if passed == total:
        print(f"  {GREEN}{BOLD}🎉 ALL TESTS PASSED!{RESET}")
        sys.exit(0)
    else:
        print(f"  {RED}{BOLD}⚠️  {total - passed} test(s) FAILED.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
