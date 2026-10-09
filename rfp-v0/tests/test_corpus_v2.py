"""The bundled demo seed and documents stay internally consistent."""

from __future__ import annotations

import json

from rfp_assistant.config import ROOT
from rfp_assistant.parsing.parser import parse_document

CORPUS = ROOT / "samples" / "corpus_v2"


def test_demo_seed_documents_and_outcomes_are_valid():
    seed = json.loads((CORPUS / "demo_seed.json").read_text(encoding="utf-8"))
    results = [(proposal["result"], proposal["loss_reason"]) for proposal in seed["proposals"]]
    assert sum(r == "won" for r, _ in results) == 4
    assert {reason for r, reason in results if r == "lost"} == {"technical fit", "price", "response quality", "incumbent"}
    for proposal in seed["proposals"]:
        assert proposal["pairs"]
        document = parse_document(proposal["file"], (CORPUS / proposal["file"]).read_bytes(), 400_000)
        assert document.char_count > 400
        for pair in proposal["pairs"]:
            assert pair["question"] in document.text
            assert pair["answer"] in document.text
    for entry in seed["sample_rfps"]:
        name = entry["file"]
        assert parse_document(name, (CORPUS / name).read_bytes(), 400_000).char_count > 400
