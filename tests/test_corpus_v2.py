"""The competitive corpus stays internally consistent: every answer key points at text that really
is in a past proposal, every proposal and RFP parses, and outcomes are well formed."""

from __future__ import annotations

import importlib.util
import json

from backend.config import ROOT
from backend.parser import parse_document

CORPUS = ROOT / "samples" / "corpus_v2"


def load_generator():
    spec = importlib.util.spec_from_file_location("make_corpus", CORPUS / "make_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_self_check_and_key_match():
    gen = load_generator()
    gen.check()
    on_disk = json.loads((CORPUS / "answer_key.json").read_text(encoding="utf-8"))
    assert on_disk == gen.manifest(), "answer_key.json is stale: rerun make_corpus.py --force"


def test_every_preferred_and_trap_answer_is_in_its_proposal_text():
    key = json.loads((CORPUS / "answer_key.json").read_text(encoding="utf-8"))
    texts = {p["client"]: parse_document(p["file"], (CORPUS / p["file"]).read_bytes(), 400_000).text
             for p in key["proposals"]}
    for rfp in key["rfps"].values():
        for q in rfp["questions"]:
            for entry in [q["preferred"], *q["traps"]]:
                if entry is None:
                    continue
                assert entry["clients"], entry["key"]
                for client in entry["clients"]:
                    assert entry["answer"] in texts[client], (entry["key"], client)


def test_corpus_mixes_outcomes_and_loss_reasons():
    key = json.loads((CORPUS / "answer_key.json").read_text(encoding="utf-8"))
    results = [(p["result"], p["loss_reason"]) for p in key["proposals"]]
    assert sum(r == "won" for r, _ in results) == 4
    assert {reason for r, reason in results if r == "lost"} == {"technical fit", "price", "response quality", "incumbent"}
    for name in key["rfps"]:
        assert parse_document(name, (CORPUS / name).read_bytes(), 400_000).char_count > 400
