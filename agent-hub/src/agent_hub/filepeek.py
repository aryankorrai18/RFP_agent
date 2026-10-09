"""A local look at an attached file, before any agent or model sees it.

Two uses, neither sends the file anywhere:
- `describe()`: what kind of document it looks like, as labels only ("27 questions, 27 with answers; names a client and a
  date"), so the hub's language model can tell a past proposal from a new RFP or a call note without reading its words.
- `proposal_details()`: the client and submission date a proposal states about itself ("Prepared for: ...",
  "Submission date: ..."), so the person does not have to type them.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import date

from .store import Upload

TEXT_EXTENSIONS = (".txt", ".md", ".csv", ".eml", ".json")
MAX_BYTES = 2_000_000  # enough for any proposal; a bigger file is described by name only

_MONTHS = {m: i for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august", "september",
                                       "october", "november", "december"), start=1)}
_CLIENT = re.compile(r"^\W*(?:prepared for|submitted to|client|customer|buyer|issuer|for the attention of)\s*[:\-]\s*(?P<v>.+)$", re.I | re.M)
_DATE_LINE = re.compile(r"^\W*(?:submission date|submitted(?: on)?|date submitted|proposal date|date)\s*[:\-]\s*(?P<v>.+)$", re.I | re.M)
_QUESTION = re.compile(r"^\s*(?:#+\s*)?(?:\d+(?:\.\d+)*[.)]?\s+|q\d*[.:]\s*|question\s*\d*\s*[:.]\s*)?.{8,}\?\s*$", re.I | re.M)
_ANSWERED = re.compile(r"^\W*(?:response|answer|our response|reply)\s*[:\-]", re.I | re.M)
_EMAIL = re.compile(r"^(?:from|to|subject|sent|cc)\s*:", re.I | re.M)
_MEETING = re.compile(r"\b(?:call notes?|meeting notes?|discovery call|attendees|next steps)\b", re.I)


def read_text(upload: Upload) -> str | None:
    """The file's text when it can be read here without a parser library: plain text formats and .docx. Else None."""
    if upload.size > MAX_BYTES:
        return None
    try:
        data = upload.read()
        if upload.ext in TEXT_EXTENSIONS:
            return data.decode("utf-8", errors="replace")
        if upload.ext == ".docx":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                xml = z.read("word/document.xml").decode("utf-8", errors="replace")
            xml = re.sub(r"</w:p>", "\n", xml)
            return re.sub(r"<[^>]+>", "", xml)
    except (OSError, KeyError, zipfile.BadZipFile):
        return None
    return None


def describe(upload: Upload) -> str:
    """Labels only: never the file's own words, names or numbers."""
    kind = upload.ext.lstrip(".") or "file"
    text = read_text(upload)
    if text is None:
        return f"{upload.filename} ({kind}; contents not checked)"
    questions = len(_QUESTION.findall(text))
    answered = len(_ANSWERED.findall(text))
    labels = []
    if questions:
        labels.append(f"{questions} question{'s' if questions != 1 else ''}"
                      + (f", {answered} with answers" if answered else ", none answered"))
    if answered and questions and answered >= max(1, questions // 2):
        labels.append("looks like a completed proposal or questionnaire")
    elif questions >= 3 and not answered:
        labels.append("looks like an RFP or questionnaire to answer")
    if _EMAIL.search(text):
        labels.append("looks like an email")
    if _MEETING.search(text):
        labels.append("looks like call or meeting notes")
    details = proposal_details(text)
    if details.get("client"):
        labels.append("names a client")
    if details.get("submitted_on"):
        labels.append("states a submission date")
    return f"{upload.filename} ({kind}; " + ("; ".join(labels) if labels else "plain text") + ")"


_INDUSTRY_LINE = re.compile(r"\bindustry\s*[:\-]\s*(?P<v>[A-Za-z][A-Za-z &/-]{1,40}?)\s*(?:[.;|]|\n|$)", re.I)
_SEGMENT_LINE = re.compile(r"\b(?:segment|company size|size)\s*[:\-]\s*(?P<v>[A-Za-z][A-Za-z -]{1,30}?)\s*(?:[.;|]|\n|$)", re.I)


def deal_details(text: str | None) -> dict[str, str]:
    """The industry and segment a deal's notes state ("Industry: logistics. Segment: mid-market."), raw; the caller
    normalises them."""
    if not text:
        return {}
    head = "\n".join(text.splitlines()[:40]).replace("**", "")
    out: dict[str, str] = {}
    industry = _INDUSTRY_LINE.search(head)
    if industry:
        out["industry"] = industry.group("v").strip()
    segment = _SEGMENT_LINE.search(head)
    if segment:
        out["segment"] = segment.group("v").strip()
    return out


def _date(value: str) -> str | None:
    value = value.strip().lower()
    iso = re.search(r"\b(20\d\d)-(\d\d)-(\d\d)\b", value)
    try:
        if iso:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3))).isoformat()
        dmy = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(_MONTHS) + r")\s+(20\d\d)\b", value)
        if dmy:
            return date(int(dmy.group(3)), _MONTHS[dmy.group(2)], int(dmy.group(1))).isoformat()
        mdy = re.search(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d\d)\b", value)
        if mdy:
            return date(int(mdy.group(3)), _MONTHS[mdy.group(1)], int(mdy.group(2))).isoformat()
    except ValueError:
        return None
    return None


def proposal_details(text: str | None) -> dict[str, str]:
    """The client and submission date a document states about itself, from its first 60 lines."""
    if not text:
        return {}
    head = "\n".join(text.splitlines()[:60]).replace("**", "")
    out: dict[str, str] = {}
    client = _CLIENT.search(head)
    if client:
        value = re.split(r"\s{2,}|\s+\(|\s+[|·]\s+", client.group("v").strip())[0].strip(" .*_")
        if 1 < len(value) <= 120:
            out["client"] = value
    when = _DATE_LINE.search(head)
    if when and _date(when.group("v")):
        out["submitted_on"] = _date(when.group("v"))  # type: ignore[assignment]
    return out
