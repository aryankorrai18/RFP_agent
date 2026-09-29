"""Generate the competitive sample corpus (corpus v2): 8 past proposals with won/lost outcomes and
competing answers per topic, 2 new RFPs to draft, and an answer key.

Everything is fictional. The vendor is Larkspur Data (see data/fact_sheet.json). Clients, people,
reference numbers and URLs are invented; domains use the reserved .example TLD.

Why this corpus exists: the V1 samples had one answer per question, so plain search could never
pick a worse answer. Here most topics have two or three competing answers:

- a WINNING answer: specific and current, from a proposal that was won;
- a LOSING answer: vague, from a proposal lost on technical fit or response quality;
- an OUTDATED answer: correct when submitted, contradicted by the current fact sheet;
- a NEUTRAL answer: good, from a proposal lost on price or to an incumbent (must not be punished).

answer_key.json records, for every question in the new RFPs, which past answer a good system
should use, which ones are traps, and which questions are true gaps (Needs SME).

Run from rfp-v0:  .venv/Scripts/python samples/corpus_v2/make_corpus.py  [--force]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import docx
import openpyxl
from openpyxl.styles import Alignment, Font

HERE = Path(__file__).resolve().parent

# --- competing answers, by topic ---------------------------------------------------------------
# Each entry: (key, question as asked in that proposal, answer as submitted, kind)
# kind: winning | losing | outdated | neutral | only

A = {
    # T01 company overview
    "overview_2026": ("Provide a brief overview of your company.",
        "Larkspur Data was founded in 2019 and provides a cloud customer-data platform for regulated industries. "
        "We have approximately 240 employees across offices in Denver, USA and Dublin, Ireland, and serve more than "
        "120 customers in banking, insurance and healthcare.", "winning"),
    "overview_2023": ("Tell us about your company, including size and locations.",
        "Larkspur Data is a Denver-based software company founded in 2019. Our team of about 150 people builds a "
        "customer-data platform for financial institutions.", "outdated"),

    # T02 SSO standards
    "sso_specific": ("Which single sign-on protocols and identity providers are supported?",
        "Larkspur supports single sign-on via SAML 2.0 and OpenID Connect. SSO is tested with Okta, Microsoft Entra "
        "ID and Google Workspace, and administrators can require SSO for every user in the organisation.", "winning"),
    "sso_vague": ("Does the platform support single sign-on?",
        "Yes, we support industry-standard single sign-on and can work with most identity providers.", "losing"),

    # T03 automated provisioning (SCIM) - not in the fact sheet
    "scim_2025": ("Describe how user accounts are provisioned and de-provisioned automatically.",
        "Larkspur supports automatic provisioning and de-provisioning through SCIM 2.0 with Okta and Microsoft "
        "Entra ID. Deactivating a user in the identity provider removes their Larkspur access within minutes, and "
        "group membership in the identity provider maps to Larkspur roles.", "winning"),
    "scim_2023": ("How are user accounts created and removed?",
        "User accounts are created by administrators in the Larkspur console, individually or by CSV upload. "
        "Automated provisioning is on our product roadmap.", "outdated"),

    # T04 administrator MFA - not in the fact sheet
    "mfa_admin": ("Is multi-factor authentication enforced for administrators?",
        "Yes. Multi-factor authentication is mandatory for every administrative account. Administrators sign in "
        "through SSO with MFA enforced by the identity provider, or with Larkspur's built-in TOTP authenticator. "
        "Admin sessions expire after 8 hours of inactivity.", "winning"),

    # T05 certifications
    "certs_2026": ("Which security certifications and attestations do you hold?",
        "Larkspur holds a SOC 2 Type II report covering 1 April 2025 to 31 March 2026, available under NDA, and is "
        "certified to ISO/IEC 27001:2022 (certificate issued October 2025).", "winning"),
    "certs_2024": ("List your current security certifications.",
        "Larkspur holds a SOC 2 Type I report dated 30 June 2024. Our SOC 2 Type II audit period is under way, "
        "and ISO/IEC 27001 certification is planned.", "outdated"),

    # T06 penetration testing - firm and cadence not in the fact sheet
    "pentest_specific": ("How often is the platform penetration tested, and by whom?",
        "Ironbridge Security, an independent CREST-accredited firm, performs a full penetration test of the "
        "platform at least once a year, and we run automated external vulnerability scans every quarter. The "
        "executive summary of the latest test is available under NDA.", "winning"),
    "pentest_vague": ("Describe your security testing practices.",
        "We perform regular security testing of our platform and take security very seriously.", "losing"),

    # T07 data residency
    "residency_2026": ("Where will our data be hosted? Can it remain in the EU?",
        "Larkspur is hosted on AWS. Each customer chooses the US region (us-east-1) or the EU region "
        "(eu-central-1, Frankfurt). Customer data, backups and logs stay in the chosen region.", "winning"),
    "residency_2023": ("Where is customer data stored?",
        "All customer data is hosted on AWS in the us-east-1 region (Northern Virginia).", "outdated"),

    # T08 availability SLA - credit tiers not in the fact sheet
    "sla_2025": ("What uptime do you commit to, and what remedies apply if you miss it?",
        "Our SLA commits to 99.9% monthly uptime for the production platform. If we miss it, customers receive "
        "service credits of 10% of the monthly fee below 99.9% and 25% below 99.0%. Uptime is published on our "
        "status page at status.larkspurdata.example.", "winning"),
    "sla_2023": ("State your availability commitment.",
        "Larkspur commits to 99.5% monthly availability, excluding scheduled maintenance windows.", "outdated"),

    # T09 support
    "support_2025": ("Describe your support model, including hours and response times.",
        "Severity-1 incidents are handled 24/7 with an initial response within 30 minutes. Other issues are handled "
        "Monday to Friday, 8am to 6pm in the customer's region, with a first response within 4 business hours. "
        "Customers reach us through the support portal or by email, and every account has a named customer "
        "success manager.", "winning"),
    "support_vague": ("How can we contact support?",
        "Our friendly support team is always happy to help through our support portal.", "losing"),

    # T10 implementation timeline
    "impl_phased": ("Describe your implementation approach and typical timeline.",
        "A typical implementation takes 6 to 8 weeks in four phases: discovery (weeks 1-2), configuration and data "
        "connection (weeks 3-5), user acceptance testing (week 6), and go-live with two weeks of hypercare "
        "(weeks 7-8). A dedicated Larkspur implementation manager leads the project throughout.", "winning"),
    "impl_2023": ("How long does implementation take?",
        "Implementations typically take 12 to 16 weeks depending on the number of data sources.", "outdated"),
    "impl_vague": ("Outline the onboarding process.",
        "Every customer is different, so we agree the plan and timeline during kickoff.", "losing"),

    # T11 historical data migration - not in the fact sheet
    "migration": ("How do you migrate our historical data onto the platform?",
        "Historical data is loaded in bulk through our Amazon S3, Snowflake or BigQuery connectors. We reconcile row "
        "counts and checksums against the source before sign-off, and most customers load two to three years of "
        "history before go-live.", "winning"),

    # T12 accessibility - not in the fact sheet
    "accessibility": ("Does the user interface meet accessibility standards?",
        "The Larkspur web application conforms to WCAG 2.1 level AA. An external accessibility firm audits it "
        "every year, and our VPAT (Voluntary Product Accessibility Template) is available on request.", "winning"),

    # T13 AI and customer data - not in the fact sheet
    "ai_2026": ("Is our data used to train AI models?",
        "No. Customer data is never used to train AI models, ours or any third party's. Larkspur's AI-assisted "
        "features are off by default, can be enabled per workspace, and process data only within the customer's "
        "chosen hosting region.", "winning"),
    "ai_vague": ("Describe your approach to artificial intelligence.",
        "We take responsible AI seriously and follow industry best practices.", "losing"),

    # T14 audit logs
    "audit_2024": ("How long are audit logs retained?",
        "Audit logs of user and administrator activity are retained for 12 months and can be exported through our "
        "REST API.", "winning"),
    "audit_2023": ("Do you keep audit logs?",
        "Yes. Administrative actions are logged and retained for 90 days.", "outdated"),

    # T15 breach notification
    "breach": ("How quickly will you notify us of a security breach?",
        "We notify affected customers of a confirmed security breach involving their data within 72 hours of "
        "confirmation, through their named customer success manager and by email to their security contact.",
        "winning"),

    # Good answers from deals lost on price or to an incumbent: must not be punished.
    "sla_neutral": ("What is your uptime guarantee?",
        "We commit to 99.9% monthly uptime for production, with service credits if we miss it.", "neutral"),
    "support_neutral": ("What support hours do you offer?",
        "Severity-1 incidents are handled around the clock; other requests are handled during business hours in "
        "your region through our portal and by email.", "neutral"),
    "residency_neutral": ("Can data be kept within the European Union?",
        "Yes. Customers can choose our EU region in Frankfurt (eu-central-1), and their data stays in that region.",
        "neutral"),
}

# --- the 8 past proposals ----------------------------------------------------------------------
# (file, client, industry, submitted_on, result, loss_reason, rfp reference, sections)

PROPOSALS = [
    ("2023-05_brightwater_credit_union.docx", "Brightwater Credit Union", "finance", "2023-05-18", "lost",
     "technical fit", "BCU-RFP-2023-11",
     [("Company", ["overview_2023"]),
      ("Identity and access", ["sso_vague", "scim_2023"]),
      ("Hosting", ["residency_2023"]),
      ("Delivery", ["impl_2023"]),
      ("Security operations", ["audit_2023"])]),
    ("2023-11_halden_mutual.docx", "Halden Mutual Insurance", "insurance", "2023-11-02", "lost", "price",
     "HMI-2023-DP-04",
     [("Service levels", ["sla_2023", "support_neutral"]),
      ("Security", ["breach"]),
      ("Hosting", ["residency_2023"])]),
    ("2024-09_st_aurelia_health.docx", "St. Aurelia Health System", "healthcare", "2024-09-12", "won", None,
     "SAHS-24-117",
     [("Security", ["pentest_specific", "audit_2024", "breach"]),
      ("Data protection", ["residency_2026"]),
      ("Implementation", ["impl_phased", "migration"])]),
    ("2024-10_kestrel_savings.docx", "Kestrel Savings Bank", "finance", "2024-10-07", "won", None, "KSB/2024/RFP/09",
     [("Assurance", ["certs_2024", "pentest_specific"]),
      ("Identity", ["sso_specific", "mfa_admin"]),
      ("Data", ["migration"])]),
    ("2025-02_corvid_logistics.docx", "Corvid Logistics", "logistics", "2025-02-20", "lost", "response quality",
     "CL-PROC-2025-003",
     [("Identity", ["sso_vague"]),
      ("Security", ["pentest_vague"]),
      ("Delivery", ["impl_vague"]),
      ("Support", ["support_vague"]),
      ("Artificial intelligence", ["ai_vague"])]),
    ("2025-06_pinecrest_federal.docx", "Pinecrest Federal Savings", "finance", "2025-06-09", "won", None,
     "PFS-2025-CDP",
     [("Identity and access", ["sso_specific", "scim_2025", "mfa_admin"]),
      ("Service levels", ["sla_2025", "support_2025"]),
      ("Accessibility", ["accessibility"]),
      ("Data", ["migration"])]),
    ("2025-10_everlake_insurance.docx", "Everlake Insurance Group", "insurance", "2025-10-14", "lost", "incumbent",
     "EIG-SEC-2025-21",
     [("Service", ["sla_neutral", "support_neutral"]),
      ("Hosting", ["residency_neutral"]),
      ("Delivery", ["impl_phased"])]),
    ("2026-03_tidewell_health.docx", "Tidewell Regional Health", "healthcare", "2026-03-03", "won", None,
     "TRH-IT-2026-008",
     [("Company", ["overview_2026"]),
      ("Assurance", ["certs_2026", "pentest_specific"]),
      ("Data protection", ["residency_2026", "ai_2026"]),
      ("Implementation", ["impl_phased"]),
      ("Security operations", ["audit_2024", "breach"])]),
]

# --- the 2 new RFPs to draft -------------------------------------------------------------------
# (ref, question, preferred answer key or None for a true gap, trap keys)

ASHFORD_RFP = [
    ("2. Company", [
        ("2.1", "Provide an overview of your company, including headcount and office locations. (Maximum 120 words)",
         "overview_2026", ["overview_2023"]),
        ("2.2", "Provide three references from US community banks, with contact names.", None, []),
    ]),
    ("3. Identity and access", [
        ("3.1", "MANDATORY: Which single sign-on protocols and identity providers do you support?",
         "sso_specific", ["sso_vague"]),
        ("3.2", "MANDATORY: How are user accounts created and removed automatically when staff join or leave?",
         "scim_2025", ["scim_2023"]),
        ("3.3", "Is multi-factor authentication enforced for administrative users?", "mfa_admin", []),
    ]),
    ("4. Security assurance", [
        ("4.1", "List your current security certifications and attestations.", "certs_2026", ["certs_2024"]),
        ("4.2", "How often is your platform penetration tested, and by whom?", "pentest_specific", ["pentest_vague"]),
        ("4.3", "How long are audit logs retained, and can we export them?", "audit_2024", ["audit_2023"]),
    ]),
    ("5. Hosting and service", [
        ("5.1", "Where will our data be stored?", "residency_2026", ["residency_2023"]),
        ("5.2", "What availability do you commit to, and what remedies apply if it is missed?", "sla_2025",
         ["sla_2023"]),
        ("5.3", "Describe your support model, including response times for critical incidents.", "support_2025",
         ["support_vague"]),
    ]),
    ("6. Delivery", [
        ("6.1", "Describe your implementation approach and a typical timeline. (Maximum 150 words)", "impl_phased",
         ["impl_2023", "impl_vague"]),
        ("6.2", "How will you migrate seven years of historical member data?", "migration", []),
        ("6.3", "Can your platform receive real-time transaction events from Fiserv DNA?", None, []),
    ]),
    ("7. Governance and commercials", [
        ("7.1", "Will our member data be used to train AI models?", "ai_2026", ["ai_vague"]),
        ("7.2", "Provide a five-year total cost of ownership, including implementation.", None, []),
    ]),
]

MEADOWGATE_QUESTIONNAIRE = [
    ("MG-01", "Access", "Do you support SAML 2.0 or OpenID Connect single sign-on?", "sso_specific", ["sso_vague"]),
    ("MG-02", "Access", "Do you support SCIM 2.0 user provisioning?", "scim_2025", ["scim_2023"]),
    ("MG-03", "Assurance", "Provide your SOC 2 report type and audit period.", "certs_2026", ["certs_2024"]),
    ("MG-04", "Assurance", "Is your platform tested by an independent penetration testing firm?", "pentest_specific",
     ["pentest_vague"]),
    ("MG-05", "Privacy", "Can all patient-related data remain within the EU?", "residency_2026",
     ["residency_2023", "residency_neutral"]),
    ("MG-06", "Privacy", "Is customer data used to train machine-learning models?", "ai_2026", ["ai_vague"]),
    ("MG-07", "Operations", "What is your monthly uptime commitment?", "sla_2025", ["sla_2023", "sla_neutral"]),
    ("MG-08", "Operations", "How quickly will you notify us of a confirmed data breach?", "breach", []),
    ("MG-09", "Accessibility", "Does the clinician-facing interface meet WCAG 2.1 AA?", "accessibility", []),
    ("MG-10", "Commercial", "Is HIPAA business associate agreement (BAA) coverage included in the standard price?",
     None, []),
]


# --- builders ------------------------------------------------------------------------------------

def build_proposal(path: Path, client: str, submitted_on: str, rfp_ref: str, sections) -> None:  # noqa: ANN001
    document = docx.Document()
    document.add_heading(f"Larkspur Data: Response to {client}", level=0)
    document.add_paragraph(f"RFP reference {rfp_ref} · Submitted {submitted_on} · Prepared by Larkspur Data "
                           "Proposals Team (proposals@larkspurdata.example) · Fictional sample document")
    document.add_paragraph("Larkspur Data thanks you for the opportunity to respond. Our answers to each question "
                           "follow, in the order of your request.")
    number = 1
    for heading, keys in sections:
        document.add_heading(heading, level=1)
        for key in keys:
            question, answer, _kind = A[key]
            document.add_paragraph(f"Q{number}. {question}")
            document.add_paragraph(f"Response: {answer}")
            number += 1
    document.save(path)


def build_ashford(path: Path) -> None:
    document = docx.Document()
    document.add_heading("Ashford Community Bank: Request for Proposal", level=0)
    document.add_paragraph("Member Data Platform, RFP reference ACB-2026-031 (fictional)")
    document.add_heading("1. Instructions", level=1)
    document.add_paragraph("Responses are due 15 December 2026, 5pm Central, to vendorrfp@ashfordbank.example.")
    document.add_paragraph("Answer every question in sections 2 to 7. Questions marked MANDATORY must be answered.")
    for heading, questions in ASHFORD_RFP:
        document.add_heading(heading, level=1)
        for ref, text, _pref, _traps in questions:
            document.add_paragraph(f"{ref} {text}")
    document.save(path)


def build_meadowgate(path: Path) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Vendor questionnaire"
    sheet.append(["ID", "Domain", "Question", "Vendor response"])
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for ref, domain, question, _pref, _traps in MEADOWGATE_QUESTIONNAIRE:
        sheet.append([ref, domain, question, ""])
    sheet.column_dimensions["C"].width = 72
    sheet.column_dimensions["D"].width = 50
    for row in sheet.iter_rows(min_row=2):
        row[2].alignment = Alignment(wrap_text=True)
    workbook.save(path)


def manifest() -> dict:
    used_in = {}
    for file, client, *_rest, sections in PROPOSALS:
        for _heading, keys in sections:
            for key in keys:
                used_in.setdefault(key, []).append(client)

    def entry(key):
        if key is None:
            return None
        question, answer, kind = A[key]
        return {"key": key, "kind": kind, "question": question, "answer": answer, "clients": used_in.get(key, [])}

    return {
        "vendor": "Larkspur Data (fictional)",
        "proposals": [
            {"file": file, "client": client, "industry": industry, "submitted_on": submitted, "result": result,
             "loss_reason": loss, "rfp_reference": ref}
            for file, client, industry, submitted, result, loss, ref, _sections in PROPOSALS
        ],
        "rfps": {
            "2026-12_ashford_community_bank_rfp.docx": {
                "client": "Ashford Community Bank", "industry": "finance",
                "questions": [{"ref": ref, "question": q, "preferred": entry(p), "traps": [entry(t) for t in traps],
                               "gap": p is None}
                              for _h, qs in ASHFORD_RFP for ref, q, p, traps in qs],
            },
            "2026-12_meadowgate_health_questionnaire.xlsx": {
                "client": "Meadowgate Health Partners", "industry": "healthcare",
                "questions": [{"ref": ref, "question": q, "preferred": entry(p), "traps": [entry(t) for t in traps],
                               "gap": p is None}
                              for ref, _d, q, p, traps in MEADOWGATE_QUESTIONNAIRE],
            },
        },
    }


def demo_seed() -> dict:
    """The demo workspace's starting history, with no model calls: every past proposal's questions
    and answers exactly as build_proposal writes them into its document, plus its outcome."""
    proposals = []
    for file, client, industry, submitted, result, loss, ref, sections in PROPOSALS:
        pairs, number = [], 1
        for heading, keys in sections:
            for key in keys:
                question, answer, _kind = A[key]
                pairs.append({"section": heading, "reference": f"Q{number}", "question": question, "answer": answer})
                number += 1
        proposals.append({"file": file, "client": client, "industry": industry, "submitted_on": submitted,
                          "result": result, "loss_reason": loss, "rfp_reference": ref, "pairs": pairs})
    return {
        "vendor": "Larkspur Data (fictional)",
        "proposals": proposals,
        "sample_rfps": [
            {"file": "2026-12_ashford_community_bank_rfp.docx", "name": "Ashford Community Bank RFP",
             "client": "Ashford Community Bank", "industry": "finance"},
            {"file": "2026-12_meadowgate_health_questionnaire.xlsx", "name": "Meadowgate Health security questionnaire",
             "client": "Meadowgate Health Partners", "industry": "healthcare"},
        ],
    }


def check() -> None:
    """Every answer is used, every key referenced exists, and each proposal's result is valid."""
    used = {key for *_r, sections in PROPOSALS for _h, keys in sections for key in keys}
    missing = set(A) - used
    assert not missing, f"answers never used in a proposal: {sorted(missing)}"
    referenced = {k for _h, qs in ASHFORD_RFP for *_x, p, traps in qs for k in [p, *traps] if k} | \
                 {k for *_x, p, traps in MEADOWGATE_QUESTIONNAIRE for k in [p, *traps] if k}
    assert referenced <= set(A), f"unknown keys in RFPs: {sorted(referenced - set(A))}"
    for key in referenced:
        assert key in used, f"{key} is expected in an RFP but no proposal contains it"
    for file, client, industry, submitted, result, loss, *_rest in PROPOSALS:
        assert result in ("won", "lost"), file
        assert (result == "lost") == (loss is not None), file


if __name__ == "__main__":
    check()
    force = "--force" in sys.argv
    outputs = {HERE / file: (lambda p, c=client, s=submitted, r=ref, secs=sections: build_proposal(p, c, s, r, secs))
               for file, client, _i, submitted, _r, _l, ref, sections in PROPOSALS}
    outputs[HERE / "2026-12_ashford_community_bank_rfp.docx"] = build_ashford
    outputs[HERE / "2026-12_meadowgate_health_questionnaire.xlsx"] = build_meadowgate
    for path, build in outputs.items():
        if path.exists() and not force:
            print(f"kept    {path.name} (exists; use --force to rewrite)")
            continue
        build(path)
        print(f"wrote   {path.name}")
    (HERE / "answer_key.json").write_text(json.dumps(manifest(), indent=2), encoding="utf-8")
    print("wrote   answer_key.json")
    (HERE / "demo_seed.json").write_text(json.dumps(demo_seed(), indent=2), encoding="utf-8")
    print("wrote   demo_seed.json")
