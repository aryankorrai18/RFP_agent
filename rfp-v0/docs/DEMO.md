# Hackathon demo

## Recommended story

 Proposal teams repeatedly answer similar questions, but copy-pasting old responses is risky. This assistant retrieves approved evidence, learns from what reviewers and buyers preferred, and refuses to invent commitments. 

## Five-minute flow

1. **Company**   show the official facts and explain that these override old proposal content.
2. **Library**   show won and lost proposals, their approved answers, and outcome labels.
3. **Projects**   open a prepared RFP and show the extracted response requirements.
4. **Draft review**   open a grounded answer and point to `FACT-*` and `ANS-*` sources.
5. **Safety gate**   show a pricing, reference, or personnel item marked **Company input required**.
6. **Memory comparison**   show which evidence changed with learned lessons.
7. **Blind judge**   show the model and human comparison without revealing which draft used lessons.
8. **Memory**   finish with the inspectable event journal and playbook.

## Virtusa prepared data

The `samples/virtusa_demo` folder contains:

- an official fact sheet;
- won and lost past proposals;
- a new bank-modernization RFP;
- a cybersecurity evidence pack that can be loaded without Gemini calls.

The documents are synthetic demonstration data. Do not describe their commercial, client, staffing, or project claims as real Virtusa commitments.

## What to say about company approval

Company approval is expected for data the system cannot safely infer:

- project-specific scope and deliverables;
- pricing, taxes, rates, and payment terms;
- named client references and contract values;
- named personnel, CVs, and availability;
- contractual timelines, roles, and acceptance criteria.

That escalation is evidence that the assistant understands its trust boundary; it is not a failed draft.

## Token-conscious testing

- Loading prepared demo evidence uses no model extraction calls.
- Automated tests use fakes and consume no provider tokens.
- Drafting consumes one model call per requirement.
- Before/after comparison consumes two calls per requirement.
- Two-order blind judging consumes two calls per compared pair.
