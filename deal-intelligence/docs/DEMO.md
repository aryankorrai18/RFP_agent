# Demo

## Recommended story

Before a call or a forecast review, an account executive has to rebuild a deal's history from emails and notes, and cannot easily tell what happened in the similar deals the team already won and lost. Deal Intelligence writes a brief that cites its sources, warns with honest counts from similar closed deals, says which obvious move has backfired before, and changes when a new outcome is recorded. The change is shown as counts and cited deals, not as a rewritten paragraph.

The story has two beats. First the pricing deal: a customer asks for a discount, and a brief with no history recommends it; memory says it has backfired in every similar deal, with the counts. Then the SSO loop: record one lost deal and watch the next brief on a different deal change.

All companies, people and numbers in the demo are synthetic. Halcyon Software, Cedarline Bank, Larkfield Credit Union, Juniper & Vale Stores and every other account are fictional.

## Before you start

- `.env` has one model key. For the Hindsight memory you also need `DEAL_HINDSIGHT_API_KEY`; without it the app uses its free local memory (see "Run it with no Hindsight account" below). The app is running on <http://127.0.0.1:8002> (see the [README](../README.md)).
- The status card (System health) shows the model, "Hindsight Cloud · chunks" (or "Local memory (SQLite, no Hindsight account)"), and the deal and lesson counts. A "Hindsight unreachable" line means briefs will use the degraded path.
- The numbers below come from running the project's offline fakes (the same builders the tests use) over the real seed, recall returning every closed deal, with no network, model or Hindsight call. Live recall can return fewer candidates, so check the first run against the screen before presenting. Model wording varies between runs; the counts do not.

## Three-minute flow

The walkthrough follows the box on the **Deals** page ("Demo walkthrough, about three minutes"), which opens the SSO deal; the pricing deal is the strongest opening and is opened from the deal list.

### 0:00 Create the demo workspace (no model calls)

1. Open the workspace menu in the top bar, choose **New workspace...**, and press **Open the demo workspace**. (On a fresh install, **Deals** shows the same button.)
2. Expected: the notice "Demo workspace ready: 24 deals loaded (5 open, 19 closed, 101 notes and emails) with no model calls." The Deals page shows 5 open deals, 19 closed deals in memory and "Won 9 of 19".
3. Say: this is a fictional company, Halcyon Software, selling a revenue-analytics platform. The workspace has its own database and its own memory (a fresh pair of Hindsight banks, or a local file).

### 0:20 The pricing deal: memory says the discount backfires

1. On **Deals**, open Juniper & Vale Stores (D-024: retail, mid-market, negotiation, $130,000). The CFO asked for 15 percent off against a competitor quote from Orbitly; a pricing objection is raised and no counter has been sent.
2. On the **Brief** tab choose **No history** and press **Generate brief**. 1 model call; the deal was already read (the seed carries its signals). Say: with no history to look at, the obvious next step is to give the discount. In our live run with a real model (Gemini), this arm recommended exactly that. Wording varies between runs, so treat what you see as one sample.
3. Choose **Hindsight memory** and press **Generate brief**. 1 model call. Expected, deterministic whatever the model writes:
   - **Likely to backfire**: "Discount offer: used in 4 similar deals, 0 won." seen in D-008, D-019, D-020 and D-021, with "Based on 19 closed deals." The card says these are counts, not proof of cause.
   - **Similar closed deals**: 8 chips (D-004, D-021, D-020, D-022, D-001, D-023, D-008, D-019).
   - **What memory says**: one warning, "Orbitly is in the deal: **1 of 2** similar closed deals were lost (1 won despite it)." citing D-004 and D-021. (The pricing objection is only raised, not unresolved, so it produces no objection warning.)
   - **Recommended next steps**: chosen by the model from the ranked list, which now excludes the discount. It starts with ROI workbook with the buyer's own numbers (3 of 3 similar deals won) and Executive sponsor call (3 of 3). If the model names the discount anyway, the step is dropped in code and the brief is flagged `avoided_play`.
4. Say, honestly: the discount was used in four lost similar deals, and the two won pricing deals closed with the ROI workbook plus an executive sponsor call or a second champion instead. This is a planted pattern in synthetic data, and a count is not a cause.
5. Live finding, say it plainly: in our run the arm with every closed deal pasted into the prompt also avoided the discount, and so did the memory arm; only No history recommended it. So at this size stuffing every summary also works. Memory's claim is targeted retrieval, counted evidence and citations, a rule enforced in code, and scale. The comparison below shows it on the same deal if you want to run it (3 model calls).

### 1:10 The SSO deal: open Cedarline and brief it in memory mode

1. Return to **Deals** and press **Open Cedarline Bank**. The deal is "Cedarline Bank - Halcyon Pipeline renewal-of-interest" (D-015): fintech, mid-market, evaluation stage, $180,000. Say: Cedarline lost a smaller deal with us last year (D-006, feature gap); this one is a revival.
2. On the **Brief** tab leave **Hindsight memory** selected (the default) and press **Generate brief**. 1 model call.
3. Expected, deterministic whatever the model writes:
   - **Deal flags** (checked without a model): two overdue promises (for example "Promise overdue by 10 days (Priya Nair): Send the security whitepaper by Friday."), an SSO objection still unresolved for about five weeks, a security-review objection still raised, and Brightline named as a competitor. Day counts grow as the workspace ages.
   - **What memory says** with two warnings:
     - "SSO objection is unresolved: **2 of 3** similar closed deals were lost (1 won despite it)." citing D-005, D-011, D-013.
     - "Brightline is in the deal: **2 of 3** similar closed deals were lost (1 won despite it)." citing D-001, D-003, D-011.
   - **Similar closed deals**: 8 chips (D-011, D-001, D-003, D-009, D-005, D-013, D-006, D-002) and "Based on 19 closed deals in memory: a small sample."
   - **Recommended next steps**: plays chosen by the model from the ranked list, each with its reasons and cited deals. The list starts with Security architecture review in week 1 (used in 4 similar deals, 3 won) and SSO walkthrough with their IdP admin (2 of 2 won), because those plays are meant to resolve the open SSO and security-review objections. Discount offer is ranked last (used in 3 similar deals, 1 won, 2 lost on a reason a play could change). There is no "Likely to backfire" card on this deal: the discount won once here (D-013), so the strict rule (at least 3 similar deals and no win) does not flag it.
4. Open the **Memory evidence** tab (no model call) to show the same counts, each similar deal with the keys it matched on, and the plays with their counts. The tab also holds the Reflect panel: **What Hindsight says about deals like this** (on the local backend, **What the memory says**). It shows the cached Reflect text, "Not asked yet" at first. On Hindsight Cloud, **Refresh** needs a second click ("Confirm: spends a little Hindsight credit") because it asks the lessons bank again; on the local backend it is free and computed from the recorded outcomes. Say: this is interpretation, shown beside the evidence; it is never cited in the brief.
5. Say, honestly: the data is planted to be noisy. D-013 won despite an unresolved SSO objection; D-003 lost even though the security review was done early (it went to a competitor). Those stay in the counts.

### 1:50 Record Larkfield as lost

1. Return to **Deals** and open Larkfield Credit Union (D-018: fintech, mid-market, negotiation, $140,000; an unresolved SSO objection, a discount request, a silent champion, the economic buyer not engaged).
2. Open the **Close deal / learning loop** tab. Press **Lost**, choose **Unresolved objection** (the card says "Plays get credit or penalty"), and tick **Discount offer** (PLAY-05) under the commercial plays.
3. Press **Record outcome**, then press again to confirm ("Confirm: this closes the deal").
4. Expected: "Outcome recorded: lost (Unresolved objection)". **Play credit** shows Discount offer -0.1, times used 8, won 1, lost (quality) 3, lost (other) 4. **What memory learned** lists "D-018 recorded as lost (unresolved_objection). Plays used: PLAY-05." and "PLAY-05 -0.1 credit after D-018 was lost (unresolved_objection).", and says 2 lessons were added.
5. Say: SQLite recorded it at once; the memory is updated in the background. On Hindsight the lessons may show as consolidating for a while, and the next step does not wait for them.

### 2:20 Brief Cedarline again and open the diff

1. Press **Brief Cedarline Bank** (or go back to its Brief tab) and press **Generate brief** in **Hindsight memory** mode. 1 model call.
2. Expected: a **What changed because of memory since your previous Hindsight brief** block appears under the brief, with the tag "Compared without a model":
   - Counts changed: the unresolved objection warning went from **2 of 3** similar deals lost to **3 of 4**.
   - New similar deals recalled: **D-018**.
   - The warning now cites D-005, D-011, D-013 and D-018. The Brightline warning is unchanged at 2 of 3. The brief now says it is based on 20 closed deals.
   - "Now recommended" or "No longer recommended" lines appear only if the model chose different plays this time; they depend on the model.
3. Also in the evidence: Discount offer is now used in 4 similar deals, 1 won, 3 lost on a reason a play could change. The ranking of the other plays does not change.
4. The block can be reopened any time under **Brief history**: tick two briefs and press **Compare the selected two**, or press **Compare the two newest Hindsight briefs**. The comparison is counted from saved evidence and makes no model call. It also lists plays that newly became, or stopped being, "likely to backfire".
5. Say: one recorded outcome changed the next brief, and the changed sentence cites the deal that caused it, D-018. The same outcome also moves the pricing deal: brief D-024 again and its backfire card now reads "Discount offer: used in 5 similar deals, 0 won."

### 2:50 Before and after comparison

1. Under the brief, in **Does memory change the brief?**, press **Compare memory on this deal**, then press again ("Confirm: this makes 3 model calls").
2. Expected after the three briefs are written: three columns, **No history**, **All deals in the prompt** and **Hindsight memory**, each with recommended plays, top claims and tokens used. The page notes whether Hindsight memory and the all-deals arm land on the same recommended play, how many closed deals Hindsight memory cited, and which arms (if any) recommended a play memory says to avoid. It warns only if the three prompts differ by more than the evidence block; all three should share one prompt hash.
3. Read out the honesty line in the blue box: "n = 20 closed deals. At this size putting every summary in the prompt also works; Hindsight's value is targeted retrieval, citations for every changed sentence, and scale."
4. Say: the middle column is the strongest alternative, and it is shown on purpose. It carries every summary and no counts, gate, warnings or ranked plays. On the SSO deal in our live run the recommended plays were identical across the three arms, and the difference was the counted warnings and the cited deals. Run the same comparison on the pricing deal to see the discount question.

### 3:20 The Memory page

Open **Memory** in the sidebar. Show:

- **What memory learned**: the journal (deals added, the outcome, the play credit).
- **Lessons from outcomes**: each outcome as plain sentences with a positive, negative or neutral signal. The Larkfield lessons are `negative` because "unresolved objection" is a quality loss. A price loss would be `neutral`.
- **Plays: what worked**: exact counts and small credit.
- **Closed deals in memory (20)**: the summaries briefs recall.
- **Memory self-check**: leave one closed deal out, rebuild the recommendations from the others using only SQLite (no model, no Hindsight), and compare with what happened. On the seed, before Larkfield is recorded (19 closed deals): the top 3 recommended plays included a play really used in **5 of 7** won deals that had a recommendation (9 won deals in all); it warned about the unresolved objection in **6 of 8** lost deals that had one; and it flagged a backfiring play that was really used in **4 of 10** lost deals. After Larkfield (20 closed deals): lost deals 11, warned 7 of 9, flagged in 4. Say it plainly: this is a sanity check on a very small sample, one deal more or less moves it, and it is not a benchmark.
- **Playbook**: on Hindsight, not built until someone presses **Build playbook** and confirms; it spends Hindsight credits, so skip it unless you have credits to spare. On the local backend the API computes it from the recorded outcomes, free, with a footer saying so.
- **Hindsight bank info** (or **Memory backend** on the local backend): reachable, extraction mode `chunks`, the two bank names.

## Run it with no Hindsight account

Leave `DEAL_HINDSIGHT_API_KEY` empty (and the URL at its Cloud default), or set `DEAL_MEMORY_BACKEND=local`. The app then uses a local memory: both banks live in `memory.db` beside the workspace database (SQLite FTS5). Everything above works, with these differences:

- Recall is keyword search (bm25), not semantic search. It is weaker than Hindsight's; the gate and the counts still decide what counts. On the seeded deals the offline check gave the same warnings, counts and ranked plays for Cedarline and the pricing deal as above (the order of the similar-deal chips can differ).
- Nothing is sent anywhere and no Hindsight credits are spent. Seeding makes 0 model calls and 0 network calls. The Reflect panel, the playbook and the self-check are free.
- The playbook and the Reflect text are computed locally from the recorded outcomes (counts, no invented claims) and say so.
- The status card shows "Local memory (SQLite, no Hindsight account)", and `GET /v1/status` reports `hindsight.backend` as `local`.

With no model key either, you can still create the demo workspace and browse the deals, the Memory evidence tab, the Memory page and the self-check; reading a new deal and writing a brief need a model key.

## What needs a model call, and roughly how many

| Action | Model calls |
|---|---|
| Create the demo workspace (seeding) | 0 |
| Memory evidence tab (including the Reflect panel's cached text), brief history and diff, Memory page, self-check, status | 0 |
| Read a new deal (**Read this deal**, uploaded notes) | 1 |
| Generate a brief | 1 (2 if the first answer names no valid next step and is asked again) |
| Compare memory on a deal | 3 (one brief per arm; the deal must have been read first) |
| Record an outcome | 0 |

This script uses 7 model calls: two briefs on the pricing deal (No history, Hindsight memory), two Hindsight briefs on Cedarline, and the three-arm comparison. The buttons state the cost before you confirm.

Hindsight Cloud usage is separate from the model. Creating the demo workspace sends about 53 lessons to the lessons bank in 3 batches (19 outcome lessons and 34 play lessons; this spends Hindsight credits), and 48 verbatim documents (19 closed-deal summaries and 29 interactions) go to the interactions bank, which uses no Hindsight LLM. Recording Larkfield adds 2 lessons. Refreshing the Reflect panel asks the lessons bank once and spends a little credit. Building or refreshing the playbook spends further credits and only runs when you confirm. On the local backend none of this is sent anywhere.

## Token-conscious testing

- Seeding, the evidence tab, the diff, the Memory page and the self-check make no model calls.
- Automated tests use fake model and Hindsight clients (or the local memory) and consume no tokens or credits.
- For a rehearsal, use **No history** or **Hindsight memory** briefs one at a time rather than the comparison.
- A demo workspace is reset by creating a new one; each gets fresh banks (or a fresh `memory.db`) and the old ones can be deleted from the workspace menu.
