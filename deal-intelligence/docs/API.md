# API contract (v1)

Same-origin JSON API served by FastAPI on `http://127.0.0.1:8002`. Errors are always
`{"error": {"code": str, "message": str}}` with a meaningful HTTP status. Jobs are polled with
`GET /v1/jobs/{id}` every second until `status` is `completed`, `failed` or `cancelled`.
`mode` is one of `none | longctx | similar | hindsight` (UI labels: "No history", "All deals in the prompt",
"Similar deals", "Hindsight memory").

## Workspaces and model (served by `main.py`)

**Per-request workspace.** Every `/v1/...` route that works on a workspace's data accepts an `X-Workspace: <workspace id>` header. The request then runs in that workspace without switching the active one, so the app's own screens are not affected and two callers can work in two workspaces at once (the Agent Hub does this, one workspace per chat). Without the header the active workspace is used, as before. An unknown id is 404 `not_found`. The first request for a non-active workspace opens it (its database, memory banks and sync) and keeps it open until the app stops, the workspace is activated or deleted; activating or deleting is refused with 409 `workspace_busy` while one of its jobs is running.

| Route | Notes |
|---|---|
| `GET /v1/workspaces` | `{active, workspaces:[{id,name,kind,created_at,active}]}` |
| `POST /v1/workspaces` | body `{name?, kind: "demo"\|"company"}`; kind=demo seeds Halcyon Software with zero model calls, returns `{workspace, seeded:{plays,deals,closed,open,interactions,already_there,demo_deal,closable_deal}}` (the current seed: 10 plays, 24 deals, 19 closed, 5 open, 101 interactions, `demo_deal` D-015, `closable_deal` D-018); 409 `workspace_busy` while a job runs |
| `POST /v1/workspaces/{id}/activate`, `DELETE /v1/workspaces/{id}` | return the same shape as the list |
| `GET /v1/models`, `PUT /v1/models` | `{provider,current,source,env_model,default,options:[{id,label,input_token_limit}],list_error}`; PUT body `{model: str\|null}` |
| `GET /v1/samples/{name}` | bundled upload samples |
| `GET /health` | |

## Status and sync

- `GET /v1/status` ->
  `{model:{provider,name,credentials,key_variable,health:null|{reason,title,detail,action,retry_helps,switch_model,blocking}},
    retrieval_mode,
    hindsight:{url,cloud:bool,api_key_set:bool,bank,lessons_bank,healthy:bool,mode:"chunks"|...|"local"|null,backend:"hindsight"|"local"},
    lessons:{enabled,pending,retained,failed,last_error},
    counts:{open,closed,won,lost,interactions,plays},
    sync:{pending}}`.
  `hindsight.backend` is the memory backend in use (`DEAL_MEMORY_BACKEND`: `auto` | `hindsight` | `local`; `auto` is local when there is no Hindsight key and the URL is Hindsight Cloud). On `local`, `cloud` is false, `healthy` is true, `mode` is `"local"`, and no Hindsight call is made.
- `POST /v1/sync` -> `{retained,deleted,failed,pending,lessons:{retained,failed,pending}|null}`

## Deals

- `GET /v1/deals` -> `{deals:[Deal]}`
- `POST /v1/deals` (multipart: `name`, `account`, `industry?`, `segment?` (smb|mid_market|enterprise), `amount?`, `owner?`, `stage?`, `files[]?` of .txt/.md/.eml/.csv/.pdf/.docx/.xlsx) -> 201 `DealDetail`
- `GET /v1/deals/{id}` -> `DealDetail`; `DELETE /v1/deals/{id}` -> `{ok:true}`
- `POST /v1/deals/{id}/files` (multipart `files[]`) -> `DealDetail`
- `POST /v1/deals/{id}/notes` body `{kind: email|call_note|meeting|crm_note, text, occurred_on?, author?, subject?}` -> `DealDetail`
- `POST /v1/deals/{id}/signals` -> 202 `{job_id}` (reads objections, competitors, promises and stakeholders from the interactions with one model call; 409 when the deal has no interactions)
- `GET /v1/plays` -> `{plays:[{code,name,description,category,addresses}]}`

`Deal` = `{id,code:"D-001",name,account,industry,segment,amount,stage,owner,opened_on,closed_on,result:"open"|"won"|"lost",loss_reason,signals_status:"none"|"extracting"|"ready"|"failed",signals_error,interactions:int,last_activity}`.

`DealDetail` = `Deal` plus `stakeholders:[{name,title,stance,engaged,economic_buyer,note}]`,
`interactions:[{id,code:"INT-0001",kind,occurred_on,author,subject,text,hindsight_status}]`,
`signals: null | {objections:[{type,text,status,first_seen_on,evidence:[INT]}], competitors:[str], pricing:{discount_requested,notes}, promises:[{text,owner,due_on,status,evidence}], source, plays_used:[{code,name}]}`,
`flags:[Flag]` (deterministic, no model: `{code,severity:"high"|"medium"|"low",text,evidence:[...]}`).

## Briefs and memory evidence

- `POST /v1/deals/{id}/brief` body `{mode?}` -> 202 `{job_id}`. 409 `signals_missing` when the deal has no signals yet (run `/signals` first).
- `POST /v1/deals/{id}/ask` body `{question}` (1 to 500 characters) -> `{deal_id, question, found, answer, sources:[INT-|D-], grounded, findings, model, mode, degraded}`. One model call, nothing stored. The answer comes from this deal's emails and notes plus the similar closed deals memory recalls; citations are checked the way the brief's are (an unknown id is dropped and reported in `findings`). `found:false` means the record does not answer it; `grounded:false` means it cited nothing valid. 422 for an empty or long question, 404 for an unknown deal, 502 `model_error` when the model fails.
- `POST /v1/portfolio/ask` body `{question}` -> `{question, found, answer, sources:[D-], grounded, findings, stats, deals_considered, deals_total, truncated, model}`. One model call, nothing stored. A question about the whole pipeline ("why do we lose fintech deals?"). The counts in `stats` (deals by result, win rate, loss reasons, result counts by industry, segment, objection type and competitor, play usage) are computed in code over every deal; the model is shown them plus up to 80 deal summaries (closed deals first, the most recent when there are more, `truncated` says so) and may only cite those D- ids. 409 `no_deals` for an empty workspace; 422 and 502 as for `/ask`.
- `POST /v1/deals/{id}/followup` body `{kind?: "email"|"call_agenda", play_code?}` -> `{deal_id, kind, subject, body, sources:[INT-], grounded, findings, play_code, step_name, model}`. One model call, nothing stored or sent. Drafts the latest brief's top next step (or the one named by `play_code`) using only facts in the deal; unknown details are left as `[bracketed]` placeholders and it never offers a price, discount or date the record does not state. 409 `no_brief` (write the brief first), 409 `no_next_step`, 409 `deal_closed`.
- `GET /v1/deals/{id}/brief?mode=` -> latest `BriefView`, or 404 `no_brief`. `GET /v1/deals/{id}/briefs` -> `{briefs:[BriefView without content]}` newest first (id, mode, memory_state, created_at, n_similar).
- `GET /v1/deals/{id}/recommendations?mode=` -> `Recommendations` (no model call; the "what memory says" evidence panel): 
  `{deal_id,mode,n_closed,degraded,lessons_used,memory_state,
    similar:[{code,account,result,loss_reason,rank,relevance,shared_keys,plays_used,summary}],
    plays:[{play_code,name,used_in_similar,won_in_similar,lost_quality_in_similar,source_deals,lesson_signal,reasons:[str]}],
    warnings:[{kind,text,objection_type,similar_lost,similar_total,source_deals}],
    avoid:[{play_code,name,used_in_similar,won_in_similar,lost_in_similar,source_deals,text}]}`.
  `avoid` ("Likely to backfire") lists plays used in at least 3 similar closed deals (the same gated set the warnings use) and in none of the similar won ones, with `won_in_similar` always 0 and honest counts in `text`, for example `"Discount offer: used in 4 similar deals, 0 won."`. Listed plays are never also in `plays`. Only modes `similar` and `hindsight` fill it; it is empty for `none` and `longctx`. It is part of `memory_state`.
- `BriefView` = `{id,deal_id,mode,status,created_at,model,prompt_version,prompt_hash,memory_state,input_tokens,output_tokens,error,flags:[finding],evidence:Recommendations,
  content:{summary,summary_sources:[INT],this_deal:[{text,source_ids}],memory:[{text,source_ids:[D-]}],
  next_steps:[{play_code,name,rationale,source_ids,counts,reasons:[str]}],missing_info:[str],flags:[Flag],
  warnings:[Warning],avoid:[AvoidPlay],similar:[D-],mode,n_closed,degraded}}`.
  `prompt_version` is `brief-v2`. `flags` are the code-checked findings, each `{code,where,detail}`; `avoided_play` means the model named a play in `avoid` and the step was dropped. `avoid` is stored in the content whatever the model said. The avoid list is sent to the model inside the evidence block, so `prompt_hash` (system prompt plus everything but the evidence) is unchanged by it and stays identical across modes.
  Citations are `INT-xxxx` (this deal) and `D-xxx` (similar closed deals); the UI renders each as a chip that opens the interaction or the closed-deal summary. Lessons are never cited.
- `GET /v1/deals/{id}/memory-says` -> `{state:"missing"|"ready", text:str|null, based_on:[{id,text,type}], created_at:str|null, backend:"hindsight"|"local"}`. The cached Reflect text from the lessons bank about deals like this one (stored in the `deal_reflections` table). Free; `missing` until someone asks. 404 `not_found` for an unknown deal.
- `POST /v1/deals/{id}/memory-says` -> same shape, refreshed: syncs pending lessons, asks the lessons bank again, and stores the answer. Spends Hindsight credits on the Hindsight backend; free on `local`, where the text is computed deterministically from recorded outcomes (counts, no invented claims). Errors: 409 `lessons_disabled` (`DEAL_LESSONS` off), 503 `hindsight_unavailable`, 404 `not_found`.

## Outcome (the learning loop)

`PUT /v1/deals/{id}/outcome` body `{result:"won"|"lost", loss_reason?, plays_used?:[PLAY-xx], closed_on?}` (omit `plays_used` to keep the plays already recorded on the deal; send `[]` to record that none were used) ->
`{deal:Deal, credit:[{play_code,delta,times_used,won,lost_quality,lost_other}], events:[str], lessons_added:int, memory_note:str}`.
Loss reasons: competitor, price, no_decision, champion_left, security_compliance, feature_gap, timing, unresolved_objection (price, no-decision, timing, champion_left and plain competitor losses give plays no credit).

## Before / after

- `POST /v1/deals/{id}/comparison` body `{arms?:["none","longctx","hindsight"]}` -> 202 `{job_id, planned_calls}`
- `GET /v1/deals/{id}/comparison` -> `{job:{id,status,...}|null, n_closed, arms:{none:BriefView, longctx:BriefView, hindsight:BriefView}, prompt_hash_same:bool, differs:{play_codes:{none:[],longctx:[],hindsight:[]}, same_play_as_longctx:bool, cited_deals:[D-], avoided_play_recommended_by:[arm], tokens:{none:{input,output},longctx:{...},hindsight:{...}}}}` (arms missing until done; the UI states n and that at this size stuffing every summary also works, so memory's claim is targeted retrieval, counted evidence, citations and scale). `avoided_play_recommended_by` lists the arms (`none`, `longctx`, `hindsight`) whose next steps include a play in the `hindsight` arm's `avoid` list; it is empty when there is no `hindsight` arm or nothing to avoid.
- `GET /v1/deals/{id}/brief-diff?from=<briefId>&to=<briefId>` -> `{from,to,added_warnings:[...], removed_warnings:[...], changed_warnings:[{kind,objection_type,from:{similar_lost,similar_total},to:{...}}], added_similar:[D-], removed_similar:[D-], added_plays:[...], removed_plays:[...], changed_plays:[{play_code,from:{used,won},to:{used,won}}], added_avoid:[AvoidPlay], removed_avoid:[AvoidPlay], changed_avoid:[{play_code,from:{used,won},to:{used,won}}], memory_state:{from,to}}` (the T0/T1 within-subject diff; deterministic, no model call). The `*_avoid` fields report plays that newly became, stopped being, or changed counts as "likely to backfire".

## Memory page

- `GET /v1/memory/journal` -> `{events:[{id,kind,deal_code,play_code,detail,created_at}]}`
- `GET /v1/memory/lessons` -> `{lessons:[{key,signal,text,tags,deal_code,hindsight_status,happened_at}], counts:{pending,retained,failed}}`
- `GET /v1/memory/play-stats` -> `{plays:[{code,name,category,times_used,won,lost_quality,lost_other,outcome_credit}]}`
- `GET /v1/memory/history` -> `{deals:[Deal + {summary, plays_used:[{code,name}], objections:[{type,status}]}]}` (closed deals, newest first)
- `GET /v1/memory/playbook` -> `{content: str|null, last_refreshed_at, is_stale, state:"missing"|"ready"|"unavailable"}`; `POST /v1/memory/playbook/refresh` -> same (spends Hindsight credits on the Hindsight backend; explicit only). On the `local` backend the playbook is computed from the stored lessons on every read, so `state` is `ready`, `is_stale` is false, refreshing is free, and the text ends with "Computed locally from recorded outcomes, not by Hindsight."
- `GET /v1/memory/quality` -> `{n_closed, top:3, won:{n,covered,hit}, lost:{n,warned,had_unresolved,avoid_hit}, method, caveat}`. The leave-one-out self-check: each closed deal is left out in turn and the recommendations are rebuilt from the others using only SQLite (`source="sqlite"`: no model, no Hindsight, no lessons bank). `won.covered` is the won deals that had at least one recommended play and `won.hit` those where a play from the top `top` was really used. `lost.had_unresolved` is the lost deals with an unresolved objection, `lost.warned` those where a warning named it, and `lost.avoid_hit` the lost deals where a play flagged as backfiring was really used. The seed (19 closed deals) gives won `{n:9,covered:7,hit:5}` and lost `{n:10,warned:6,had_unresolved:8,avoid_hit:4}`. The sample is small and `caveat` says so; treat it as a sanity check, not a benchmark.

## Jobs

`GET /v1/jobs/{id}` -> `{id,kind,target_id,status,done,total,error,warning,payload,error_info:null|{...explain...}}`; `POST /v1/jobs/{id}/cancel`.
