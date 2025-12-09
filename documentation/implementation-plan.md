Here’s how I’d spell out the agent system Matt is implicitly on the hook to build, based on what everyone said on this call.

---

## 1. North Star: What this agent is supposed to do

**Goal:**
Give principals and superintendents the feeling of having a super-analyst who has read *every* evaluation they have access to, and can:

* Answer complex questions across **all teachers / all schools / entire districts**
* Surface **early warning signals** (red/yellow/green) by domain
* Recommend **PD moves** and **strategic paths forward**
* Provide **board-ready stories and soundbites**

…despite the fact that:

* Evaluations are huge, text-heavy, and numerous
* LLMs have context limits and can’t just “read everything at once”

So the system has to **decompose big questions**, send them down into smaller “micro-agents” that each look at a slice of the data, and then **re-aggregate** into coherent, role-appropriate answers.

---

## 2. Key users & core questions

### 2.1 Principals

Typical questions:

* “Who in my school is struggling in Domain II-B right now?”
* “Where do I see the biggest growth in classroom management over the last year?”
* “Which teachers need targeted coaching in questioning & discussion techniques?”
* “Help me turn this into a PD plan for January.”

Needs:

* **Teacher-level and school-level** insights
* Grouping teachers for PD (“show me all yellow in I-A”)
* Concrete examples and scripts they can share with staff

### 2.2 Superintendents / ESC leaders

Typical questions (Jesse & Will’s comments):

* “Across all of my schools, who’s out of green in each domain? Show me red/yellow/green.”
* “Where do I have **150 people in yellow** across multiple schools?”
* “What PD should we be providing across the system?”
* “What stories can I tell my board to celebrate strengths or acknowledge blind spots?”

Needs:

* **Summative, cross-school** patterns
* Early warning and **risk flags** (burnout, potential churn, blind spots)
* Few strong **anecdotes & soundbites** that are board-ready

---

## 3. Data inputs the agent works over

Structurally, the system will be sitting on top of what you already store in SwiftScore:

* **Evaluations**

  * Domain/component scores
  * Overall ratings
  * Timestamps (so we can see change over time)
* **Text fields**

  * Low-inference notes
  * Evidence snippets
  * Feedback / next steps
* **People + org graph**

  * Teachers (with IDs, schools, subjects, etc.)
  * Principals, superintendents, ESCs, districts
  * Who reports to whom / access control (which evals a given user can see)

The agent will not load “all text at once”; instead it will:

1. **Query structured data** (scores, domains) to narrow where to look.
2. **Retrieve relevant text** for a teacher/domain/time window.
3. Use micro-agents to read small chunks and summarize.

---

## 4. Overall agent architecture

Think of it as a **4-level hierarchy** that mirrors the organization:

> **District/ESC Agent** → **School Agent** → **Teacher Agent** → **Evaluation Agent**

With one extra piece at the top:

> **Chat Orchestrator (front-end agent)**

### 4.1 Top-level: Chat Orchestrator

Responsibilities:

1. **Identify the user & role**

   * Is this a principal? A superintendent? Director at an ESC?
   * Use permissions to decide what data they can see.

2. **Interpret the question**

   * “Everybody in these domains who’s out of green, yellow, or red?”
   * “Where are my biggest strengths in Domain II-B this semester?”
   * “Give me board-ready talking points about Domain III.”

3. **Plan the work**

   * Decide which **hierarchical path** to trigger:

     * For a principal: mostly **School → Teacher → Evaluation**
     * For a superintendent: mostly **District → School → Teacher → Evaluation**
   * Decide **which domains** and **which time window** to query.

4. **Call the underlying agents**

   * “For each relevant school, run a SchoolAgent with this question, then combine.”
   * “For each teacher in this school in Domain II-B, run TeacherAgents, etc.”

5. **Compose the final answer**

   * Natural language explanation
   * Optional tables / charts (red/yellow/green counts)
   * PD recommendations, plus 1–3 stories / soundbites

---

## 5. The hierarchical agent layers

### 5.1 EvaluationAgent (bottom layer)

**Scope:** A single evaluation for a single teacher.

**Inputs:**

* Evaluation ID
* Domain(s)/components of interest
* The top-level question (“How is this teacher doing in Domain II-B? Evidence for growth vs. concern?”)

**Process:**

1. Load **scores and rubric info** for that evaluation/domain.
2. Retrieve **low-inference notes & feedback** for that evaluation/domain (chunked to fit context).
3. Answer a **small, focused prompt**, e.g.:

> “For this evaluation and this domain:
> – What did the teacher do in this domain?
> – Is there evidence of strength, growth, stagnation, or decline?
> – Does the text suggest risk factors (burnout, disengagement, misalignment)?
> – Extract 1–2 short evidence quotes.”

**Outputs (structured JSON-ish object):**

* `teacher_id`
* `school_id`
* `evaluation_id`
* `date`
* `domain_id`
* `score`
* `status_color` (e.g., green/yellow/red based on score & rubric thresholds)
* `trend_hint` (improving / stable / declining if we compare to previous evals later)
* `key_evidence_quotes` (short text snippets)
* `flags` (e.g., “needs_PD”, “risk_of_leaving”, “exemplar_practice”)

These are **small**, composable outputs designed for further aggregation.

---

### 5.2 TeacherAgent

**Scope:** All relevant evaluations for a single teacher (over time).

**Inputs:**

* Teacher ID
* All `EvaluationAgent` outputs for that teacher in the selected time window
* The user’s question

**Process:**

1. Aggregate the evaluation-level results:

   * Combine scores per domain over time
   * Look at `status_color` patterns (how often green vs yellow vs red)
   * Infer **trend** in each domain

2. Build a **teacher profile** for the question:

   * Strength domains (sustained green; upward trends)
   * Concern domains (persistent yellow/red; downward trends)
   * Extract 1–2 **representative evidence quotes** per domain (from evaluation-level outputs)

3. Mark **PD + risk**:

   * Suggested PD focus areas (e.g., “Domain II-B: questioning strategies”)
   * Any risk-related flags if patterns suggest issues (burnout, disengagement, non-growth)

**Outputs:**

* `teacher_id`
* `school_id`
* `per_domain_summary`:

  * color (green/yellow/red)
  * trend (up/down/stable)
  * 1–2 bullet points on what’s going on
* `recommended_PD_topics`
* `overall_risk_level` (low/med/high)
* `highlight_story` (short narrative + quote for use in PD or board-level stories)

---

### 5.3 SchoolAgent

**Scope:** All relevant teachers in a school.

**Inputs:**

* School ID
* All `TeacherAgent` outputs for that school
* The user’s question (from a principal or superintendent)

**Process:**

1. Compute **school-wide metrics**:

   * For each domain:

     * `#teachers_green`, `#teachers_yellow`, `#teachers_red`
     * Percentages / trends over time
   * Identify **concentration of yellow/red** (e.g., “15 teachers yellow in Domain II-B”)

2. Build **groupings for PD**:

   * Lists of teachers by status per domain (“everyone red in II-B”, “everyone yellow in I-A”)
   * Optional slicing by grade level, subject, etc.

3. Extract **stories**:

   * 2–3 exemplar teachers (sustained green with great quotes) for celebration
   * 2–3 anonymized patterns that reveal a blind spot or concern

**Outputs:**

* `school_id`
* `domain_level_stats`:

  * counts + percentages by green/yellow/red
  * mini trend notes
* `PD_cohorts`:

  * Domain → list of teacher IDs to group
* `school_highlights`:

  * Stories that could be shared with superintendent / staff
* `school_concerns`:

  * Where the school is “red/yellow heavy”

For a **principal**, the Chat Orchestrator will mostly use this layer’s outputs, plus drill-down teacher lists.

---

### 5.4 District/ESC Agent

**Scope:** All schools in a district or ESC.

**Inputs:**

* District/ESC ID
* All `SchoolAgent` outputs for their schools
* Superintendent-level questions

**Process:**

1. **Compare schools**:

   * For each domain, show which schools are strongest/weakest
   * Show where there are **large clusters of yellow/red** (“150 yellow in Domain II-B across 7 schools”)

2. Build **system-level early warning**:

   * Domains with systemwide risk (lots of yellow/red)
   * Schools that are outliers (very low or very high performance)

3. Translate into **strategy**:

   * System-level PD priorities (“district-wide push in Domain II-B questioning”)
   * Suggestions for **where to pilot RocketPD programs** or specific courses
   * Ties to strategic plan and board-level outcomes (as Will described)

4. Extract **board-ready stories**:

   * At least one strong “we’re killing it here” story
   * At least one “this is our blind spot; here’s our plan” story

**Outputs:**

* `district_summary`:

  * per-domain red/yellow/green counts across schools
  * ranking of schools by domain
* `priority_domains`
* `system_PD_recommendations`
* `board_story_highlights` (short, quotable narratives)

For a **superintendent**, the Chat Orchestrator leans heavily on this layer.

---

## 6. Early-warning system & red/yellow/green

Jesse’s “out of green / yellow / red” framing turns into a core concept:

* Define **thresholds per domain** (configurable via org settings):

  * e.g., a numeric mapping of performance levels → color
* At each level (teacher, school, district), maintain:

  * Counts and percentages by color
  * Time trends (is the proportion of yellow in II-B going up or down?)

This enables questions like:

* “Show me everyone in yellow or red in Domain II-B in my region.”
* “Which schools have more than 25% of teachers yellow+red in Domain III?”
* “Where did red → yellow or yellow → green over the last semester?”

The agents use this both:

* **On-demand** (when answering a question), and
* **Precomputed** for dashboards / monitoring (e.g., nightly jobs that run TeacherAgent/SchoolAgent summaries).

---

## 7. Role behavior: principal vs superintendent

Matt’s idea: **change how it thinks**, not just what data it sees.

Will’s push: you may not need radically different logic, but you do need different **framing and granularity**.

So, concretely:

### For principals

* **Access scope:** their school (and maybe feeder teachers)
* **Tone:** coaching & operational, less board-speak
* **Answer style:**

  * More teacher-specific lists (“here are the 9 teachers yellow in II-B”)
  * Detailed PD suggestions (“run a 3-session cycle on formative questioning; here’s an outline”)
  * Direct scripts (“Here’s how you could explain this to your superintendent”)

### For superintendents / ESC leaders

* **Access scope:** all schools under their purview
* **Tone:** strategic & political, board-ready
* **Answer style:**

  * Cross-school comparisons, heatmaps, and trends
  * System-level PD priorities
  * Storytelling: “Despite perceptions, we’re actually strong in X; our blind spot is Y; here’s how we’ll address it.”

The **Orchestrator** picks which agent layer’s output dominates (School vs District) and tunes the prompt style accordingly.

---

## 8. How this deals with LLM context limits

The entire architecture is a response to:

> “These evaluations are huge. They take up a lot of words.”

Concretely:

1. **Never** ask the LLM to read “all evaluations.”

2. Do **fan-out / fan-in**:

   * Fan-out: many small EvaluationAgent calls, each over a single eval or small subset of text.
   * Fan-in: Teacher/School/District agents only consume small structured outputs or short summaries.

3. **Cache** results:

   * If an evaluation hasn’t changed, its EvaluationAgent output is reusable.
   * TeacherAgent outputs can be cached per time-window and invalidated when new evals arrive.

4. **Sampling & prioritization** when needed:

   * For very large districts, you can sample teachers or limit to “recent year” first, unless user explicitly asks for multi-year patterns.

---

## 9. Surfaces: chatbot + reporting

### 9.1 Chatbot

User types a question → Orchestrator calls the agent hierarchy → reply includes:

* Narrative answer
* Optional: table or chart (counts by color)
* Buttons/links like:

  * “View list of teachers in this PD cohort”
  * “Generate a PD plan outline for these teachers”
  * “Generate board meeting talking points”

### 9.2 Reporting / dashboards

Standard “saved questions” wired into UI:

* “Early Warning” page (red/yellow/green heatmap by domain & school)
* “PD Planning” page (PD cohorts by domain + suggested topics)
* “Board Stories” export (narrative + data visual)

Under the hood, these are just pre-baked prompts to the same agents.

---

## 10. How this generalizes to internal use (pipeline/CRM)

Matt mentioned using this **internally as well**. The same pattern can be reused:

* **EvaluationAgent → DealAgent** (per deal)
* **TeacherAgent → DistrictAgent** (per district in pipeline)
* **SchoolAgent → ChannelAgent** (per acquisition channel)
* **DistrictAgent → RocketPDStrategyAgent** (system view of pipeline)

So you can run the same hierarchical pattern over deals instead of evaluations to:

* Diagnose which channels are working
* Understand CAC by source
* Suggest where to invest more time/events/email

But the primary thing this call is pointing him at is the **evaluation hierarchy** above.

---

If you’d like, I can next:

* Turn this into a one-pager spec for Matt (“Agent System v1.0”)
* Or write concrete prompt schemas / JSON output contracts for each agent layer so he can just start implementing.




Exactly, you’re describing a **hierarchical map–reduce agent** over your org tree, where:

* **Leaves = individual evaluations**
* **Internal nodes = teacher → principal/school → superintendent/district**
* Each node:

  * **Consumes** structured summaries from children
  * **Summarizes** them with its own LLM call
  * **Bubbles up** a higher-level summary

Let me spell out the system in a way Matt can basically implement.

---

## 1. Mental model: map–reduce over the evaluation tree

You can think of a single query as:

1. **Map step:**

   * For *every evaluation* the user is allowed to see, run a small LLM program:

     > “Given this evaluation and this question, extract/summarize X.”

2. **Reduce steps (hierarchical):**

   * **Teacher-level reduce:** aggregate all evaluation outputs for each teacher.
   * **Principal/school-level reduce:** aggregate all teacher-level outputs per school.
   * **Superintendent/district-level reduce:** aggregate all school-level outputs.
   * At each level, another LLM call turns many small structured items into a compressed summary.

3. **Final answer:**

   * The top-level agent (chatbot) takes the *district or school summary* + original question and generates the human-facing response.

No retrieval to decide *which* evaluations to hit:

> “When we ask a question, we always hit *every* evaluation this user has access to.”

Retrieval only matters inside each evaluation if you need to chunk the text.

---

## 2. Data model for the tree

At runtime, for a given user + question:

### 2.1 Org tree for a principal

For a **principal**:

```text
Principal (user)
 └── Teachers (T1, T2, ..., Tn)
      └── Evaluations (E1, E2, ..., Ek)
```

Scope = all evaluations whose `teacher.school_id == principal.school_id`
(and that permissions allow).

### 2.2 Org tree for a superintendent / ESC leader

For a **superintendent**:

```text
Superintendent (user)
 └── Principals / Schools (S1, S2, ..., Sm)
      └── Teachers (per school)
           └── Evaluations
```

Scope = all evaluations whose `school_id` is in the superintendent’s org graph.

This is built from your DB, not with RAG.

---

## 3. Core pipeline: from question to answer

Let’s define a clean pipeline that works for both principals and superintendents.

### Step 0: Identify user and scope

* Determine `role` (principal vs superintendent vs ESC).
* Build the **tree**: for this user, list:

  * all schools in scope
  * all teachers in those schools
  * all evaluations for those teachers (with IDs + metadata)

This is pure SQL / application logic.

---

### Step 1: Evaluation-level LLM calls (leaf mapper)

For each evaluation `E` in scope, we run the same “leaf agent” with parameters:

* `question`: whatever the user asked
* `eval_metadata`: teacher, school, date, domain scores
* `eval_text`: observation notes, feedback, etc. (maybe chunked)

**Prompt pattern (conceptual):**

> You are an evaluation summarizer.
>
> * You see: metadata + text for ONE evaluation of ONE teacher.
>
> * You also see the user’s question: `<QUESTION>`.
>
> 1. Extract information **relevant to the question** for this teacher in this evaluation.
> 2. Ignore unrelated domains/components.
> 3. Output ONLY JSON with these fields:
>
>    * `teacher_id`
>    * `school_id`
>    * `evaluation_id`
>    * `date`
>    * `per_domain` (dictionary)
>    * `flags`
>    * `evidence_snippets`

**Example output schema:**

```json
{
  "teacher_id": "uuid-teacher-1",
  "school_id": "uuid-school-1",
  "evaluation_id": "uuid-eval-1",
  "date": "2025-10-12",
  "per_domain": {
    "II-B": {
      "score": 2,
      "color": "yellow",
      "relevance_to_question": "high",
      "summary": "Teacher used mostly closed questions; limited student discussion.",
      "growth_signals": ["Some use of wait time; students occasionally discussed in pairs."]
    }
  },
  "flags": {
    "needs_PD": ["II-B"],
    "exemplar": false,
    "risk_of_leaving": false
  },
  "evidence_snippets": [
    "Teacher asked yes/no questions for most of the lesson; few students spoke at length."
  ]
}
```

Implementation points:

* **Parallelize heavily**: these are independent LLM calls.
* **Batching**: depending on provider, you can send multiple evals in one call and get multiple JSONs back; or just fan out.

---

### Step 2: Teacher-level aggregation (first reduce)

Now group all evaluation outputs by `teacher_id`:

```python
teacher_to_eval_summaries = {
    teacher_id: [eval_summary_1, eval_summary_2, ...]
}
```

You can do **two layers** here:

1. A **lightweight deterministic aggregation** (no LLM) to compute:

   * average score per domain
   * counts of green/yellow/red per domain
   * recent trend hints (e.g., last 3 evals)

2. A **Teacher-level LLM call** to turn that + the question into a narrative summary.

**Teacher LLM prompt pattern:**

> You are summarizing ONE teacher for a principal/superintendent.
>
> * You see this teacher’s per-evaluation summaries (JSON list).
> * You see the user’s question: `<QUESTION>`.
>
> Produce a JSON summary with:
>
> * `teacher_id`
> * `per_domain_overview`: color, trend, 1–2 bullet points
> * `recommended_PD_focus` (list of topics)
> * `notable_evidence` (up to 3 short quotes)
> * `risk_level` ("low" | "medium" | "high") if relevant
> * `overall_short_summary` (2–3 sentences tailored to the question)

**Teacher summary schema:**

```json
{
  "teacher_id": "uuid-teacher-1",
  "per_domain_overview": {
    "II-B": {
      "color": "yellow",
      "trend": "improving",
      "bullets": [
        "Questioning is still mostly teacher-directed.",
        "More peer discussion visible in most recent observation."
      ]
    }
  },
  "recommended_PD_focus": ["Questioning and discussion strategies (Domain II-B)"],
  "notable_evidence": [
    "Students talked in pairs for 5 minutes about the central question.",
    "Only 3 students spoke during whole-group discussion."
  ],
  "risk_level": "medium",
  "overall_short_summary": "This teacher is making progress in questioning but remains below proficient in Domain II-B; targeted PD could move them to green."
}
```

Again, highly parallelizable: one LLM call per teacher.

---

### Step 3: Principal/school-level aggregation

For a **principal**, this is often your **top level**; for a superintendent, it’s an intermediate level.

Group teacher summaries by `school_id`:

```python
school_to_teacher_summaries = {
    school_id: [teacher_summary_1, teacher_summary_2, ...]
}
```

You can again do:

1. **Pre-aggregation (no LLM):**

   * For each domain:

     * `#teachers_green/yellow/red`
     * Percentages
   * Build lists:

     * `teachers_needing_PD_in_domain_X`
     * `exemplar_teachers_in_domain_Y`

2. **School-level LLM call**:

> You are summarizing ONE school for a principal or superintendent.
>
> * You see teacher-level summaries and precomputed stats.
> * You see the question: `<QUESTION>`.
>
> Produce JSON with:
>
> * `school_id`
> * `domain_stats`: counts and percents by green/yellow/red
> * `PD_cohorts`: domain → list of teacher_ids to group
> * `school_strengths`: 2–3 bullets
> * `school_needs`: 2–3 bullets
> * `stories_for_supervisor_or_board`: 1–3 short narratives tied to the question

**School summary schema:**

```json
{
  "school_id": "uuid-school-1",
  "domain_stats": {
    "II-B": { "green": 5, "yellow": 12, "red": 3 },
    "III-C": { "green": 10, "yellow": 7, "red": 3 }
  },
  "PD_cohorts": {
    "II-B": ["teacher-1", "teacher-7", "teacher-12"],
    "III-C": ["teacher-3", "teacher-9"]
  },
  "school_strengths": [
    "Strong performance in Domain III-C (student engagement) with 10 teachers in green.",
    "Multiple exemplar classrooms that model student-led discussion."
  ],
  "school_needs": [
    "High concentration of yellow in Domain II-B; questioning skills are a systemic growth area."
  ],
  "stories_for_supervisor_or_board": [
    "Although there is concern about questioning, 3 classrooms show exemplary discussion practices that could anchor PD."
  ]
}
```

For **principals**, this school-level summary is what the chat agent mostly uses.
It can then drill down into teacher lists as needed.

---

### Step 4: Superintendent / district-level aggregation

For superintendents or ESC leaders, now you aggregate across schools:

```python
district_school_summaries = [school_summary_s1, school_summary_s2, ...]
```

Again, do:

1. **Pre-aggregation:**

   * Per domain:

     * total green/yellow/red across all schools
   * Per school:

     * risk index, strengths index

2. **District-level LLM call:**

> You are summarizing a full district/region for a superintendent.
>
> * You see school-level summaries and global stats.
> * You see the question: `<QUESTION>`.
>
> Produce JSON with:
>
> * `priority_domains` (systemwide focus areas)
> * `district_strengths` (2–3 bullets)
> * `district_needs` (2–3 bullets)
> * `school_rankings_by_domain` (simple ordered lists)
> * `board_ready_stories` (2–4 short narratives with evidence)
> * `recommended_PD_strategy` (1–3 actionable moves)

**District summary schema:**

```json
{
  "priority_domains": ["II-B", "I-A"],
  "district_strengths": [
    "Strong performance in Domain III-C across most schools.",
    "Two schools with exemplary practice in Domain II-B that could serve as PD hubs."
  ],
  "district_needs": [
    "System-wide concern in Domain II-B, with 150 teachers in yellow and 40 in red.",
    "Uneven implementation of feedback cycles in Domain I-A."
  ],
  "school_rankings_by_domain": {
    "II-B": ["school-7", "school-3", "school-1", "..."]
  },
  "board_ready_stories": [
    "Despite perceptions, the district is excelling in student engagement (III-C), with 68% of teachers in green.",
    "Our blind spot is questioning (II-B): 190 teachers are not yet at green; we are launching a targeted PD initiative focused on discussion protocols."
  ],
  "recommended_PD_strategy": [
    "Launch a cross-school learning community focused on questioning strategies, anchored by School 7.",
    "Use exemplars from School 3 and School 7 to design model lessons and coaching cycles."
  ]
}
```

---

### Step 5: Final user-facing LLM call (Chat Orchestrator)

Now the top-level chatbot has:

* Original `QUESTION`
* User role and permissions
* Either:

  * A **school summary** (principal)
  * Or a **district summary** (superintendent)

It runs one final LLM call to convert that JSON into:

* A natural language answer
* Possibly a table/structured view for the UI
* Follow-up actions (PD cohort creation, exports, etc.)

Prompt idea:

> You are a strategic assistant for `<ROLE>` (principal/superintendent).
> You see a summarized view of their evaluations data (JSON).
> Answer their question clearly.
>
> 1. First, directly answer in 2–3 paragraphs.
> 2. Then, if helpful, show a bullet list of recommended actions (e.g., PD, reports, board talking points).
> 3. Keep language appropriate to the role (board-level for superintendents, coaching-level for principals).

---

## 4. How recursion works conceptually

For a **superintendent question**, the recursion looks like:

1. `Question` → `EvaluationAgent` over all evals in their schools.
2. Group by teacher → `TeacherAgent` per teacher.
3. Group by school → `SchoolAgent` per school.
4. Aggregate schools → `DistrictAgent`.
5. `DistrictAgent` + `Question` → final Chat response.

For a **principal question**, you can skip step 4:

1. `Question` → `EvaluationAgent` over evals in their school.
2. Teacher-level summaries.
3. `SchoolAgent` for that one school.
4. `SchoolAgent` + `Question` → final Chat response.

You can implement this as explicit code (no “magic recursion”) that runs these stages in order, with clear intermediate artifacts.

---

## 5. No retrieval for *which* evals — but still chunk inside each eval

You’re right on this point:

> “We don't really need an agent to do retrieval here because we can simply know that we want to be asking questions to every single evaluation that this user has access to.”

That means:

* You do **NOT** need an LLM to decide which evaluations to use.
* You *do* still likely need:

  * Some simple **text chunking** inside each evaluation if notes are long.
  * A small internal scheme like:

    * “For this eval, chunk notes into ≤N tokens each, LLM reads all chunks in one or multiple passes, but the scope is still just that one eval.”

The retrieval-type problem is reduced to:

> “Given all the notes for a single eval, how do I pack them into 1–2 LLM calls?”

---

## 6. Practical implementation notes for Matt

Some concrete dev guidance:

1. **Define schemas FIRST**

   * Hard-code the JSON schemas above for:

     * `EvaluationSummary`
     * `TeacherSummary`
     * `SchoolSummary`
     * `DistrictSummary`
   * Use Pydantic/TypeScript types to validate LLM outputs.

2. **Separate orchestration from prompts**

   * One module for:

     * building the org tree from DB
     * grouping objects
     * running batches of LLM calls
   * Another module for:

     * actual prompt templates per layer
     * JSON parsing and validation

3. **Parallelism**

   * Evaluation → Teacher step will be the heaviest (most calls).
   * Use async + concurrency limits to avoid hitting rate limits.
   * Cache evaluation-level summaries keyed by `(evaluation_id, question_signature)` if you want reuse.

4. **Cost control**

   * You can:

     * Limit timeframe (“last year” by default).
     * Sample evaluations if a teacher has 30+; e.g., “last 5 plus worst 2 & best 2” based on scores.

5. **Role-based behavior**

   * In the top-level orchestrator, pass:

     * `role: "principal"` vs `"superintendent"`
     * `scope: { schools: [...], teachers: [...] }`
   * Use different **answer prompts** for the final call, but keep internal summaries role-agnostic.

---

If you want, next I can:

* Turn this into concrete TypeScript/Python function signatures for the whole pipeline, or
* Draft the exact prompts and JSON schemas Matt can drop into the codebase as-is.
