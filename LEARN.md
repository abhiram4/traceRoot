# 📘 The Complete Guide to AI Agents, Silent Failures & TraceRoot

Welcome! If you understand AI (like LLMs, tokens, and prompts) but are new to **AI Agents**, this guide is written specifically for you.

We start from zero assumptions about agents, explain every single term and tool with crystal-clear analogies, walk through how agents silently break in the real world, and show how **TraceRoot** detects and fixes those failures.

---

## 🧭 Roadmap & Summary Guide

Here is the flow of this document:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Core Foundations        What is an AI Agent? (LLM vs Agent, Tools, ReAct)│
│ 2. The Big Problem         Why AI Agents Fail & The Illusion of Pass/Fail  │
│ 3. Essential Glossary      Every Agent Term Demystified in Plain English    │
│ 4. How TraceRoot Works     Under the Hood: 6-Step Pipeline                  │
│ 5. The Error Taxonomy      The 12 Ways Agents Break (With Real Examples)    │
│ 6. Tour of the Codebase    Where Every File & Feature Lives                 │
│ 7. Hands-on Quickstart     Writing, Tracing & Diagnosing Your First Agent   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 🧠 Core Foundations: What is an AI Agent?

### A. LLM vs. AI Agent

| Concept | What It Is | Analogy |
| :--- | :--- | :--- |
| **LLM (Language Model)** | A text prediction engine. Given an input prompt, it generates the most statistically likely next words in a single pass. | A brilliant scholar locked in a room with no phone, internet, or calculator. |
| **AI Agent** | An LLM wrapped in an **autonomous loop** equipped with **Tools** and **Memory**. It can make plans, run code, browse the web, check results, and decide what to do next. | The scholar is given a computer with internet, a terminal, calculator, and a notepad, and is told: *"Solve this task on your own."* |

---

### B. What is a "Tool" in Agent Terminology?

A **Tool** is simply a Python function or API endpoint that an LLM is allowed to trigger.

* **Example Tools:**
  * `web_search(query: str)`: Searches Google.
  * `execute_sql(query: str)`: Queries a PostgreSQL database.
  * `calculator(expression: str)`: Runs math calculations.
  * `send_email(recipient: str, body: str)`: Sends an email.

#### How Tool Calling Works Behind the Scenes:
1. You provide the LLM with a list of available tools (their names, descriptions, and expected parameters).
2. Instead of answering the user directly, the LLM outputs a structured JSON block:
   ```json
   {
     "tool_name": "fetch_stock_price",
     "arguments": {"symbol": "AAPL"}
   }
   ```
3. Your application executes `fetch_stock_price(symbol="AAPL")` in Python, gets the return value `"$182.50"`, and feeds that return value back into the LLM as an **Observation**.
4. The LLM reads the observation and decides whether to call another tool or write the final answer.

---

### C. The ReAct Loop (Reasoning + Acting)

Almost all modern AI agents follow the **ReAct (Reason + Act)** loop:

```
                      ┌────────────────────────┐
                      │  User Gives a Task     │
                      └───────────┬────────────┘
                                  ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                     THE RE-ACT AGENT LOOP                       │
 │                                                                 │
 │   1. THOUGHT: Agent reasons internally about what to do next.   │
 │       ("I need to check the user's flight status first.")       │
 │                                                                 │
 │   2. ACTION: Agent chooses a tool and formulates arguments.     │
 │       (Calls `get_flight_status(flight_id="AA102")`)            │
 │                                                                 │
 │   3. OBSERVATION: Environment runs tool & returns real data.    │
 │       (Result: `{"status": "Delayed by 2 hours"}`)              │
 │                                                                 │
 │   4. EVALUATION: Did I finish the task?                         │
 │       • NO  --> Loop back to Step 1.                            │
 │       • YES --> Synthesize final response to user.              │
 └─────────────────────────────────────────────────────────────────┘
```

---

### D. What is a "Trajectory" (or "Trace")?

A **Trajectory** is the complete, timestamped step-by-step recording of an agent trying to accomplish a task. It includes:
* The original user prompt.
* Every internal thought/reasoning step.
* Every tool called and the arguments passed.
* Every observation/error returned by the tools.
* The final synthesized answer.

---

## 2. 🚨 The Big Problem: Why Agents Silently Break

### The Illusion of Binary Pass/Fail

Traditional benchmarks (like GAIA, τ-bench, or SWE-bench) evaluate agents with a simple binary score:
* **Score = 1 (Pass):** The agent output matched the expected answer.
* **Score = 0 (Fail):** The agent output was wrong.

### The "Silent Error Cascade"

When an agent fails a 20-step task, **where did it actually break?**

Consider this real-world scenario:

```
Step 1: User says: "Book a flight from NYC to London leaving June 10 and returning June 1."
Step 2 [Reasoning]: "I will search for flights." (Valid)
Step 3 [Tool Call]: Calls search_flights(depart="2024-06-10", return="2024-06-01")
                    ⚠️ ROOT CAUSE: Return date is BEFORE departure date!
Step 4 [Tool Return]: Tool returns "Error 400: Invalid Date Range".
Step 5 [Reasoning]: "The search tool returned empty results, so there are no flights."
                    (Agent misinterprets the 400 error as a valid empty search result)
Step 6 [Final Answer]: "Sorry, all flights between NYC and London are fully booked."
```

#### Why Binary Evaluation Lies to You:
* A standard test only checks Step 6 and marks: **`FAIL (0)`**.
* A human engineer looking at the crash might say: *"The flight tool threw a 400 error at Step 4, our tool server must be buggy!"*
* **The Truth:** The tool server was fine! The true root cause was **Step 3** where the agent passed an impossible date order.

> **TraceRoot's Mission:** TraceRoot analyzes the full trajectory and proves mathematically:
> **Root Cause = Step 3 (`invalid_api_usage`) $\rightarrow$ Steps 4, 5, 6 were just downstream domino effects.**

---

## 3. 📖 Essential Agent Glossary (Demystified)

| Term | Simple Definition |
| :--- | :--- |
| **Trajectory ($T$)** | The full chronological history of an agent's execution from prompt to final answer. |
| **Atomic Step ($s_i$)** | A single discrete action in the trajectory (e.g., one thought, one tool invocation, or one observation). |
| **Turn ($\tau$)** | One full back-and-forth round of user-agent conversation. |
| **Root Cause ($s_{root}$)** | The **earliest unforced error** without which all subsequent failures would not have occurred. |
| **Propagated Error** | A downstream step that failed *only because* it inherited bad information from an earlier root cause. |
| **Causal DAG ($G = (V, E)$)** | A Directed Acyclic Graph (DAG) visualizing how the root error branched out and poisoned subsequent steps. |
| **Blast Radius ($B$)** | The total percentage of steps contaminated by the root error. |
| **LLM Judge** | An independent, high-intelligence LLM (like Gemini 2.0 Flash) prompted with strict evaluation rules to analyze and critique the trajectory. |
| **ECE (Expected Calibration Error)** | A mathematical metric measuring how much an LLM overestimates its own confidence. (Lower is better). |
| **Temperature Scaling** | A statistical technique that adjusts LLM confidence scores so that an "80% confidence" diagnosis is actually right 80% of the time. |
| **Cohen's Kappa ($\kappa$)** | A statistical agreement score between 0 and 1 measuring whether two evaluators (e.g. Human vs. Gemini) agree beyond random chance. |

---

## 4. ⚙️ How TraceRoot Works Under the Hood

TraceRoot operates as a 6-stage automated pipeline:

```
Raw Agent Log / Live Code
         │
         ▼
[Stage 1: Normalizer] ────> Standardizes LangChain/OpenAI/Inspect logs into NormalizedTrajectory
         │
         ▼
[Stage 2: Segmenter] ─────> Categorizes each step into one of 7 lifecycle phases
         │
         ▼
[Stage 3: LLM Judge] ─────> Applies counterfactual slicing to isolate the EARLIEST root cause
         │
         ▼
[Stage 4: DAG Builder] ───> Constructs causal dependency graph G = (V, E)
         │
         ▼
[Stage 5: Calibrator] ────> Applies temperature scaling to calibrate confidence scores
         │
         ▼
[Stage 6: Reporter] ──────> Generates Rich terminal tables, JSON metrics, and Interactive HTML UI
```

---

## 5. 🗂️ The 12-Category Agent Error Taxonomy

TraceRoot categorizes all agent failures into 5 high-level families and 12 specific failure types:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. COGNITIVE (Mind & Logic Errors)                                          │
│    • misunderstood_task: Misread user prompt or violated constraints.       │
│    • invalid_plan: Created an impossible or missing-subgoal plan.           │
│    • reasoning_hallucination: Made claims with zero supporting evidence.    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. TOOL INTERACTION (Hands & Execution Errors)                              │
│    • wrong_tool_selected: Used calculator when a search engine was needed.  │
│    • malformed_arguments: Invalid JSON syntax or wrong data types.          │
│    • invalid_api_usage: Semantic parameter violation (e.g. bad date range). │
│    • missing_required_parameter: Forgot a mandatory parameter.              │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. ENVIRONMENTAL (World & Server Errors)                                    │
│    • tool_execution_error: Server 500, crash, or timeout outside control.   │
│    • environment_state_inconsistency: Database changed out from underneath. │
│    • unexpected_observation: Tool returned corrupted or garbled text.      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. RECOVERY (Reaction Errors)                                               │
│    • unhandled_tool_error: Tool threw an error, but agent ignored it.       │
│    • repetitive_failing_action: Retried the exact same failing tool in loop.│
│    • premature_termination: Gave up early when other tools were available.  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. SYNTHESIS (Final Answer Errors)                                          │
│    • unsupported_conclusion: Final answer contradicts tool evidence.        │
│    • evidence_omission: Found the right data but left it out of the answer. │
│    • constraint_violation: Asked for JSON, gave markdown bullet points.     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 📁 Tour of the Codebase (Where Everything Lives)

```text
d:/PROJECTS/traceRoot/
│
├── traceroot/
│   ├── core/
│   │   ├── types.py          <- All Enums: ExecutionPhase, ErrorCategory, PropagationRelation
│   │   └── config.py         <- YAML config loader & Judge settings
│   │
│   ├── trace/
│   │   ├── schema.py         <- Pydantic models: TrajectoryStep, NormalizedTrajectory
│   │   ├── normalize.py      <- Ingests raw logs (OpenAI, LangChain, Inspect AI)
│   │   └── segment.py        <- Classifies steps into the 7 execution phases
│   │
│   ├── taxonomy/
│   │   ├── categories.py     <- Definitions of the 12 error taxonomy classes
│   │   └── mappings.py       <- Validation rules for phase-to-error constraints
│   │
│   ├── diagnosis/
│   │   ├── judge.py          <- LLMJudge (Gemini/OpenAI/Claude) & MockJudge
│   │   ├── prompts.py        <- Causal CoT & Few-shot exemplar prompts
│   │   ├── localizer.py      <- Isolates first root-cause step
│   │   ├── propagation.py    <- Builds NetworkX Causal Failure DAG
│   │   └── ensemble.py       <- Multi-judge majority voting & agreement
│   │
│   ├── calibration/
│   │   ├── metrics.py        <- Computes Expected Calibration Error (ECE) & Brier Score
│   │   ├── calibration.py    <- Temperature Scaler & Platt Scaling
│   │   └── reliability.py    <- Generates ASCII & SVG Reliability Diagrams
│   │
│   ├── evaluation/
│   │   ├── benchmark.py      <- Evaluates Research Questions RQ1 through RQ5
│   │   ├── classification.py <- Macro F1 & Precision/Recall reports
│   │   ├── localization.py   <- Exact Step & Tolerance (k=1) Accuracy
│   │   └── agreement.py      <- Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha
│   │
│   ├── execution/
│   │   ├── trace_capture.py  <- @traceroot.trace decorator & TrajectoryTracer
│   │   └── inspect_runner.py <- Inspect AI runner adapter
│   │
│   ├── reporting/
│   │   ├── report.py         <- Generates structured attribution reports
│   │   ├── visualizer.py     <- Rich terminal tables & colored ASCII DAGs
│   │   └── templates/        <- report_html.jinja2 & report_markdown.jinja2
│   │
│   └── cli.py                <- CLI interface (diagnose, benchmark, ui, calibrate)
│
├── configs/                  <- YAML configurations for judges and ensembles
├── data/seed_benchmark/      <- 25+ gold-standard trajectories with human annotations
├── reports/                  <- Generated HTML/JSON benchmark evaluation reports
└── tests/                    <- 33 unit tests (100% passing)
```

---

## 7. 🚀 Hands-on Quickstart: Your First Agent Diagnosis

### Step 1: Write an Agent with the `@traceroot.trace` Decorator

```python
import traceroot

@traceroot.trace("Find order status for #1001")
def my_order_agent(tracer: traceroot.TrajectoryTracer):
    # Step 1: Internal Thought
    tracer.log_thought("Looking up order status via database API")
    
    # Step 2: Tool Call
    tracer.log_tool_call("fetch_order", {"order_id": 1001})
    
    # Step 3: Tool Observation
    tracer.log_tool_result("fetch_order", {"status": "Shipped", "tracking": "TRK992"})
    
    # Step 4: Final Synthesis
    tracer.log_final_response("Order #1001 has shipped. Tracking is TRK992.", success=True)
    return "Done"

# Run your agent
result, trajectory = my_order_agent()
```

---

### Step 2: Diagnose the Trajectory with Free Gemini Judge

```python
# 1-line root-cause diagnosis
diagnosis = traceroot.diagnose(trajectory, judge_provider="gemini")

print(f"Failure Detected: {diagnosis.failure_detected}")
print(f"Confidence: {diagnosis.confidence:.1%}")
print(f"Summary: {diagnosis.summary_rationale}")
```

---

### Step 3: Launch the Interactive Web Dashboard

In your terminal:
```powershell
python -m traceroot.cli ui
```
This will compile and launch the interactive, dark-mode visualizer with causal DAGs directly in your default browser.

---

### Step 4: Run the Scientific Benchmark Suite

```powershell
python -m traceroot.cli benchmark --judge gemini
```

This runs all 25 benchmark trajectories and outputs formal empirical metrics for:
* **RQ1**: Failure localization accuracy.
* **RQ2**: Taxonomy classification accuracy & Cohen's Kappa.
* **RQ3**: Diagnostic actionability.
* **RQ4**: Lifecycle phase distribution of errors.
* **RQ5**: Confidence calibration (ECE reduction).
