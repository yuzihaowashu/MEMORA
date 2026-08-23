# Evaluation

The evaluation package exposes the two MEMORA-Bench protocols.

| Module | Role |
|---|---|
| `eam_qa/tasks.py` | Loads and validates released questions and multi-run specifications |
| `eam_qa/prompts.py` | Defines EAM-QA task, abstention, and No-Memory instructions |
| `eam_qa/answers.py` | Parses multiple-choice answer letters |
| `eam_qa/runner.py` | Executes EAM-QA over a selected memory condition |
| `../memory_agent/memory_representations/` | Defines the MEMORA, Flat-1D, and Graph-2D representation interfaces |
| `planning/runner.py` | Orchestrates MEMORA-Planning runs and the public CLI |
| `planning/planning_environment.py` | Maintains planning state and executes tool calls |
| `planning/context.py` | Retrieves the initial memory context supplied before ReAct |
| `planning/prompts.py` | Maps public evaluation settings to their system prompts |
| `planning/prompt_profiles/` | Stores the MEMORA and baseline prompt contracts by setting |
| `planning/tasks.py` | Loads and validates Replay and Generalize task suites |
| `planning/parser.py` | Extracts ordered steps from a planner's final response |
| `planning/step_checks.py` | Provides deterministic predicates used by OrderExec |
| `settings.py` | Defines the seven paper conditions shared by both protocols |
| `results.py` | Writes evaluation outputs atomically for both protocols |

Saved-output aggregation for paper tables lives under the repository-level
`scripts/paper_results/` directory. The reported
RGP panel is entirely rule-based: evaluation does not call an LLM judge.

The public commands are `memora-eam-qa` and `memora-plan`, registered in
`pyproject.toml`. The `__init__.py` files mark package boundaries; the package
does not maintain duplicate `python -m` wrappers.

## EAM-QA path

1. `eam_qa/tasks.py` validates the released five-choice questions.
2. `../memory_agent/memory_representations/` defines what each memory
   representation lets the agent query; `eam_qa/prompts.py` adds task-specific
   answer guidance.
3. `eam_qa/answers.py` interprets the final multiple-choice response.
4. `eam_qa/runner.py` selects the evaluation condition, invokes the model and
   memory tools, and records per-question results.
5. `results.py` saves the evaluation output for later aggregation.

## Planning path

1. `planning/tasks.py` reads a released Replay or Generalize suite.
2. `planning/planning_environment.py` creates the condition-specific memory interface.
3. `planning/context.py` retrieves an initial memory context; the planner then
   gathers additional evidence through the registered ReAct tools.
4. `planning/parser.py` converts the final numbered response into ordered steps.
5. `results.py` saves model responses and tool traces without scoring them.
6. `scripts/paper_results/planning_metrics.py` computes OrderExec, KeyObj,
   PrefAdh, and RGP
   from the saved outputs using deterministic rules.
