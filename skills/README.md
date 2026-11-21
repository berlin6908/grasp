# Grasp agent skills

Project-specific workflows for an assistant working in this repository. Each
directory contains a portable `SKILL.md` with a name, a task-specific trigger,
and instructions grounded in the files shipped here.

| Skill | Use it for |
|---|---|
| [grasp-results-report](grasp-results-report/SKILL.md) | Find and compare existing results, including supplementary conditions and the specification audit |
| [grasp-evaluation](grasp-evaluation/SKILL.md) | Evaluate saved Grasp answers and evidence, or explain the paper metric contract |
| [grasp-experiments](grasp-experiments/SKILL.md) | Run the planning, grounding, answering and evaluation stages with a recorded configuration |
| [grasp-paper-figures](grasp-paper-figures/SKILL.md) | Reproduce or revise tables and figures from the bundled result notebook |
| [grasp-development](grasp-development/SKILL.md) | Change the implementation using its existing modules, dependencies and behavioral checks |
| [grasp-artifact-release](grasp-artifact-release/SKILL.md) | Add completed experiment results, update provenance and publish a requested release |

For example, ask an agent:

> Read `skills/grasp-results-report/SKILL.md` and compare the WebQSP GCR*
> supplementary condition with its recorded reference, keeping model roles
> and denominators explicit.

Run commands from the repository root. Paths in skill instructions identify
files in this checkout; relative Markdown links resolve from the skill file.
If a skill is installed elsewhere, locate the intended Grasp checkout before
using its commands. Select the skill for the requested task rather than
loading the whole collection.

The workflows consolidate the project's result-reporting, evaluation,
experiment and figure instructions around the public repository layout.
Experiment populations, numerical results and selected conditions remain in
`artifact/`; the skills do not maintain a second result registry. Runtime
dependencies remain in `pyproject.toml` and `requirements-notebook.txt`.
