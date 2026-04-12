---
name: White: Start Agent
description: Start the White LangChain agent to generate song proposals. Pass --with-html to also generate character sheets and fiction HTML.
category: White
tags: [white, agent, langchain]
---

Start the White concept agent to generate a new song proposal chain.

**Steps**

1. Check `$ARGUMENTS` for flags:
   - `--with-html` → include HTML generation (character sheets, timeline pages). Adds LLM + image cost.
   - No flags → HTML is skipped (default, cheaper).

2. Confirm the user wants to proceed — a new run costs LLM calls and produces a new thread in `chain_artifacts/`.

3. Run the agent:
   - Without HTML: `python -m run_white_agent start`
   - With HTML: `python -m run_white_agent start --with-html`

4. After the run completes, report:
   - The thread UUID
   - The song title and color from the final proposal
   - Next step: `wpipe status --production-dir shrink_wrapped/<thread>/production/<slug>/`

**Notes**
- The agent writes to `chain_artifacts/` (gitignored). Run `/white session-close` after the production pipeline is complete to shrinkwrap and commit.
- Negative constraints are loaded automatically from `shrink_wrapped/index.yml`.
