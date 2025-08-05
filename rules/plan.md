<!--
description: Always include these rules.
-->

# PLANNING WORKFLOW

---

## Pre-Planning Protocol

Before every Plan/Architect task ALWAYS do 3 things:

1. Read the existing documentation in [docs](/docs/):

- [architecture](/docs/architecture.md)
- [product_requirement_docs](/docs/product_requirement_docs.md)
- [technical](/docs/technical.md).

2. Read the plans and related task planning & context in [tasks](/tasks/):

- [active_context](/tasks/active_context.md)
- [tasks_plan](/tasks/tasks_plan.md).

3. Get required solution conytext from the code files in [src](/src/) and other codes at other places.

---

## Step 1: Understand the Requirements

### Clarification

- Always ask for clarifications and follow-ups.
- Identify underspecified requirements and ask for detailed information.
- Fully understand all the aspects of the problem and gather details to make it very precise and clear.
- Ask towards all the hypothesis and assumptions needed to be made. Remove all the ambiguities and uncertainties.
- Suggest solutions that I didn't think about, i.e. anticipate my needs and things to be specified.
- Only after having **100% clarity and confidence**, proceed for SOLUTION.

---

## Step 2: Formulating the Solution

### Step-by-Step Reasoning

#### Decompose

- Have a meta architecture plan for the solution.
- Break down the problem into key concepts and smaller sub-problems.

#### Evaluate Options

- Think about all possible ways to solve the problem.
- Set up the evaluation criterias and trade-offs to access the merit of the solutions.
- Find the optimal solution and the criterias making it optimal and the trade-offs involved.

#### Web Use (optional)

Can use the web if needed using use_mcp_tool commands, particularly use the search tool from Perplexity. Example:

```json
<use_mcp_tool>
<server_name>perplexity-mcp</server_name>
<tool_name>search</tool_name>
<arguments>
{
  "param1": "value1",
  "param2": "value2"
}
</arguments>
</use_mcp_tool>
```

#### MULTI ATTEMPTS

- Reason out rigorously about the **optimality** of the solution.
- Question every assumption and inference, and support them with comprehensive reasoning.
- Think of **better solutions** than the present one Combining the strongest aspects of different solutions.
- Repeat the process [multi attempts](#multi-attempts) refining and integrating different solutions into one until a strong solution is found.
- Can use [web use](#web-use-optional) if needed to do research.

---

## Step 3: Solution Validation

### Reasoning Presentation

- Provide the **PLAN** with as much detail as possible.
- Break down the solution step-by-step and think every step in through detail with clarity.
- Reason out its optimality w.r.t. other promising solutions.
- Explicitly tell all your **assumptions**, **choices** and **decisions**
- Explain **trade-offs** in solutions
- restate my query in your own words if necessary after giving the solution

**Before implementing**, validate the SOLUTION plan produced by [reasoning presentation](#reasoning-presentation).

---

## Features of the Plan:

The plan should be:

- **extendable**: Further codes can be easily build on the current planning. And extending it in future will be well supported. Anticipate future functionalities and make the plan adaptable to those.
- **detailed**: The plan be very detailed, taking care of every aspect that will be affected with it and in every possible ways.
- **robust**: Plan for error scenarious and failure cases and have fallbacks for possible failure cases.
- **accurate**: Every aspect should be in sync with each other and individual components should be correct and the interfaces should be correct.

---

## Post-Planning Protocol

After every Plan/Architect task ALWAYS do 2 things:

1. Document the plan into existing documentation and update files in [docs](/docs/):

- [architecture](/docs/architecture.md)
- [product_requirement_docs](/docs/product_requirement_docs.md)
- [technical](/docs/technical.md)

2. Document the plans and related task planning & context in [tasks](/tasks/):

- [active_context](/tasks/active_context.md)
- [tasks_plan](/tasks/tasks_plan.md).
