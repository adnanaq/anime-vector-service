<!--
description: Always attach when Implementing Code (Act/Code MODE)
-->

<!--
description: Include these rules while IMPLEMENTATION/Coding.
-->

# Code Implementation Protocol (ACT/Code MODE)

---

## Pre-Implementation Checklist

Before every code implementation/change ALWAYS do **2 things**:
a. **Read and understand** the documentation in [docs](/docs/) and [tasks](/tasks/)
a. **Get required code context** from [src](/src/) and other codes locations

---

## Programming Principles

- **Algorithm Efficiency**: Use the most efficient algorithms and data structures.
- **Modularity**: Write modular code; break complex logic into smaller atomic parts. Whenever possible break into classes, files, directories, functions.
- **File Management**: Break long files into smaller, more manageable ones with smaller functions.
- **Import Statements**: Prefer importing functions over modifying original files directly.
- **File Organization**: Organize files into directories and folders.
- **Reuse**: Prefer reusing existing code instead of writing from scratch.
- **Code Preservation**: Preserve what works—don't modify working components without necessity.
- **Systematic Sequence**: Complete one step before starting another; Keep systematic sequence of functionalities.
- **Design Patterns**: Apply appropriate design patterns. Use design patterns for maintainability, scalability, flexibility.
- **Proactive Testing**: All functionality must have corresponding test code [Testing](#-testing-always-test-after-implementation).

---

## Systematic Code Protocol

### Step 1: Analyze Code

#### Dependency Analysis

- Which components will be affected?
- What dependencies exist?
- Is this local or does it affect core logic?
- Which functionalities will be affected and how?
- What cascading effects will this change have?

#### Flow Analysis

- Before proposing any changes, conduct a complete end-to-end flow analysis of the relevant use case from entry point (e.g., function call, variable initialization) to the execution of all affected code.
- Track the flow of data and logic across components to understand full scope.

> **Document dependencies thoroughly**, including specific usage of functions/logic in files referenced by [memory](/rules/memory.md)

---

### Step 2: Plan Code

- If needed, initiate [Clarification](/rules/plan.md#clarification)
- Use step-by-step reasoning to outline a detailed plan, including component dependencies and architectural considerations
- Use [step-by-step reasoning](/rules/plan.md#step-by-step-reasoning) to outline a detailed plan including component dependencies, architectural considerations before coding
  Use [reasoning presentation](/rules/plan.md#reasoning-presentation) to explain all code changes, what each part does, and how it affects other areas.

#### Structured Proposals

Provide a proposal that specifies:

1. Ffiles, functions, or lines being changed
2. Justification (bug fix, improvement, new feature)
3. All affected modules or files
4. Potential side effects
5. Detailed explanation of tradeoffs and design reasoning

---

### Step 3: Make Changes

#### 1. Document Current State

1. Document Current State in files specified by [memory](/rules/memory.md)

- What’s currently working?
- What’s broken, the current error/issue?
- Which files will be affected?

#### 2. Plan Single Logical Change

##### Incremental Rollouts

- One logical feature at a time
- But fully resolve this one change by accomodating appropriate changes in other parts of the code.
- Adjust all existing dependencies and issues created by this change.
- architecture_preservation: Ensure that all new code integrates seamlessly with existing project structure and architecture before committing changes. Do not make changes that disrupt existing code organization or files.

#### 3. Run Simulation Testing

##### Simulation Analysis

- Simulate user interactions and behaviors by performing dry runs, trace calls, or other appropriate methods to rigorously analyze the impact of proposed changes on both expected and edge-case scenarios.
- Generate feedback on all potential side effects.

##### Simulation Validation

- Do not propose a change unless the simulation passes and verifies that all existing functionality is preserved, and if a simulation breaks, provide fixes immediately before proceeding.

> If Simulation Testing Passes, do the actual implementation.

---

### Step 4: Perform Testing

(See Testing section below)

---

### Step 5: Loop 1–4 and Implement All Changes

- Incorporate all the changes systematically, one by one.
- Verify the changes and test them one by one before proceeding.

[Step: 6] Optimize the implemented codes

- Optimize the implemented code, after all changes are tested and verified.

---

### Step 6: Optimize Final Code

- After testing, refactor or optimize for performance, clarity, and maintainability

## Reference

- Reference relevant documentation and best practices
- Use **Web Use** if external documentation is needed

---

# Testing (Always Test After Implementation)

### Dependency-Based Testing

- Create unit tests for any new functionality
- Rerun existing tests to confirm no breakage from [Analysis code](#step-1-analyze-code) and that existing behavior is still as expected.

### No Breakage Assertion

After you propose a change:

- Run the tests yourself and verify pass status
- Never assume tests will pass without verification

> Do not rely on me to do this, and be certain that my code will not be broken.

### Testing Structure

1. Keep test logic in **separate files** to keep codebase clean
2. Write test code for ANY added critical functionality ALWAYS. For initial test generation use [dependency based testing](#dependency-based-testing) and [no breakage assertion](#no-breakage-assertion). Then use [test plan](#-test-plan) to write code for extensive testing.
3. Dcoument test coverage and outcomes as per [memory](/rules/memory.md)

### 🧪 Test Plan

- Think of sufficiently exhaustive test plans for the functionalities added/updated against the requirements and desired outcomes.
- Define comprehensive test scenarios covering edge cases
- Specify appropriate validation methods for the project's stack
- Suggest monitoring approaches to verify the solution's effectiveness
- Consider potential regressions and how to prevent them

---

## Final Implementation Rule

- When implementing something new:
  **Be relentless**. Implement everything to the letter. Stop only when you're done till successfully testing, not before.

---

## Post-Implementation Checklist

After every code implementation/change ALWAYS do **2 things**:
a. Update any **affected codes** in [src](/src/) and other codes at other locations
b. Update related **documentation** in [docs](/docs/) and [tasks](/tasks/).
