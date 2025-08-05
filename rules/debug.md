<!--
description: Always include while DEBUGGING
-->

# Debugging

> Below debugging routine is for persistent errors or incomplete fixes. So use this routine only when got stuck.

---

## Diagnose

### What to do:

- Gather all error messages, logs, and behavioral symptoms
- Add relevant context from files
- Retrieve relevant project architecture, plan and current working task as specified in @memory.mdc

---

## General Debugging Workflow

- Whenever you fail with any test result:
  - always add more context using [Diagnore](#diagnose)
  - Debug the issue effectively, thoroughly and have complete information **before** moving attempting a fix.
- Clearly explain your:
  - **Observation** (what you're seeing)
  - **Reasonings** (why this is **exactly** the issue and not anything else)
- If you **aren't sure**:
  - first get more OBSERVATIONS by adding more [Diagnose](#diagnose) context to the issue so you exactly and specifically know what's wrong.
  - Additionally you can seek [Clarification](/rules/plan.md#clarification) if needed.

---

## Deep Debugging Tactics

- Understand architecture using [analyze code ](/rules/implement.md#step-1-analyze-code) relevant to the issue.
- Use [step-by-step reasoning](/rules/plan.md#step-by-step-reasoning) to think of all possible causes
- Architectural misalignment
- Design flaws
- Broader causes beyond bug

- Search for similar issues or solved patterns:
- In codebase
- In [error documentation](/rules/error-documentation.md)
- Via [web use](/rules/plan.md#web-use-optional) if needed

---

## Resolution & Fix

- Present your fix using [reasoning presentation](/rules/plan.md#reasoning-presentation) for review/validation.
- Start modifying code to update and fix things using
- [systematic code protocol](/rules/implement.md#systematic-code-protocol)
- [Testing](/rules/implement.md#testing-always-test-after-implementation)
