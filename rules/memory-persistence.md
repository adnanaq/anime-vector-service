# Memory Persistence System

## Overview
This system maintains context and knowledge across development sessions, implementing the principle that "Living memory persists. Learning compounds."

## Memory Architecture

### 1. Session Memory (WORKING_LOG/)
- **Purpose**: Capture development decisions, patterns, and insights per session
- **Persistence**: Git-tracked, session-based files
- **Structure**: Daily logs with template-based consistency

### 2. Active Context (tasks/active_context.md)
- **Purpose**: Current development priorities and context
- **Updates**: Real-time during development sessions
- **Integration**: Links to WORKING_LOG insights

### 3. Compounded Learning (rules/lessons-learned.md)
- **Purpose**: Accumulated patterns and best practices
- **Source**: Distilled from WORKING_LOG sessions
- **Application**: Referenced in @LESSONS token

### 4. Error Memory (rules/error-documentation.md)
- **Purpose**: Known issues and their resolutions
- **Source**: Captured from debugging sessions
- **Application**: Referenced in @ERRORS token

## Memory Update Protocol (@MEM_UPDATE)

### Automatic Updates
Trigger memory updates after significant development activities:

```
@MEM_UPDATE → Review session context → Update active_context.md → 
Extract patterns → Update lessons-learned.md → Document errors → 
Update error-documentation.md → Create session log
```

### Update Triggers
1. **After major implementation**: New features, architectural changes
2. **After debugging**: Problem resolution, root cause analysis
3. **End of session**: Session summary, knowledge extraction
4. **Pattern discovery**: New insights, anti-patterns identified

## Living Memory Components

### Session Continuity
- Each session begins with context review from WORKING_LOG/
- Previous session insights inform current development approach
- Patterns compound across multiple sessions

### Knowledge Extraction
- Successful patterns → rules/lessons-learned.md
- Resolved errors → rules/error-documentation.md  
- Architectural insights → docs/architecture.md updates
- Task learnings → tasks/active_context.md

### Context Preservation
- Development decisions documented with rationale
- Alternative approaches considered and rejected
- Impact assessment for future reference
- Integration points with existing system

## Memory Integration with Token System

### Enhanced Tokens
- `@LESSONS` → Now includes WORKING_LOG/ patterns
- `@ERRORS` → Enhanced with session-based error resolution
- `@MEM_UPDATE` → Structured memory persistence workflow

### Context Queries
Before major implementations, query memory system:
- "Have we solved similar problems before?"
- "What patterns worked well in recent sessions?"
- "What architectural constraints emerged recently?"
- "What testing approaches proved effective?"

## Implementation Guidelines

### Memory Capture
1. **Real-time**: Update active_context.md during development
2. **Session-end**: Complete WORKING_LOG/ entry with insights
3. **Weekly**: Extract patterns into lessons-learned.md
4. **On-demand**: Document errors when resolved

### Memory Retrieval
1. **Session start**: Review recent WORKING_LOG/ entries
2. **Before implementation**: Check lessons-learned.md and error-documentation.md
3. **During debugging**: Search error-documentation.md for similar issues
4. **Planning phases**: Integrate active_context.md priorities

### Memory Evolution
- Memory files evolve based on project needs
- Outdated patterns get archived or updated
- New categories emerge from session patterns
- Cross-references maintained between memory components

## Core Principles

### Context Continuity
- Every session builds on previous knowledge
- No context lost between development cycles
- Decisions documented with full reasoning

### Adaptive Learning
- System learns from successes and failures
- Patterns refined based on outcomes
- Error prevention through historical awareness

### Structured Knowledge
- Information categorized for easy retrieval
- Cross-references between related concepts
- Searchable format for rapid access

---
*"Living memory persists. Learning compounds."* - Applied to our development workflow