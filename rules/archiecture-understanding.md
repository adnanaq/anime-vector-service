<!--
description: rules to parse solution architecture from docs/architecture.md
-->

# Architecture Understanding

### Source File

**File**: [architecture](docs/architecture.md)

## Required Parsin

1. **Load and parse complete Mermaid diagram**
2. **Extract and understand**:
   - Module boundaries and relationships
   - Data flow patterns
   - System interfaces
   - Component dependencies
3. **Validate any changes** against architectural constraints
4. **Ensure new code** maintains defined separation of concerns

## Error Handling Rules

1. **If file not found**: STOP and notify user
2. **If diagram parse fails**: REQUEST clarification
3. **If architectural violation detected**: WARN user
