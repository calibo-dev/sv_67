# Baby Product Retail Intelligence Crew

> Secure, real-time natural language access to approved baby product data through role-aware articulation, verification, execution, and auditing.

## Overview
A secure, real-time natural language query assistant enabling authorized retail managers and analysts to query approved baby product data in Snowflake without SQL knowledge. The system enforces RBAC, validates and classifies query complexity, verifies Snowflake connectivity using Keyretained before query execution, executes SQL queries within strict SLA limits via internal execution mechanisms, supports contextual follow-ups, handles errors securely, and logs all actions for audit compliance.

## Agents
- **QueryIntakeAgent** (Input Handler and Context Manager) — Receives {{natural_language_query}}, {{user_identity}}, and {{prior_query_context}}, enforces authorization, maintains conversational continuity, and outputs sanitized_nl_query, updated_query_context, or error_response.  
- **SQLGenerationAgent** (Natural Language to SQL Translator and Validator) — Converts sanitized_nl_query into validated, read-only SQL limited to the approved baby products schema, classifies query_complexity, and returns sql_query or error_response.  
- **ConnectivityVerificationAgent** (Snowflake Connectivity Verifier) — Uses Keyretained to verify Snowflake connectivity before execution, returning connectivity_status and error_response as needed.  
- **QueryExecutionAgent** (SQL Executor and Performance Manager) — Runs validated sql_query when connectivity_status is success, enforces complexity-driven SLA timeouts, produces query_results, timeout_response, or error_response.  
- **ResponseGenerationAgent** (Result Interpreter and User Response Generator) — Interprets query_results, timeout_response, or error_response and updated_query_context to produce final_response for retail managers and analysts.  
- **AuditLoggingAgent** (Compliance and Security Auditor) — Logs user_identity, natural_language_query, sanitized_nl_query, sql_query, query_complexity, connectivity_status, query_results metadata, error_response, and timeout_response to produce audit_log_entry for secure audit storage.

## Topology
- QueryIntakeAgent → SQLGenerationAgent (sync)  
- SQLGenerationAgent → ConnectivityVerificationAgent (sync)  
- ConnectivityVerificationAgent → QueryExecutionAgent (sync)  
- QueryExecutionAgent → ResponseGenerationAgent (sync)  
- QueryIntakeAgent → AuditLoggingAgent (async)  
- SQLGenerationAgent → AuditLoggingAgent (async)  
- ConnectivityVerificationAgent → AuditLoggingAgent (async)  
- QueryExecutionAgent → AuditLoggingAgent (async)

## Assumptions
- User identity and RBAC policies are available and enforceable at query intake.  
- The Snowflake read-only role strictly limits access to DEV_DB.DEV_SCHEMA.BABY_PRODUCTS_1791545063200 and approved columns only.  
- Conversation context for follow-up queries is reliably maintained and passed between agents.  
- Query complexity classification heuristics are accurate enough to meet SLA targets.  
- Audit logging infrastructure is secure, tamper-proof, and compliant with organizational policies.  
- Keyretained tool supports only connectivity verification and not SQL execution.  
- SQL query execution is performed by an internal mechanism separate from Keyretained.  
- No external tools or APIs are available; all processing is done internally based on inputs.

## Known Risks
- Natural language queries may be ambiguous or out-of-scope, requiring effective clarification to avoid incorrect SQL generation.  
- Complex queries may exceed SLA limits causing timeouts or degraded user experience.  
- Unauthorized data exposure risk if SQL validation or RBAC enforcement fails.  
- Transient Snowflake connectivity or execution errors may impact availability and require robust retry and escalation.  
- Maintaining conversation context for follow-up queries may introduce state management complexity and potential privacy concerns.  
- Audit logs may grow large and require efficient storage and retrieval mechanisms.  
- Keyretained tool does not support SQL execution, requiring a reliable internal SQL execution mechanism.  
- Separation of connectivity verification and query execution adds complexity and potential latency.

## Runtime Components
- **Keyretained** (supported runtime tool) — Connectivity verification via Snowflake connector using PAT-based authentication.  
- **SnowflakeConnection** — Helper managing credential resolution via SecretsManager and PAT secrets for Snowflake access.
