# Comprehensive RAG Application Testing & Quality Audit

Act as a Senior AI/ML Engineer, RAG Systems Architect, QA Engineer, and LLM Evaluation Specialist.

I built this RAG (Retrieval-Augmented Generation) application using you. I now want you to conduct a **thorough, end-to-end quality audit and testing of the entire application**.

Your objective is to determine whether the application works correctly, retrieves relevant information, generates accurate answers grounded in the source documents, handles edge cases, and avoids hallucinations.

Do not limit yourself to checking whether the application runs. Test its actual functionality, reliability, retrieval quality, response accuracy, and failure-handling capabilities.

## PHASE 1: Understand the Entire Application

Before running tests:

1. Inspect the complete project structure.
2. Read the README, configuration files, source code, dependencies, and existing tests.
3. Understand the complete RAG pipeline:
   - Document ingestion
   - Document loading and parsing
   - Text preprocessing and cleaning
   - Chunking and splitting
   - Metadata extraction and management
   - Embedding generation, if applicable
   - Indexing and storage
   - Retrieval mechanism
   - Query processing
   - Prompt construction
   - LLM generation
   - Context injection
   - Source attribution and citations
   - Response formatting
4. Identify the retrieval architecture currently implemented. Determine whether it uses keyword-based retrieval, semantic search, hybrid search, or another approach. Do not assume anything without inspecting the code.
5. Identify the LLM, embedding model, vector database, indexing mechanism, and other dependencies.
6. Understand how the frontend and backend communicate.

Create a clear architecture overview of the actual implementation before proceeding.

## PHASE 2: Functional Testing

Test every major feature of the application.

Include:

- Application startup and shutdown.
- Document upload and ingestion.
- Support for different file formats.
- Empty files and unsupported file formats.
- Large documents.
- Duplicate document ingestion.
- Malformed or corrupted documents.
- Special characters, tables, code blocks, and unusual formatting.
- Chunk generation and chunk boundaries.
- Metadata preservation.
- Index creation and persistence.
- Retrieval from indexed documents.
- Query submission and response generation.
- Conversation history, if implemented.
- Source citations and document references.
- Error handling and API responses.
- Data persistence after application restart.

Verify that each feature behaves as expected. Record actual results rather than assuming a feature works because its code exists.

## PHASE 3: Retrieval Quality Testing

This is one of the most important parts of the audit.

Evaluate whether the retrieval system consistently finds the correct information from the knowledge base.

Test the following query types:

1. Exact keyword matches.
2. Partial keyword matches.
3. Semantic or paraphrased questions.
4. Questions using synonyms.
5. Questions with different sentence structures.
6. Questions involving multiple documents.
7. Questions requiring information from different sections of a document.
8. Questions involving numbers, dates, names, and technical terminology.
9. Questions with spelling mistakes.
10. Ambiguous queries.
11. Very short queries.
12. Long and detailed queries.
13. Queries containing irrelevant information.
14. Queries where the answer exists but uses different terminology.
15. Queries where the requested information does not exist in the knowledge base.

For each test, evaluate:

- Whether the correct document was retrieved.
- Whether the correct chunk was retrieved.
- Whether the retrieved content contains sufficient information to answer.
- Whether irrelevant chunks were retrieved.
- Whether important information was missed.
- Whether the retrieval ranking makes sense.

If possible, x
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Hit Rate@K
- Normalized Discounted Cumulative Gain (NDCG@K)

Clearly document the assumptions and ground truth used to calculate these metrics.

## PHASE 4: LLM Response Quality Testing

Evaluate whether the LLM generates accurate, relevant, and contextually grounded answers.

Test:

1. Questions with answers explicitly present in the documents.
2. Questions requiring multiple pieces of information.
3. Questions requiring summarization.
4. Questions requiring comparison.
5. Questions requiring numerical calculations based on retrieved content.
6. Questions requiring information from multiple documents.
7. Questions with incomplete context.
8. Questions with contradictory information across documents.
9. Questions where no answer exists.
10. Questions containing misleading assumptions.
11. Questions that require the model to acknowledge uncertainty.
12. Questions that might cause the model to invent information.

Check whether:

- The answer is factually supported by retrieved context.
- The answer correctly interprets the source material.
- The model introduces unsupported facts.
- The model changes numbers, dates, names, or technical details.
- The answer omits important information.
- The answer contradicts the source documents.
- The model appropriately refuses to answer when evidence is insufficient.

## PHASE 5: Hallucination & Grounding Tests

Perform dedicated hallucination testing.

Create questions for which:

- The answer is clearly present in the knowledge base.
- The answer is partially present.
- The answer is completely absent.
- The documents contain conflicting information.
- The question contains a false premise.
- The retrieved context is irrelevant.
- The context contains insufficient evidence to answer confidently.

Pay special attention to whether the model invents answers when the knowledge base does not contain the requested information.

Verify that the application follows this principle:

**If the answer cannot be established from the available source documents, the model should explicitly acknowledge that the information is unavailable rather than fabricate an answer.**

Also test prompt injection attempts contained within user queries and uploaded documents. The application must treat retrieved document content as untrusted data, not as instructions that override the system prompt or application rules.

## PHASE 6: Chunking, Indexing & Ingestion Evaluation

Inspect whether document chunking and indexing are implemented correctly.

Evaluate:

- Chunk size.
- Chunk overlap.
- Splitting strategy.
- Whether chunks preserve meaningful context.
- Whether sentences or important sections are unnecessarily separated.
- Whether headings and metadata are preserved.
- Whether chunks contain duplicate information.
- Whether document boundaries are respected.
- Whether re-ingestion creates duplicate records.
- Whether updates and deletions are reflected in the index.
- Whether indexing and retrieval remain consistent after restarting the application.

Determine whether the existing chunking strategy is appropriate for the type of documents being processed.

Identify potential improvements, but do not change the implementation during the initial audit.

## PHASE 7: API, Backend & Integration Testing

Test the backend independently wherever possible.

Check:

- API endpoint functionality.
- Request and response validation.
- Invalid payloads.
- Missing parameters.
- Incorrect data types.
- Authentication and authorization, if implemented.
- HTTP status codes.
- Timeout handling.
- Rate-limit handling.
- LLM provider failures.
- Embedding provider failures.
- Database connection failures.
- Vector store or indexing failures.
- Retry mechanisms.
- Logging and exception handling.

Verify that failures are handled gracefully and do not expose sensitive information.

## PHASE 8: Performance & Load Testing

Evaluate the application's performance under different workloads.

Test:

- Small documents.
- Medium-sized documents.
- Large documents.
- Increasing numbers of indexed documents.
- Short queries.
- Complex queries.
- Multiple consecutive queries.
- Concurrent requests, where supported.
- Repeated ingestion operations.

Measure and report:

- Document ingestion time.
- Indexing time.
- Retrieval latency.
- LLM response latency.
- Total end-to-end response time.
- Memory and CPU consumption where practical.
- Token usage, if available.
- API errors and failure rates.

Identify bottlenecks and performance degradation as the dataset grows.

Do not run destructive or excessive load tests against external services or production systems. Use reasonable limits and local resources.

## PHASE 9: Security & Reliability Testing

Evaluate:

- Prompt injection vulnerabilities.
- Insecure file uploads.
- Malicious or malformed document content.
- Path traversal risks.
- Sensitive data exposure.
- API key and secret management.
- Error message information leakage.
- Cross-user document access, if the application supports multiple users.
- Unauthorized access to indexed documents.
- Resource exhaustion through oversized files or requests.

Do not expose, print, or transmit actual API keys, passwords, or other secrets during testing.

## PHASE 10: Build a Comprehensive Test Dataset

Create a dedicated test dataset using the application's existing sample documents or create a controlled synthetic dataset if the application does not contain sufficient test data.

The dataset should include:

- Questions with directly available answers.
- Questions requiring contextual understanding.
- Questions requiring multiple chunks.
- Questions requiring multiple documents.
- Paraphrased questions.
- Numerical and factual questions.
- Unanswerable questions.
- Ambiguous questions.
- Contradictory information.
- Prompt injection attempts.

For every test case, document:

- Test case ID.
- Input query.
- Expected answer or expected behavior.
- Relevant source document.
- Expected supporting information.
- Actual retrieved results.
- Actual generated response.
- Pass/fail status.
- Failure explanation.

Do not manufacture ground truth from the model's own responses. Establish expected answers from the source documents.

## PHASE 11: Automated Testing

Where possible, implement automated tests using the project's existing testing framework.

Create a structured test suite covering:

- Unit tests.
- Integration tests.
- End-to-end tests.
- Retrieval evaluation tests.
- Generation evaluation tests.
- Regression tests.
- Error-handling tests.

Use mocked external services where appropriate to avoid unnecessary API costs.

Keep the tests repeatable and ensure they can be executed again after code changes.

Do not overwrite or break existing tests.

## PHASE 12: Final Audit Report

After executing the tests, generate a detailed report with the following structure:

### 1. Executive Summary
Overall assessment of the application's quality and reliability.

### 2. Architecture Overview
Explain how the current RAG pipeline actually works.

### 3. Test Coverage
List every testing category and the number of tests executed.

### 4. Test Results
Include passed, failed, skipped, and blocked tests.

### 5. Retrieval Evaluation
Report retrieval metrics wherever valid ground truth is available.

### 6. Answer Quality Evaluation
Assess factual accuracy, relevance, completeness, and grounding.

### 7. Hallucination Evaluation
Identify instances of unsupported answers and fabricated information.

### 8. Performance Evaluation
Report measured performance and bottlenecks.

### 9. Security Evaluation
Document identified vulnerabilities and risks.

### 10. Critical Bugs
List bugs that could cause incorrect results, data loss, application crashes, or security issues.

### 11. Improvement Recommendations
Categorize issues into:

- Critical: Must fix immediately.
- High: Significantly affects reliability or correctness.
- Medium: Affects quality or performance.
- Low: Minor optimizations.

For every issue, provide:
- Description.
- Evidence from testing.
- Likely root cause.
- Recommended solution.
- Relevant file or code location.

### 12. Overall RAG Quality Score
Provide separate scores from 0–100 for:

- Functional correctness.
- Ingestion quality.
- Retrieval quality.
- Answer accuracy.
- Context grounding.
- Hallucination resistance.
- Performance.
- Reliability.
- Security.

Clearly distinguish measured scores from subjective assessments. Explain the scoring methodology.

### 13. Final Verdict
Classify the application as:

- Not functional.
- Functional but unreliable.
- Functional with significant limitations.
- Reliable for controlled use.
- Production-ready, subject to the tested scope.

Do not label the application production-ready merely because all basic tests pass.

---

## IMPORTANT EXECUTION RULES

1. **Actually execute the tests.** Do not simply inspect the source code and tell me what should work.
2. Do not assume existing functionality is correct.
3. Do not fabricate test results, metrics, or performance measurements.
4. Distinguish between verified failures, potential risks, and untested areas.
5. If a test cannot be executed because of missing credentials, external dependencies, unavailable services, or environment limitations, document the blocker.
6. Do not modify the existing application code during the initial audit.
7. Preserve the existing project structure and data.
8. Do not delete, overwrite, or reset existing databases, indexes, or user documents.
9. Keep test data isolated from actual application data wherever possible.
10. Use the existing environment and testing tools where possible.
11. If you discover bugs, first document them with reproducible steps and evidence.
12. After completing the audit, provide a prioritized remediation plan. Do not automatically implement fixes until the audit report is complete.

**The ultimate objective is to establish whether my RAG application genuinely retrieves the right information and generates reliable, source-grounded answers—not merely whether the application can return a response.**