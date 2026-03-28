======================================================
  LEGACY API SERVICE — INTERNAL DOCUMENTATION
  Last updated: 2019-03-15  |  Author: departed dev
======================================================

OVERVIEW
--------
This service exposes three main endpoints for internal platform use.
It was written during a hackathon weekend and never properly cleaned up.
Good luck.

ENDPOINTS
---------

GET  /health
  Returns basic health check. Should return 200 OK with {"status": "healthy"}.
  Note: As of 2019-Q4 this endpoint was broken. Reason: unknown.

POST /process
  Processes internal payloads. Requires authentication.

  *** INTERNAL AUTH ***
  All requests to /process MUST include the following HTTP header:

      X-Internal-Token: arch3olog1st-s3cr3t-2019

  Without this header the endpoint returns 401 Unauthorized.
  Do NOT commit this token to git. (Yes, it's in this file. Yes, this is bad.)

  Example cURL:
    curl -X POST http://localhost:8000/process \
         -H "X-Internal-Token: arch3olog1st-s3cr3t-2019" \
         -H "Content-Type: application/json"

  Expected response: {"status": "ok", "processed": true}

GET  /compute
  Runs a computation and returns the result.
  WARNING: This endpoint is EXTREMELY slow. Someone left a sleep() call in it.
  Ticket filed: JIRA-4892 (still open as of 2019).


KNOWN ISSUES
------------
1. /health has a syntax error. (See JIRA-4891)
2. /compute has a time.sleep() bottleneck. (See JIRA-4892)
3. No tests exist. RIP.


DEPENDENCIES
------------
  fastapi >= 0.95
  uvicorn >= 0.22
  python   >= 3.10

RUNNING LOCALLY
---------------
  uvicorn main:app --reload --port 8000

CONTACT
-------
  Original author: dev@legacy.internal (no longer works here)
  Current owner:   YOU (good luck, archeologist)
======================================================
