# Runbook: Database Connection Exhaustion

## Symptoms

* Errors: "too many connections"
* Slow queries
* Application timeouts

## Possible Causes

* Connection pool misconfiguration
* Query locks
* Long running transactions

## Diagnosis Steps

1. Check DB connection pool metrics.
2. Identify long running queries.
3. Inspect query lock status.

## Resolution

* Restart application instances.
* Tune connection pool settings.
* Kill long-running queries.

## Escalation

Database Operations Team
