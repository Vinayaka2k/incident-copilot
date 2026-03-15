# Runbook: Payment Service Timeout

## Symptoms

* HTTP 504 errors on `/charge`
* Payment API latency above 2 seconds
* Increase in checkout failures
* Alerts triggered: `payment_latency_p95`

## Possible Causes

* Redis latency
* Database connection pool exhaustion
* Recent deployment introducing retry storms
* Downstream dependency slowdown

## Diagnosis Steps

1. Check if a deployment occurred in the last 30 minutes.
2. Inspect Redis latency dashboard.
3. Verify database connection pool usage.
4. Check worker queue backlog.
5. Inspect payment-service logs.

## Resolution

* Roll back recent deployment if latency spike started after release.
* Restart payment worker pods.
* Increase connection pool size if exhaustion is observed.

## Escalation

Payments Platform Team



