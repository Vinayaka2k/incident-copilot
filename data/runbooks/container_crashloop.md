# Runbook: Kubernetes CrashLoopBackOff

## Symptoms

* Pod repeatedly restarting
* `CrashLoopBackOff` status in Kubernetes
* Service unavailable

## Possible Causes

* Bad deployment config
* Missing environment variable
* Dependency service unavailable
* Container memory limit exceeded

## Diagnosis Steps

1. Inspect container logs.
2. Check environment variables.
3. Verify dependent services.
4. Inspect resource usage.

## Resolution

* Rollback deployment.
* Fix configuration errors.
* Increase resource limits if needed.

## Escalation

Service Owner Team