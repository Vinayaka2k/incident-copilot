# Runbook: Redis Latency

## Symptoms

* Increased API latency across services
* Redis operations exceeding 100ms
* Cache hit ratio dropping
* Timeout errors in dependent services

## Possible Causes

* Redis node CPU saturation
* Network congestion
* Large key eviction cycles
* Failover event

## Diagnosis Steps

1. Check Redis CPU and memory usage.
2. Inspect network packet loss.
3. Review Redis slowlog.
4. Verify cluster failover events.

## Resolution

* Restart Redis node if safe.
* Scale Redis cluster.
* Flush problematic large keys if necessary.

## Escalation

Infrastructure Platform Team
