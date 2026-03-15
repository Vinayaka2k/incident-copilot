# Runbook: Kafka Consumer Lag

## Symptoms

* Consumer lag exceeding threshold
* Delayed event processing
* Backlog in event pipelines

## Possible Causes

* Consumer crash or restart loop
* Downstream database slowdown
* Large message bursts
* Broker overload

## Diagnosis Steps

1. Check Kafka consumer group lag metrics.
2. Inspect consumer logs.
3. Verify broker health.
4. Check downstream DB write latency.

## Resolution

* Restart consumer pods.
* Increase consumer replica count.
* Rebalance partitions.

## Escalation

Data Platform Team
