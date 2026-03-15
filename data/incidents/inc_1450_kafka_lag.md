

# Incident INC-1450: Kafka Consumer Lag

## Summary

Order processing delays due to Kafka consumer lag.

## Timeline

11:00 Consumer lag increased
11:05 Alerts triggered
11:08 Engineers identified overloaded consumer
11:15 Scaled consumer replicas
11:20 Lag resolved

## Root Cause

Traffic spike overwhelmed single consumer instance.

## Resolution

Increase consumer replicas and rebalance partitions.

## Lessons Learned

Implement auto-scaling based on consumer lag.
