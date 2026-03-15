# Incident INC-1023: Payment Timeout

## Summary

Checkout failures occurred due to payment service timeouts.

## Timeline

10:01 Deploy version 4.3
10:03 Payment latency spike
10:05 Alerts triggered
10:08 Engineers investigated Redis latency
10:12 Rollback initiated
10:15 System recovered

## Root Cause

Redis node experienced high CPU usage due to increased traffic.

## Resolution

Rollback deployment and scale Redis cluster.

## Lessons Learned

Add autoscaling to Redis cluster.
