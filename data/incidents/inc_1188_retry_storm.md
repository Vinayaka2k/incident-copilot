# Incident INC-1188: Retry Storm After Deployment

## Summary

A retry logic bug caused massive retry loops in the payment service.

## Timeline

09:58 Deploy version 4.5
10:02 Error rate spike
10:04 Checkout failures reported
10:06 Engineers detected retry storm
10:10 Rollback deployed
10:13 Service stabilized

## Root Cause

Incorrect retry logic introduced in deployment.

## Resolution

Rollback deployment and patch retry logic.

## Lessons Learned

Add integration test for retry scenarios.
