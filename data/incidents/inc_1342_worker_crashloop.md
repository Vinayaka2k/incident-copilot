
# Incident INC-1342: Worker CrashLoopBackOff

## Summary

Payment worker pods repeatedly restarted.

## Timeline

14:02 Deployment applied
14:04 Pods started crashing
14:06 Alert triggered
14:09 Logs revealed missing environment variable
14:12 Fix deployed

## Root Cause

Configuration change removed required environment variable.

## Resolution

Restore environment variable and redeploy service.

## Lessons Learned

Add configuration validation to deployment pipeline.
