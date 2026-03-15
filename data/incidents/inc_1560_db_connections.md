
# Incident INC-1560: Database Connection Exhaustion

## Summary

Checkout API returned errors due to exhausted DB connections.

## Timeline

16:10 Increased API errors
16:12 Alerts fired
16:15 Engineers identified connection pool exhaustion
16:18 Restarted application pods
16:20 System recovered

## Root Cause

Connection pool size too small for traffic spike.

## Resolution

Increase connection pool size.

## Lessons Learned

Tune connection pools based on peak traffic.