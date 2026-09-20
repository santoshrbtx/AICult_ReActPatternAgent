"""Local mock data used by the tools.

Everything here is in-memory / on-disk plaintext so the POC is 100% local.
Nothing here calls AWS.
"""

CUSTOMERS = {
    "CUST1001": {
        "customer_id": "CUST1001",
        "name": "Acme Corporation",
        "tenant_id": "TENANT-ACME",
        "plan": "Enterprise",
        "primary_contact": "ops@acme.example.com",
    },
    "CUST1002": {
        "customer_id": "CUST1002",
        "name": "Globex Industries",
        "tenant_id": "TENANT-GLOBEX",
        "plan": "Business",
        "primary_contact": "it@globex.example.com",
    },
    "CUST1003": {
        "customer_id": "CUST1003",
        "name": "Initech LLC",
        "tenant_id": "TENANT-INITECH",
        "plan": "Starter",
        "primary_contact": "admin@initech.example.com",
    },
}

# Convenience: allow lookup by company name too.
CUSTOMER_NAME_INDEX = {c["name"].lower(): c["customer_id"] for c in CUSTOMERS.values()}
CUSTOMER_NAME_INDEX["acme"] = "CUST1001"
CUSTOMER_NAME_INDEX["globex"] = "CUST1002"
CUSTOMER_NAME_INDEX["initech"] = "CUST1003"

TENANT_METRICS = {
    "TENANT-ACME": {
        "tenant_id": "TENANT-ACME",
        "active_db_connections": 198,
        "max_db_connections": 200,
        "cpu_utilization_pct": 92.4,
        "memory_utilization_pct": 87.1,
        "error_rate_pct": 6.3,
        "avg_query_latency_ms": 1450,
        "status": "DEGRADED",
    },
    "TENANT-GLOBEX": {
        "tenant_id": "TENANT-GLOBEX",
        "active_db_connections": 42,
        "max_db_connections": 200,
        "cpu_utilization_pct": 31.5,
        "memory_utilization_pct": 44.0,
        "error_rate_pct": 0.4,
        "avg_query_latency_ms": 85,
        "status": "HEALTHY",
    },
    "TENANT-INITECH": {
        "tenant_id": "TENANT-INITECH",
        "active_db_connections": 12,
        "max_db_connections": 100,
        "cpu_utilization_pct": 18.0,
        "memory_utilization_pct": 22.0,
        "error_rate_pct": 0.1,
        "avg_query_latency_ms": 60,
        "status": "HEALTHY",
    },
}

# A tiny mock log store keyed by tenant_id. Each entry is one line.
TENANT_LOGS = {
    "TENANT-ACME": [
        "2026-09-20T14:02:11Z INFO  request_id=req-8891 tenant=TENANT-ACME action=login user=alice ok",
        "2026-09-20T14:02:44Z WARN  tenant=TENANT-ACME db.pool active=190/200 nearing capacity",
        "2026-09-20T14:03:02Z ERROR tenant=TENANT-ACME db.pool connection acquisition timeout after 5000ms",
        "2026-09-20T14:03:05Z ERROR tenant=TENANT-ACME db.pool connection pool exhaustion, unable to serve request",
        "2026-09-20T14:03:19Z WARN  tenant=TENANT-ACME slow query detected duration=8123ms sql='SELECT * FROM invoices ...'",
        "2026-09-20T14:03:47Z ERROR tenant=TENANT-ACME long-running query blocked pool for 12s",
        "2026-09-20T14:04:12Z ERROR tenant=TENANT-ACME db.pool connection acquisition timeout after 5000ms",
        "2026-09-20T14:04:33Z INFO  tenant=TENANT-ACME retry scheduled for request_id=req-8901",
    ],
    "TENANT-GLOBEX": [
        "2026-09-20T14:00:01Z INFO  tenant=TENANT-GLOBEX action=heartbeat ok",
        "2026-09-20T14:01:12Z INFO  tenant=TENANT-GLOBEX action=report_generated report_id=r-42",
    ],
    "TENANT-INITECH": [
        "2026-09-20T14:00:03Z INFO  tenant=TENANT-INITECH action=heartbeat ok",
        "2026-09-20T14:02:55Z INFO  tenant=TENANT-INITECH action=user_login user=peter ok",
    ],
}
