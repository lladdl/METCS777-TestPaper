#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YARN-safe Parquet Top Drivers script
------------------------------------
Prevents executor kills on small clusters.
• Renames columns if header missing
• Repartitions and coalesces data
• Adds Spark configs to handle shuffle safely
• Computes top 10 drivers by $/minute
"""

import sys, time
from pyspark.sql import SparkSession

# ------------------ Setup ------------------
if len(sys.argv) != 3:
    print("Usage: taxi_top_drivers_parquet_safe.py <input_parquet_path> <output_dir>", file=sys.stderr)
    sys.exit(-1)

input_path = sys.argv[1]
output_dir = sys.argv[2]

start_time = time.time()
spark = (
    SparkSession.builder
    .appName("TaxiAnalysisRDD-TopDrivers-Parquet-SAFE")
    # === YARN-safe tuning for 2-worker cluster ===
    .config("spark.executor.instances", "2")
    .config("spark.executor.cores", "1")
    .config("spark.executor.memory", "2g")
    .config("spark.executor.memoryOverhead", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.default.parallelism", "4")
    .config("spark.task.maxFailures", "8")
    .config("spark.network.timeout", "800s")
    .config("spark.executor.heartbeatInterval", "60s")
    .config("spark.speculation", "false")
    .getOrCreate()
)
sc = spark.sparkContext

# ------------------ Load Parquet ------------------
df = spark.read.parquet(input_path)
print("🪣 Original columns:", df.columns)

# Fix header if missing
expected_cols = [
    "medallion", "hack_license", "pickup_datetime", "dropoff_datetime",
    "trip_time_in_secs", "trip_distance", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "payment_type", "fare_amount",
    "surcharge", "mta_tax", "tip_amount", "tolls_amount", "total_amount"
]
if df.columns[0] not in expected_cols:
    print("⚙️  Detected missing header — renaming columns...")
    df = df.toDF(*expected_cols)

print("✅ Columns after load:", df.columns[:5], "...")

# Repartition small (avoid too many tasks)
df = df.repartition(4).cache()
df.count()  # force materialization safely

# ------------------ Compute Top Drivers ------------------
rdd = df.select("hack_license", "trip_time_in_secs", "total_amount").rdd.coalesce(4)

def parse_row(row):
    try:
        hack_license = row[0]
        trip_time_secs = row[1]
        total_amount = row[2]
        if hack_license is None or trip_time_secs is None or total_amount is None:
            return None
        trip_time_mins = float(trip_time_secs) / 60.0
        if trip_time_mins > 0:
            return (hack_license, (trip_time_mins, float(total_amount)))
    except Exception:
        return None
    return None

parsed = rdd.map(parse_row).filter(lambda x: x is not None)

# Combine step in two passes to reduce shuffle size
# Pass 1: partial aggregation (local combine)
partial = (
    parsed.mapValues(lambda x: (x[0], x[1]))  # keep as (time, money)
           .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
)

# Pass 2: compute dollars/minute
money_per_minute = partial.mapValues(lambda x: x[1] / x[0] if x[0] > 0 else 0)
top_drivers = money_per_minute.sortBy(lambda x: -x[1]).take(10)

# ------------------ Economics + Performance Block ------------------
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60

try:
    num_exec = int(sc.getConf().get("spark.executor.instances", "2"))
    cores_per_exec = int(sc.getConf().get("spark.executor.cores", "1"))
    vcpus = num_exec * cores_per_exec + 2
except:
    vcpus = 4

cost_per_vcpu_min = 0.00079
estimated_cost = vcpus * cost_per_vcpu_min * runtime_minutes
total_records = df.count()

try:
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    path = sc._jvm.org.apache.hadoop.fs.Path(input_path)
    size_bytes = fs.getContentSummary(path).getLength()
    input_gb = size_bytes / (1024 ** 3)
except:
    input_gb = 1.0

records_per_sec = total_records / runtime_seconds if runtime_seconds > 0 else 0
cost_per_gb = estimated_cost / input_gb if input_gb > 0 else 0
cost_per_record = estimated_cost / total_records if total_records > 0 else 0
cost_efficiency = total_records / estimated_cost if estimated_cost > 0 else 0

summary = [("--- Top 10 Drivers ---", "")]
summary.extend([(d, round(v, 4)) for d, v in top_drivers])
summary.extend([
    ("--- Economic Metrics ---", ""),
    ("Estimated Cost (USD)", round(estimated_cost, 4)),
    ("Cost per GB Processed (USD/GB)", round(cost_per_gb, 6)),
    ("Cost per Record (USD/record)", round(cost_per_record, 10)),
    ("Cost Efficiency (records per $)", round(cost_efficiency, 2)),
    ("--- Performance Metrics ---", ""),
    ("Total Records", total_records),
    ("Records per Second", round(records_per_sec, 2)),
    ("Input Size (GB)", round(input_gb, 4)),
    ("Runtime (seconds)", round(runtime_seconds, 2)),
    ("Cluster Size (vCPUs)", vcpus)
])

sc.parallelize(summary, 1).saveAsTextFile(output_dir + "/results_summary")

# ------------------ Final Output ------------------
if top_drivers:
    print(f"\n✅ Job complete: Top driver = {top_drivers[0][0]} (${top_drivers[0][1]:.4f}/min)")
else:
    print("⚠️ No driver records found.")

print(f"💰 Estimated cost ${estimated_cost:.4f} | ⚙️ {records_per_sec:.2f} rec/s | 🖥 {vcpus} vCPUs")
spark.stop()
