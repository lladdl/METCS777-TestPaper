#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YARN-safe Parquet Best-Hour Analysis
------------------------------------
• Auto-renames missing headers
• Keeps memory & shuffle use small
• Adds Spark configs to avoid executor kills (exit 143)
• Logs cost, runtime, throughput, and efficiency
"""

import sys, time
from datetime import datetime
from pyspark.sql import SparkSession

# --- Args -------------------------------------------------------------
if len(sys.argv) != 3:
    print("Usage: taxi_best_hour_parquet_safe.py <input_parquet_path> <output_dir>", file=sys.stderr)
    sys.exit(-1)

input_path = sys.argv[1]
output_dir = sys.argv[2]

# --- Spark Session (YARN-safe tuning) --------------------------------
spark = (
    SparkSession.builder
    .appName("TaxiAnalysisRDD-BestHour-Parquet-SAFE")
    .config("spark.executor.instances", "2")          # match 2 workers
    .config("spark.executor.cores", "1")              # 1 core each = small footprint
    .config("spark.executor.memory", "2g")
    .config("spark.executor.memoryOverhead", "1g")
    .config("spark.sql.shuffle.partitions", "4")      # shrink shuffle fan-out
    .config("spark.default.parallelism", "4")
    .config("spark.task.maxFailures", "8")
    .config("spark.yarn.maxAppAttempts", "2")
    .config("spark.network.timeout", "800s")
    .config("spark.executor.heartbeatInterval", "60s")
    .config("spark.speculation", "false")
    .getOrCreate()
)
sc = spark.sparkContext

start_time = time.time()

# --- Load Parquet ----------------------------------------------------
df = spark.read.parquet(input_path)
print("🪣 Original columns:", df.columns)

# if first row of data became header, rename properly
expected_cols = [
    "medallion", "hack_license", "pickup_datetime", "dropoff_datetime",
    "trip_time_in_secs", "trip_distance", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "payment_type", "fare_amount",
    "surcharge", "mta_tax", "tip_amount", "tolls_amount", "total_amount"
]
if df.columns[0] not in expected_cols:
    print("⚙️  Detected missing header — renaming columns...")
    df = df.toDF(*expected_cols)



# keep small number of partitions, cache once
df = df.repartition(4).cache()
df.count()   # force materialization safely

rdd = df.select("pickup_datetime", "trip_distance", "surcharge").rdd.coalesce(4)

# --- Compute Best Hour -----------------------------------------------
def parse_row(row):
    try:
        pickup = row.asDict().get("pickup_datetime")
        trip_distance = row.asDict().get("trip_distance")
        surcharge = row.asDict().get("surcharge")
        if trip_distance is None or surcharge is None or trip_distance <= 0:
            return None
        if isinstance(pickup, datetime):
            hour = pickup.hour
        elif isinstance(pickup, str):
            hour = datetime.strptime(pickup[:19], "%Y-%m-%d %H:%M:%S").hour
        else:
            return None
        ratio = float(surcharge) / float(trip_distance)
        return (hour, (ratio, 1))
    except:
        return None

hour_stats = (
    rdd.map(parse_row)
       .filter(lambda x: x is not None)
       .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
)
hour_avg = hour_stats.mapValues(lambda x: x[0] / x[1] if x[1] > 0 else 0)
best_hour = hour_avg.sortBy(lambda x: -x[1]).first() if not hour_avg.isEmpty() else None

# --- Economics & Performance -----------------------------------------
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

summary = [("--- Best Hour Result ---", "")]
if best_hour:
    summary.extend([
        ("Best Hour", best_hour[0]),
        ("Average Ratio", round(best_hour[1], 4))
    ])
else:
    summary.append(("No valid records found", ""))

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

if best_hour:
    print(f"\n✅ Job complete: Best Hour = {best_hour[0]}, Avg Ratio = {best_hour[1]:.4f}")
else:
    print("\n⚠️  No valid records found to compute best hour.")
print(f"💰 Estimated cost ${estimated_cost:.4f} | ⚙️  {records_per_sec:.2f} rec/s | 🖥  {vcpus} vCPUs")

spark.stop()
