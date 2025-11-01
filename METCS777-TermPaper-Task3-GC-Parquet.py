#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YARN-safe Parquet Top Taxis
Finds the top 10 most active taxi medallions safely on small clusters.
Now uses fixed column names ('medallion', 'hack_license') without detection.
"""

import sys, time
from pyspark.sql import SparkSession, functions as F

# ------------------ Setup ------------------
if len(sys.argv) != 3:
    print("Usage: taxi_top_taxis_parquet_safe.py <input_parquet_path> <output_dir>", file=sys.stderr)
    sys.exit(-1)

input_path, output_dir = sys.argv[1], sys.argv[2]

start_time = time.time()
spark = (
    SparkSession.builder
    .appName("TaxiAnalysisRDD-TopTaxis-Parquet-SAFE")
    # --- YARN-friendly tuning ---
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

# If first row became header, rename it
expected_cols = [
    "medallion", "hack_license", "pickup_datetime", "dropoff_datetime",
    "trip_time_in_secs", "trip_distance", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "payment_type", "fare_amount",
    "surcharge", "mta_tax", "tip_amount", "tolls_amount", "total_amount"
]
if df.columns[0] not in expected_cols:
    print("⚙️  Detected missing header — renaming columns...")
    df = df.toDF(*expected_cols)

print("✅ Using columns:", df.columns[:5], "...")

# Keep few partitions, cache safely
df = df.repartition(4).cache()
df.count()  # materialize once

# ------------------ Compute Top Taxis safely ------------------
agg_df = (
    df.select("medallion", "hack_license")
      .where(F.col("medallion").isNotNull() & F.col("hack_license").isNotNull())
      .distinct()
      .groupBy("medallion")
      .agg(F.count("*").alias("license_count"))
      .orderBy(F.desc("license_count"))
)

top10 = agg_df.limit(10).collect()

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
    input_size_bytes = fs.getContentSummary(path).getLength()
    input_gb = input_size_bytes / (1024 ** 3)
except:
    input_gb = 1.0

records_per_sec = total_records / runtime_seconds if runtime_seconds > 0 else 0
cost_per_gb = estimated_cost / input_gb if input_gb > 0 else 0
cost_per_record = estimated_cost / total_records if total_records > 0 else 0
cost_efficiency = total_records / estimated_cost if estimated_cost > 0 else 0

# ------------------ Save Results ------------------
summary = [("--- Top 10 Taxis ---", "")]
summary.extend([(row["medallion"], row["license_count"]) for row in top10])
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
if top10:
    print(f"\n✅ Job complete: Top taxi = {top10[0]['medallion']} ({top10[0]['license_count']} unique licenses)")
else:
    print("⚠️ No taxi records found.")

print(f"💰 Estimated cost ${estimated_cost:.4f} | ⚙️ {records_per_sec:.2f} rec/s | 🖥 {vcpus} vCPUs")
spark.stop()
