#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMR/S3-safe CSV➜Parquet Best-Hour Analysis
------------------------------------------
• Reads CSV from s3a://...
• Normalizes/renames headers, light schema casting
• Writes compact Parquet staging set (snappy) to S3
• Computes Best Hour on Parquet with small shuffle footprint
• Logs cost, runtime, throughput, and efficiency
"""

import sys, time
from datetime import datetime
from pyspark.sql import SparkSession, functions as F, types as T

# --- Args -------------------------------------------------------------
if len(sys.argv) not in (3, 4):
    print("Usage: taxi_best_hour_csv_to_parquet_s3.py <input_csv_s3_path> <output_dir_s3> [staging_parquet_s3_path]",
          file=sys.stderr)
    sys.exit(-1)

csv_path = sys.argv[1]          # e.g., s3a://my-bucket/nyc-taxi/raw.csv.gz (or folder)
out_dir  = sys.argv[2]          # e.g., s3a://my-bucket/nyc-taxi/out
stage_parquet = sys.argv[3] if len(sys.argv) == 4 else f"{out_dir.rstrip('/')}/staging_parquet"

# --- Spark Session (EMR/YARN-safe tuning) -----------------------------
spark = (
    SparkSession.builder
    .appName("Taxi-BestHour-CSV-to-Parquet-EMR-SAFE")
    # Small, steady footprint; let EMR/YARN handle containers via instance profile
    .config("spark.executor.instances", "2")
    .config("spark.executor.cores", "1")
    .config("spark.executor.memory", "2g")
    .config("spark.executor.memoryOverhead", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.default.parallelism", "4")
    .config("spark.task.maxFailures", "8")
    .config("spark.yarn.maxAppAttempts", "2")
    .config("spark.network.timeout", "800s")
    .config("spark.executor.heartbeatInterval", "60s")
    .config("spark.speculation", "false")
    # S3A best practices (EMR role provides creds)
    .config("spark.hadoop.fs.s3a.fast.upload", "true")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.connection.maximum", "200")
    .getOrCreate()
)
sc = spark.sparkContext
start_time = time.time()

# --- Expected schema / columns ---------------------------------------
expected_cols = [
    "medallion", "hack_license", "pickup_datetime", "dropoff_datetime",
    "trip_time_in_secs", "trip_distance", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "payment_type", "fare_amount",
    "surcharge", "mta_tax", "tip_amount", "tolls_amount", "total_amount"
]

schema = T.StructType([
    T.StructField("medallion", T.StringType(), True),
    T.StructField("hack_license", T.StringType(), True),
    T.StructField("pickup_datetime", T.StringType(), True),   # cast later if needed
    T.StructField("dropoff_datetime", T.StringType(), True),
    T.StructField("trip_time_in_secs", T.IntegerType(), True),
    T.StructField("trip_distance", T.DoubleType(), True),
    T.StructField("pickup_longitude", T.DoubleType(), True),
    T.StructField("pickup_latitude", T.DoubleType(), True),
    T.StructField("dropoff_longitude", T.DoubleType(), True),
    T.StructField("dropoff_latitude", T.DoubleType(), True),
    T.StructField("payment_type", T.StringType(), True),
    T.StructField("fare_amount", T.DoubleType(), True),
    T.StructField("surcharge", T.DoubleType(), True),
    T.StructField("mta_tax", T.DoubleType(), True),
    T.StructField("tip_amount", T.DoubleType(), True),
    T.StructField("tolls_amount", T.DoubleType(), True),
    T.StructField("total_amount", T.DoubleType(), True),
])

# --- Load CSV from S3 -------------------------------------------------
# Be tolerant of messy CSVs; header=True to attempt proper names
raw = (
    spark.read
         .option("header", "false")
         .option("multiLine", "false")
         .option("mode", "PERMISSIVE")
         .option("enforceSchema", "false")
         .csv(csv_path)
)

print("🪣 CSV columns:", raw.columns)

# If header row ended up as data or columns are wrong, fix names
if len(raw.columns) != len(expected_cols) or any(c is None or c == "" for c in raw.columns):
    print("⚙️  Detected missing/incorrect header — renaming columns...")
    raw = raw.toDF(*expected_cols[:len(raw.columns)])  # trim if needed

# Align to full expected schema (add missing columns as nulls)
for c in expected_cols:
    if c not in raw.columns:
        raw = raw.withColumn(c, F.lit(None).cast(schema[c].dataType))

# Reorder & cast
df_csv = raw.select([F.col(c).cast(schema[c].dataType) for c in expected_cols])

# --- Light cleanup / normalize pickup_datetime ------------------------
# Keep string; we’ll parse hour later safely regardless of string/timestamp
df_csv = df_csv.repartition(4).cache()
df_csv.count()  # materialize

# --- Write compact Parquet staging set to S3 --------------------------
# Small number of files to cut driver/executor memory & shuffle overhead
(df_csv
 .repartition(4)                                      # compact
 .write
 .mode("overwrite")
 .option("compression", "snappy")
 .parquet(stage_parquet))

print(f"📦 Wrote staging Parquet to: {stage_parquet}")

# --- Read Parquet staging (analysis input) ----------------------------
df = spark.read.parquet(stage_parquet).repartition(4).cache()
df.count()

# --- RDD for Best Hour -----------------------------------------------
rdd = df.select("pickup_datetime", "trip_distance", "surcharge").rdd.coalesce(4)

def parse_row(row):
    try:
        pickup = row[0]
        trip_distance = row[1]
        surcharge = row[2]
        if trip_distance is None or surcharge is None or trip_distance <= 0:
            return None

        # Parse hour defensively (str or timestamp)
        if isinstance(pickup, datetime):
            hour = pickup.hour
        elif isinstance(pickup, str):
            s = pickup.strip()
            # Try common formats quickly
            for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    hour = datetime.strptime(s[:19], fmt).hour
                    break
                except Exception:
                    hour = None
            if hour is None:
                return None
        else:
            return None

        ratio = float(surcharge) / float(trip_distance)
        return (hour, (ratio, 1))
    except Exception:
        return None

hour_stats = (
    rdd.map(parse_row)
       .filter(lambda x: x is not None)
       .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
)

hour_avg = hour_stats.mapValues(lambda x: x[0] / x[1] if x[1] > 0 else 0.0)

best_hour = hour_avg.sortBy(lambda x: -x[1]).first() if not hour_avg.isEmpty() else None

# --- Economics & Performance -----------------------------------------
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60.0

try:
    num_exec = int(sc.getConf().get("spark.executor.instances", "2"))
    cores_per_exec = int(sc.getConf().get("spark.executor.cores", "1"))
    vcpus = num_exec * cores_per_exec + 2  # +2 for driver/YARN AM approx
except Exception:
    vcpus = 4

# Very rough EMR on-demand vCPU-min cost proxy (adjust if you want spot)
cost_per_vcpu_min = 0.00079
estimated_cost = vcpus * cost_per_vcpu_min * runtime_minutes

total_records = df.count()

# Input size by querying S3 via Hadoop FS
try:
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    path = sc._jvm.org.apache.hadoop.fs.Path(csv_path)
    size_bytes = fs.getContentSummary(path).getLength()
    input_gb = size_bytes / (1024.0 ** 3)
except Exception:
    input_gb = 1.0

records_per_sec = (total_records / runtime_seconds) if runtime_seconds > 0 else 0.0
cost_per_gb = (estimated_cost / input_gb) if input_gb > 0 else 0.0
cost_per_record = (estimated_cost / total_records) if total_records > 0 else 0.0
cost_efficiency = (total_records / estimated_cost) if estimated_cost > 0 else 0.0

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
    ("Cluster Size (vCPUs)", vcpus),
    ("Staging Parquet", stage_parquet)
])

# Write summary to S3
(spark.sparkContext
     .parallelize(summary, 1)
     .saveAsTextFile(out_dir.rstrip('/') + "/results_summary"))

if best_hour:
    print(f"\n✅ Job complete: Best Hour = {best_hour[0]}, Avg Ratio = {best_hour[1]:.4f}")
else:
    print("\n⚠️  No valid records found to compute best hour.")
print(f"💰 Estimated cost ${estimated_cost:.4f} | ⚙️  {records_per_sec:.2f} rec/s | 🖥  {vcpus} vCPUs")

spark.stop()
