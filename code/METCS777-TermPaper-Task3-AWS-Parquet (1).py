#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMR/S3-safe CSV➜Parquet staging + Task 2 (Top 10 drivers by $/min)
Saves:
  <out>/task2_top_drivers_10/  (JSON)
  <out>/results_summary/       (text tuples, 1 part)
"""

import sys, time
from pyspark.sql import SparkSession, functions as F, types as T

# --- Args -------------------------------------------------------------
if len(sys.argv) not in (3, 4):
    print("Usage: taxi_csv_to_parquet_task2_top_drivers.py <input_csv_s3_path> <output_dir_s3> [staging_parquet_s3_path]",
          file=sys.stderr)
    sys.exit(-1)

csv_path = sys.argv[1]
out_dir  = sys.argv[2]
stage_parquet = sys.argv[3] if len(sys.argv) == 4 else f"{out_dir.rstrip('/')}/staging_parquet_task2"

# --- Spark ------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("Taxi-CSV-to-Parquet-Task2-EMR-SAFE")
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
    .config("spark.hadoop.fs.s3a.fast.upload", "true")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.connection.maximum", "200")
    .getOrCreate()
)
sc = spark.sparkContext
spark.sparkContext.setLogLevel("WARN")
start_time = time.time()

# --- Expected columns & dtypes ---------------------------------------
expected_cols = [
    "medallion","hack_license","pickup_datetime","dropoff_datetime",
    "trip_time_in_secs","trip_distance","pickup_longitude","pickup_latitude",
    "dropoff_longitude","dropoff_latitude","payment_type","fare_amount",
    "surcharge","mta_tax","tip_amount","tolls_amount","total_amount"
]
schema = T.StructType([
    T.StructField("medallion", T.StringType(), True),
    T.StructField("hack_license", T.StringType(), True),
    T.StructField("pickup_datetime", T.StringType(), True),
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

# --- Load CSV ------------------------------
raw = (spark.read
       .option("header", "false")
       .option("multiLine", "false")
       .option("mode", "PERMISSIVE")
       .option("enforceSchema", "false")
       .csv(csv_path))

print("CSV columns:", raw.columns)

if len(raw.columns) != len(expected_cols) or any(c is None or c == "" for c in raw.columns):
    print("Detected missing/incorrect header — renaming columns…")
    raw = raw.toDF(*expected_cols[:len(raw.columns)])

for c in expected_cols:
    if c not in raw.columns:
        raw = raw.withColumn(c, F.lit(None).cast(schema[c].dataType))

df_csv = raw.select([F.col(c).cast(schema[c].dataType) for c in expected_cols])

# --- Write compact Parquet staging -----------------------------------
(df_csv
 .repartition(4)
 .write.mode("overwrite")
 .option("compression", "snappy")
 .parquet(stage_parquet))
print(f"Staged Parquet: {stage_parquet}")

# --- Read staged Parquet------------------------
df = spark.read.parquet(stage_parquet).repartition(4).cache()
total_records = df.count()
filtered = (
    df.select("hack_license","trip_time_in_secs","total_amount")
      .withColumn("trip_time_in_secs", F.col("trip_time_in_secs").cast(T.DoubleType()))
      .withColumn("total_amount", F.col("total_amount").cast(T.DoubleType()))
      .filter((F.col("trip_time_in_secs") > 0) & F.col("total_amount").isNotNull())
      .cache()
)
filtered_rows = filtered.count()
print(f"Rows after cleaning: {filtered_rows} / staged total: {total_records}")

# --- Task 3-----------------------------------------------------------
t2_start = time.time()
drivers = (filtered.groupBy("hack_license")
           .agg(F.sum("total_amount").alias("total_money"),
                F.sum("trip_time_in_secs").alias("total_secs"))
           .withColumn("total_minutes", F.col("total_secs") / F.lit(60.0))
           .filter(F.col("total_minutes") > 0)
           .withColumn("money_per_min", F.col("total_money") / F.col("total_minutes")))
top_drivers = (drivers.orderBy(F.desc("money_per_min"))
                      .limit(10)
                      .select("hack_license","money_per_min"))
top_drivers.show(truncate=False)
t2_secs = time.time() - t2_start

# --- Write outputs ----------------------------------------------------
outbase = out_dir.rstrip('/')
(top_drivers.coalesce(1).write.mode("overwrite").json(f"{outbase}/task2_top_drivers_10"))
print(f"Wrote JSON: {outbase}/task2_top_drivers_10")

# --- Metrics/summary (text tuples) -----------------------------------
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60.0

try:
    num_exec = int(sc.getConf().get("spark.executor.instances","2"))
    cores    = int(sc.getConf().get("spark.executor.cores","1"))
    vcpus    = num_exec * cores + 2
except Exception:
    vcpus = 4

cost_per_vcpu_min = 0.00079
estimated_cost = vcpus * cost_per_vcpu_min * runtime_minutes

# Input size
try:
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    path = sc._jvm.org.apache.hadoop.fs.Path(csv_path)
    size_bytes = fs.getContentSummary(path).getLength()
    input_gb = size_bytes / (1024.0 ** 3)
except Exception:
    input_gb = 1.0

records_per_sec = (filtered_rows / runtime_seconds) if runtime_seconds > 0 else 0.0
cost_per_gb     = (estimated_cost / input_gb) if input_gb > 0 else 0.0
cost_per_record = (estimated_cost / filtered_rows) if filtered_rows > 0 else 0.0
cost_eff        = (filtered_rows / estimated_cost) if estimated_cost > 0 else 0.0

summary = [
    ("Task", "Top 10 drivers by $/min"),
    ("CSV Input", csv_path),
    ("Staging Parquet", stage_parquet),
    ("Rows (staged)", total_records),
    ("Rows (filtered)", filtered_rows),
    ("Runtime (sec)", round(runtime_seconds, 2)),
    ("Task2 Sec", round(t2_secs, 3)),
    ("Records/sec", round(records_per_sec, 2)),
    ("Input Size (GB)", round(input_gb, 4)),
    ("vCPUs est.", vcpus),
    ("Est. Cost (USD)", round(estimated_cost, 6)),
    ("Cost/GB", round(cost_per_gb, 6)),
    ("Cost/Record", round(cost_per_record, 12)),
    ("Cost Efficiency (records/$)", round(cost_eff, 2)),
    ("Output JSON", f"{outbase}/task2_top_drivers_10"),
]
(sc.parallelize(summary, 1)
   .saveAsTextFile(f"{outbase}/results_summary"))

print(f"Done. Summary: {outbase}/results_summary")
spark.stop()
