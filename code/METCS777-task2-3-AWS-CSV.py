#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AWS EMR/YARN-safe PySpark script (CSV-only; no Parquet conversion)

Tasks:
  1) Top 10 taxis (medallions) by # of distinct drivers
  2) Top 10 drivers by dollars per minute

Also computes runtime, throughput, and rough cost metrics, and saves:
  - top_taxis_10 (JSON)
  - top_drivers_10 (JSON)
  - results_summary (text tuples, 1 part file)

Usage (on EMR/YARN):
  spark-submit taxi_tasks_df_csv_only.py s3://bucket/path/input.csv.bz2 s3://bucket/path/out/
  spark-submit taxi_tasks_df_csv_only.py s3://bucket/path/input.csv s3://bucket/path/out/ \
      --no-csv-header --no-csv-infer-schema

Notes:
  - Input stays as CSV; we do NOT write or convert to Parquet.
  - Ensure your IAM/EMR role has S3 read/write permissions.
"""

import sys
import time
import argparse
from pyspark.sql import SparkSession, functions as F, types as T


# ---------------------- CLI ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_path", help="S3 path to CSV input (e.g., s3://bucket/data/file.csv[.bz2|.gz])")
    p.add_argument("output_dir", help="S3 path to output dir (e.g., s3://bucket/out)")
    # CSV options
    p.add_argument("--csv-header", dest="csv_header", action="store_true", default=True,
                   help="CSV has header row (default: True)")
    p.add_argument("--no-csv-header", dest="csv_header", action="store_false",
                   help="CSV has no header row")
    p.add_argument("--csv-infer-schema", dest="csv_infer_schema", action="store_true", default=True,
                   help="Infer CSV schema (default: True)")
    p.add_argument("--no-csv-infer-schema", dest="csv_infer_schema", action="store_false",
                   help="Disable schema inference (treat columns as strings)")
    # Cost estimate knob
    p.add_argument("--cost-per-vcpu-min", type=float, default=0.00079,
                   help="Rough $/vCPU-minute for cost estimate (default: 0.00079)")
    return p.parse_args()


# ---------------------- Spark ----------------------
def make_spark():
    spark = (
        SparkSession.builder
        .appName("TaxiTasks-DF-AWS-CSV")
        # YARN-safe defaults; tune to your cluster
        .config("spark.sql.shuffle.partitions", "400")
        .config("spark.sql.caseSensitive", "false")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.speculation", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------- I/O ----------------------
def load_csv_df(spark, input_path, csv_header=False, csv_infer=True):
    # If schema inference is off, Spark will read columns as strings.
    return (spark.read
            .option("header", str(csv_header).lower())
            .option("inferSchema", str(csv_infer).lower())
            .option("mode", "DROPMALFORMED")
            .csv(input_path))


# ---------------------- Schema Normalization ----------------------
def normalize_schema(df):
    """
    Normalize to expected columns/types:
      medallion (string), hack_license (string),
      trip_time_in_secs (double), total_amount (double)

    Tries common NYC taxi column name variants and casts as needed.
    """
    cols = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    med = pick("medallion", "medallion_id", "med_id", "taxi_id")
    hack = pick("hack_license", "driver_license", "driver_id", "license")
    ttime = pick("trip_time_in_secs", "trip_time", "trip_seconds", "duration")
    tamt = pick("total_amount", "total_amt", "fare_amount_total", "fare_total", "total")

    missing = [name for name, v in [
        ("medallion", med), ("hack_license", hack),
        ("trip_time_in_secs", ttime), ("total_amount", tamt)
    ] if v is None]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {df.columns}"
        )

    out = (df
           .withColumn("medallion", F.col(med).cast(T.StringType()))
           .withColumn("hack_license", F.col(hack).cast(T.StringType()))
           .withColumn("trip_time_in_secs", F.col(ttime).cast(T.DoubleType()))
           .withColumn("total_amount", F.col(tamt).cast(T.DoubleType()))
           .select("medallion", "hack_license", "trip_time_in_secs", "total_amount"))
    return out


# ---------------------- Tasks ----------------------
def run_tasks(filtered):
    # Task 1: Top 10 taxis by distinct drivers
    t0 = time.time()
    top_taxis = (
        filtered.groupBy("medallion")
                .agg(F.approx_count_distinct("hack_license").alias("num_drivers"))
                .orderBy(F.desc("num_drivers"))
                .limit(10)
    )
    print("\n=== Task 1: Top 10 Taxis by #Distinct Drivers ===")
    top_taxis.show(truncate=False)
    task1_secs = time.time() - t0
    print(f"Task 1 exec time: {task1_secs:.3f}s")

    # Task 2: Top 10 drivers by money per minute
    t1 = time.time()
    drivers = (
        filtered.groupBy("hack_license")
                .agg(
                    F.sum("total_amount").alias("total_money"),
                    F.sum("trip_time_in_secs").alias("total_secs"),
                )
                .withColumn("total_minutes", F.col("total_secs") / F.lit(60.0))
                .filter(F.col("total_minutes") > 0)
                .withColumn("money_per_min", F.col("total_money") / F.col("total_minutes"))
    )
    top_drivers = (
        drivers.orderBy(F.desc("money_per_min"))
               .limit(10)
               .select("hack_license", "money_per_min")
    )
    print("\n=== Task 2: Top 10 Drivers by $/min ===")
    top_drivers.show(truncate=False)
    task2_secs = time.time() - t1
    print(f"Task 2 exec time: {task2_secs:.3f}s")

    return top_taxis, top_drivers, task1_secs, task2_secs


# ---------------------- Metrics ----------------------
def estimate_vcpus(sc):
    try:
        num_exec = int(sc.getConf().get("spark.executor.instances", "2"))
        cores_per_exec = int(sc.getConf().get("spark.executor.cores", "2"))
        # Add ~2 for the driver (rough default)
        return num_exec * cores_per_exec + 2
    except Exception:
        return 6

def get_input_size_gb(sc, input_path):
    try:
        fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
        path = sc._jvm.org.apache.hadoop.fs.Path(input_path)
        size_bytes = fs.getContentSummary(path).getLength()
        return size_bytes / (1024.0 ** 3)
    except Exception:
        return None

def save_summary(sc, output_dir, rows):
    sc.parallelize(rows, 1).saveAsTextFile(output_dir.rstrip("/") + "/results_summary")


# ---------------------- Main ----------------------
def main():
    args = parse_args()
    spark = make_spark()
    sc = spark.sparkContext

    start = time.time()
    print(f"Reading CSV from {args.input_path} … (header={args.csv_header}, inferSchema={args.csv_infer_schema})")

    raw = load_csv_df(spark, args.input_path, csv_header=args.csv_header, csv_infer=args.csv_infer_schema)
    # count can be expensive; comment out if not needed
    total_rows = raw.count()
    print(f"Loaded DF with {total_rows} rows and columns: {raw.columns}")

    filtered = (
        normalize_schema(raw)
        .filter((F.col("trip_time_in_secs") > 0) & F.col("total_amount").isNotNull())
        .cache()
    )
    filtered_rows = filtered.count()
    print(f"'filtered' rows after cleaning: {filtered_rows}")

    # Run tasks
    top_taxis, top_drivers, t1_secs, t2_secs = run_tasks(filtered)

    # Save outputs (JSON only; no Parquet conversion)
    (top_taxis.coalesce(1)
              .write.mode("overwrite")
              .json(args.output_dir.rstrip("/") + "/top_taxis_10"))
    (top_drivers.coalesce(1)
               .write.mode("overwrite")
               .json(args.output_dir.rstrip("/") + "/top_drivers_10"))
    print(f"\n💾 Saved JSON under:\n  {args.output_dir}/top_taxis_10\n  {args.output_dir}/top_drivers_10")

    end = time.time()
    runtime_seconds = end - start

    # Rough economics
    vcpus = estimate_vcpus(sc)
    cost_per_vcpu_min = float(args.cost_per_vcpu_min)
    est_cost = vcpus * cost_per_vcpu_min * (runtime_seconds / 60.0)

    input_gb = get_input_size_gb(sc, args.input_path)
    cost_per_gb = (est_cost / input_gb) if (input_gb and input_gb > 0) else None
    recs_per_sec = (filtered_rows / runtime_seconds) if runtime_seconds > 0 else None
    cost_per_record = (est_cost / filtered_rows) if filtered_rows > 0 else None
    cost_eff = (filtered_rows / est_cost) if est_cost > 0 else None

    # Summary tuples
    summary = [
        ("--- Job & Cluster ---", ""),
        ("Cluster Size (vCPUs)", vcpus),
        ("Runtime (seconds)", round(runtime_seconds, 2)),
        ("--- Dataset ---", ""),
        ("Total Records (filtered)", filtered_rows),
        ("Input Size (GB)", round(input_gb, 4) if input_gb else -1.0),
        ("Records per Second", round(recs_per_sec, 2) if recs_per_sec else -1.0),
        ("--- Economics (rough) ---", ""),
        ("Estimated Cost (USD)", round(est_cost, 6)),
        ("Cost per GB Processed (USD/GB)", round(cost_per_gb, 6) if cost_per_gb else -1.0),
        ("Cost per Record (USD/record)", round(cost_per_record, 12) if cost_per_record else -1.0),
        ("Cost Efficiency (records per $)", round(cost_eff, 2) if cost_eff else -1.0),
        ("--- Task Timings ---", ""),
        ("Task1 Runtime (seconds)", round(t1_secs, 3)),
        ("Task2 Runtime (seconds)", round(t2_secs, 3)),
    ]

    # Save printable summary to S3
    save_summary(sc, args.output_dir, summary)
    print(f"✅ Wrote summary to {args.output_dir}/results_summary")

    spark.stop()


if __name__ == "__main__":
    main()
