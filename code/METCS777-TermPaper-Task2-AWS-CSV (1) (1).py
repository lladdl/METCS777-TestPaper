#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1 only: Top 10 medallions by # of distinct drivers (EMR/YARN-safe)

- RDD-style CSV load (textFile → drop header → split(','))
- Auto schema for generic _c* (supports classic21/20/17) or --schema-hint
- Cleans & filters, runs Task 1, writes JSON + text summary
"""

import sys
import time
import argparse
import traceback
from pyspark.sql import SparkSession, functions as F, types as T

# ---------------------- CLI ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_path", help="S3 path to CSV (e.g., s3://bucket/data.csv[.bz2|.gz])")
    p.add_argument("output_dir", help="S3 output dir (e.g., s3://bucket/out)")
    p.add_argument("--csv-header", dest="csv_header", action="store_true", default=True,
                   help="CSV has header row; drop first line (default: True)")
    p.add_argument("--no-csv-header", dest="csv_header", action="store_false",
                   help="CSV has no header row")
    p.add_argument("--schema-hint", choices=["classic21","classic20","classic17"], default=None,
                   help="Force positional schema for generic _c* input")
    p.add_argument("--cost-per-vcpu-min", type=float, default=0.00079,
                   help="Approx $/vCPU-minute for cost estimate (default: 0.00079)")
    return p.parse_args()

# ---------------------- Spark ----------------------
def make_spark():
    spark = (
        SparkSession.builder
        .appName("Taxi-Task1-TopTaxis-CSV-RDDLoader")
        .config("spark.sql.shuffle.partitions", "400")
        .config("spark.sql.caseSensitive", "false")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.speculation", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

# ---------------------- Loader  ----------------------
def load_csv_df_rdd_style(spark, input_path, csv_header=True):
    sc = spark.sparkContext
    lines = sc.textFile(input_path)

    if csv_header:
        header = lines.first()
        without_header = lines.filter(lambda l: l != header)
    else:
        without_header = lines

    def safe_split(line):
        try:
            return line.split(",")  
        except Exception:
            return None

    parts = without_header.map(safe_split).filter(lambda x: x is not None)
    max_len = parts.map(len).max()
    padded = parts.map(lambda r: r + [None] * (max_len - len(r)))
    cols = [f"_c{i}" for i in range(max_len)]
    return padded.toDF(cols)  

# ---------------------- Schema helpers ----------------------
CLASSIC21 = [
    "medallion","hack_license","vendor_id","rate_code","store_and_fwd_flag",
    "pickup_datetime","dropoff_datetime","passenger_count","trip_time_in_secs",
    "trip_distance","pickup_longitude","pickup_latitude","dropoff_longitude",
    "dropoff_latitude","payment_type","fare_amount","surcharge","mta_tax",
    "tip_amount","tolls_amount","total_amount"
]
def classic20_names():
    base = CLASSIC21[:-2]        
    return base[:-1] + ["total_amount"]  

CLASSIC17 = [
    "medallion","hack_license","vendor_id","rate_code","store_and_fwd_flag",
    "pickup_datetime","dropoff_datetime","passenger_count","trip_time_in_secs",
    "trip_distance","pickup_longitude","pickup_latitude","dropoff_longitude",
    "dropoff_latitude","payment_type","fare_amount","total_amount"
]

REQUIRED = ["medallion","hack_license","trip_time_in_secs","total_amount"]

def rename_generic_by_scheme(df, scheme):
    n = len(df.columns)
    if scheme == "classic21":
        mapping = CLASSIC21
        if n < 21: raise ValueError("classic21 requires ≥21 columns")
    elif scheme == "classic20":
        mapping = classic20_names()
        if n < 20: raise ValueError("classic20 requires ≥20 columns")
    elif scheme == "classic17":
        mapping = CLASSIC17
        if n < 17: raise ValueError("classic17 requires ≥17 columns")
    else:
        raise ValueError(f"Unknown schema hint: {scheme}")
    out = df
    for i, name in enumerate(mapping):
        if i < n:
            out = out.withColumnRenamed(f"_c{i}", name)
    return out

def auto_rename_generic(df):
    n = len(df.columns)
    if n >= 21: return rename_generic_by_scheme(df, "classic21"), "classic21"
    if n == 20: return rename_generic_by_scheme(df, "classic20"), "classic20"
    if n == 17: return rename_generic_by_scheme(df, "classic17"), "classic17"
    return df, None

def normalize_schema(df, schema_hint=None):
    cols = df.columns
    lower = {c.lower(): c for c in cols}

    if all(c.startswith("_c") for c in cols):
        if schema_hint:
            df = rename_generic_by_scheme(df, schema_hint)
        else:
            df, picked = auto_rename_generic(df)
        cols = df.columns
        lower = {c.lower(): c for c in cols}

    def pick(*cands):
        for c in cands:
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    med  = pick("medallion","medallion_id","med_id","taxi_id")
    hack = pick("hack_license","driver_license","driver_id","license")
    tsec = pick("trip_time_in_secs","trip_time","trip_seconds","duration")
    tot  = pick("total_amount","total_amt","fare_amount_total","fare_total","total")

    missing = [n for n,v in [("medallion",med),("hack_license",hack),("trip_time_in_secs",tsec),("total_amount",tot)] if v is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Seen: {cols}")

    return (df
            .withColumn("medallion", F.col(med).cast(T.StringType()))
            .withColumn("hack_license", F.col(hack).cast(T.StringType()))
            .withColumn("trip_time_in_secs", F.col(tsec).cast(T.DoubleType()))
            .withColumn("total_amount", F.col(tot).cast(T.DoubleType()))
            .select(*REQUIRED))

# ---------------------- Utils ----------------------
def estimate_vcpus(sc):
    try:
        e = int(sc.getConf().get("spark.executor.instances","2"))
        c = int(sc.getConf().get("spark.executor.cores","1"))
        return e*c + 1
    except Exception:
        return 3

def get_input_size_gb(sc, input_path):
    try:
        fs   = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
        path = sc._jvm.org.apache.hadoop.fs.Path(input_path)
        return fs.getContentSummary(path).getLength() / (1024.0**3)
    except Exception:
        return None

def save_summary(sc, outdir, rows):
    sc.parallelize(rows, 1).saveAsTextFile(outdir.rstrip("/") + "/results_summary")

# ---------------------- Task 1 ----------------------
def main():
    args = parse_args()
    spark = make_spark()
    sc = spark.sparkContext
    start = time.time()

    try:
        print(f"[DEBUG] Task1 start | input={args.input_path}")
        df0 = load_csv_df_rdd_style(spark, args.input_path, csv_header=args.csv_header)

        # Diagnostics
        print(f"[DIAG] Columns ({len(df0.columns)}): {df0.columns[:32]}")
        fr = df0.limit(1).collect()
        if fr: print(f"[DIAG] First row len={len(fr[0])}: {fr[0]}")

        # Normalize & filter
        df = (normalize_schema(df0, schema_hint=args.schema_hint)
              .filter((F.col("trip_time_in_secs") > 0) & F.col("total_amount").isNotNull())
              .cache())

        raw_rows = df0.count()
        rows     = df.count()
        print(f"[INFO] Raw rows: {raw_rows} | Filtered rows: {rows}")

        # ---- Task 1 ----
        t0 = time.time()
        top_taxis = (df.groupBy("medallion")
                       .agg(F.approx_count_distinct("hack_license").alias("num_drivers"))
                       .orderBy(F.desc("num_drivers")).limit(10))
        top_taxis.show(truncate=False)
        task_secs = time.time() - t0
        outbase = args.output_dir.rstrip('/')
        (top_taxis.coalesce(1).write.mode("overwrite").json(f"{outbase}/task1_top_taxis_10"))

        end = time.time()
        runtime = end - start
        vcpus = estimate_vcpus(sc)
        cost  = vcpus * float(args.cost_per_vcpu_min) * (runtime/60.0)
        in_gb = get_input_size_gb(sc, args.input_path)
        rps   = (rows/runtime) if runtime>0 else None

        summary = [
            ("Task", "Top 10 medallions by distinct drivers"),
            ("Runtime_sec", round(runtime,2)),
            ("Task1_sec", round(task_secs,3)),
            ("Rows_raw", raw_rows),
            ("Rows_filtered", rows),
            ("Input_GB", round(in_gb,4) if in_gb else -1.0),
            ("vCPU_min_est", round(vcpus*(runtime/60.0),2)),
            ("Cost_USD_est", round(cost,6)),
            ("Records_per_sec", round(rps,2) if rps else -1.0),
            ("Output_JSON", f"{outbase}/task1_top_taxis_10"),
        ]
        save_summary(sc, args.output_dir, summary)
        print(f"Wrote Task1 JSON to {outbase}/task1_top_taxis_10 and summary to {outbase}/results_summary")

    except Exception:
        print("[ERROR] Unhandled exception in Task1:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr); sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
