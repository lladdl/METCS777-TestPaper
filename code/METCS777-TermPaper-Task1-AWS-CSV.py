import sys, time, os
from pyspark import SparkContext
from datetime import datetime

if len(sys.argv) != 3:
    print("Usage: taxi_best_hour.py <s3_input_path> <s3_output_dir>", file=sys.stderr)
    sys.exit(-1)

input_path = sys.argv[1]   # e.g. s3://my-bucket/data/taxi-data.csv
output_dir = sys.argv[2]   # e.g. s3://my-bucket/output/taxi_best_hour

start_time = time.time()
sc = SparkContext(appName="TaxiAnalysisRDD-BestHour-S3")

# ------------------ Load and Transform ------------------
lines = sc.textFile(input_path)
header = lines.first()
data = lines.filter(lambda line: line != header)

def safe_split(line):
    try:
        return line.split(",")
    except:
        return None

fields = data.map(safe_split).filter(lambda f: f is not None)

def parse_line(parts):
    try:
        pickup_datetime = parts[2]
        trip_distance = float(parts[5])
        surcharge = float(parts[12])
        if trip_distance > 0:
            hour = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S").hour
            profit_ratio = surcharge / trip_distance
            return (hour, (profit_ratio, 1))
    except:
        return None
    return None

hour_stats = (fields.map(parse_line)
              .filter(lambda x: x is not None)
              .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])))

hour_avg = hour_stats.mapValues(lambda x: x[0] / x[1] if x[1] > 0 else 0)
best_hour = hour_avg.takeOrdered(1, key=lambda x: -x[1])

# ------------------ Stop Timer ------------------
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60

# ------------------ ECONOMICS METRICS ------------------
try:
    num_exec = int(sc.getConf().get("spark.executor.instances", "2"))
    cores_per_exec = int(sc.getConf().get("spark.executor.cores", "2"))
    vcpus = num_exec * cores_per_exec + 2
except:
    vcpus = 6  # EMR default if not specified

cost_per_vcpu_min = 0.00079  # adjust if using different EC2 type
estimated_cost = vcpus * cost_per_vcpu_min * runtime_minutes

total_records = data.count()

# AWS S3 size check uses Hadoop FS interface with "s3a://"
try:
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    path = sc._jvm.org.apache.hadoop.fs.Path(input_path)
    input_size_bytes = fs.getContentSummary(path).getLength()
    input_gb = input_size_bytes / (1024 ** 3)
except Exception as e:
    input_gb = 1.0  # fallback if permissions restricted

records_per_sec = total_records / runtime_seconds if runtime_seconds > 0 else 0
cost_per_gb = estimated_cost / input_gb if input_gb > 0 else 0
cost_per_record = estimated_cost / total_records if total_records > 0 else 0
cost_efficiency = total_records / estimated_cost if estimated_cost > 0 else 0

# ------------------ SAVE RESULTS ------------------
summary = [
    ("--- Best Hour Result ---", ""),
    ("Best Hour", best_hour[0][0]),
    ("Average Ratio", round(best_hour[0][1], 4)),
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
]

sc.parallelize(summary, 1).saveAsTextFile(output_dir + "/results_summary")
print(f"\n✅ Job complete: Best Hour = {best_hour[0][0]}, Avg Ratio = {best_hour[0][1]:.4f}")
print(f"💰 Estimated cost ${estimated_cost:.4f} | ⚙️ {records_per_sec:.2f} rec/s | 🖥 {vcpus} vCPUs")

sc.stop()
