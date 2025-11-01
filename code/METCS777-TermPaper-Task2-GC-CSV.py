import sys, time, os
from pyspark import SparkContext

if len(sys.argv) != 3:
    print("Usage: taxi_top_drivers.py <input_file_path> <output_dir>", file=sys.stderr)
    sys.exit(-1)

input_path = sys.argv[1]
output_dir = sys.argv[2]

start_time = time.time()
sc = SparkContext(appName="TaxiAnalysisRDD-TopDrivers")

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
        hack_license = parts[1]
        trip_time_secs = float(parts[4])
        total_amount = float(parts[16])
        trip_time_mins = trip_time_secs / 60.0
        if trip_time_mins > 0:
            return (hack_license, (trip_time_mins, total_amount))
    except:
        return None
    return None

driver_stats = (fields.map(parse_line)
                .filter(lambda x: x is not None)
                .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])))

money_per_minute = driver_stats.mapValues(lambda x: x[1] / x[0] if x[0] > 0 else 0)
top_drivers = money_per_minute.takeOrdered(10, key=lambda x: -x[1])

# --- Economics + Performance Block (same for all scripts) ---
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60

try:
    num_exec = int(sc.getConf().get("spark.executor.instances", "2"))
    cores_per_exec = int(sc.getConf().get("spark.executor.cores", "2"))
    vcpus = num_exec * cores_per_exec + 2
except:
    vcpus = 6

cost_per_vcpu_min = 0.00079
estimated_cost = vcpus * cost_per_vcpu_min * runtime_minutes
total_records = data.count()

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
print(f"\n✅ Job complete: Top driver = {top_drivers[0][0]} (${top_drivers[0][1]:.4f}/min)")
print(f"💰 Estimated cost ${estimated_cost:.4f} | ⚙️ {records_per_sec:.2f} rec/s | 🖥 {vcpus} vCPUs")
sc.stop()
