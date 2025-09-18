import os
import json
from collections import defaultdict

def analyze_results(root_dir):
    city_list = ["los_angeles", "miami", "toronto", "singapore", "auckland",
                 "rome", "lisboa", "beijing", "hangzhou", "budapest", "geneva"]

    model_stats = defaultdict(lambda: {
        "count": 0,
        "acc_short_sum": 0.0,
        "acc_long_sum": 0.0,
        "token_short_sum": 0,
        "token_long_sum": 0,
        "via_stops_score_sum": 0.0,
        "num_via_stop_score_sum": 0.0,
        "map_score_short_sum": 0.0,
        "map_score_long_sum": 0.0
    })

    for dirpath, _, filenames in os.walk(root_dir):
        rel_path = os.path.relpath(dirpath, root_dir)
        parts = rel_path.split(os.sep)

        if len(parts) >= 2:
            country, city = parts[0], parts[1]
            if city not in city_list:
                continue
        else:
            continue

        for filename in filenames:
            if filename.endswith(".json") and "_" in filename:
                try:
                    i_sample, model_shortname = filename[:-5].split("_", 1)
                    filepath = os.path.join(dirpath, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        acc_short = data.get("acc short", 0.0)
                        acc_long = data.get("acc long", 0.0)
                        token_short = data.get("short token count", 0)
                        token_long = data.get("long token count", 0)
                        via_stops_score = data.get("save via stops score", 0.0)
                        num_via_stop_score = data.get("save num via stop score", 0.0)
                        map_score_short = data.get("best map api score short", 0.0)
                        map_score_long = data.get("best map api score long", 0.0)

                        stats = model_stats[model_shortname]
                        stats["count"] += 1
                        stats["acc_short_sum"] += acc_short
                        stats["acc_long_sum"] += acc_long
                        stats["token_short_sum"] += token_short
                        stats["token_long_sum"] += token_long
                        stats["via_stops_score_sum"] += via_stops_score
                        stats["num_via_stop_score_sum"] += num_via_stop_score
                        stats["map_score_short_sum"] += map_score_short
                        stats["map_score_long_sum"] += map_score_long

                except Exception as e:
                    print(f"❌ Error processing {filename} in {dirpath}: {e}")

    if not model_stats:
        print("⚠️ 没有找到任何符合格式的JSON文件")
        return

    for model, stats in model_stats.items():
        count = stats["count"]
        print(f"📊 Model: {model}")
        print(f"  Total files: {count}")
        print(f"  Average acc_short: {stats['acc_short_sum'] / count:.4f}")
        print(f"  Average acc_long: {stats['acc_long_sum'] / count:.4f}")
        print(f"  Average token_short: {stats['token_short_sum'] / count:.2f}")
        print(f"  Average token_long: {stats['token_long_sum'] / count:.2f}")
        print(f"  Average save via stops score: {stats['via_stops_score_sum'] / count:.4f}")
        print(f"  Average save num via stop score: {stats['num_via_stop_score_sum'] / count:.4f}")
        print(f"  Average best map api score short: {stats['map_score_short_sum'] / count:.4f}")
        print(f"  Average best map api score long: {stats['map_score_long_sum'] / count:.4f}")
        print()


if __name__ == "__main__":
    root_directory = "./results"  # Change this to your results directory
    analyze_results(root_directory)