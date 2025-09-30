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

def analyze_results_v2_weighted(root_dir):
    difficulty_weights = {
        "easy": 1.0,
        "middle": 1.5,
        "hard": 2.0
    }

    type_weights = {
        "Counting1": 1.0,
        "Counting2": 1.0,
        "Counting3": 1.0,
        "TorF1": 1.0,
        "TorF2": 1.0,
    }

    counting_types = {"Counting1", "Counting2", "Counting3"}
    torf_types = {"TorF1", "TorF2", "TorF"}  # 兼容可能的默认值 "TorF"

    def safe_div(n, d):
        return (n / d) if d else 0.0

    def add_group(g, acc, w):
        g["count"] += 1
        g["sum"]   += acc
        g["w_sum"] += acc * w
        g["w_total"] += w

    model_stats = defaultdict(lambda: {
        "count": 0,
        "acc_short_sum": 0.0,
        "acc_short_weighted_sum": 0.0,
        "weight_total": 0.0,
        "token_count_sum": 0.0,
        "groups": {
            "counting": {"count": 0, "sum": 0.0, "w_sum": 0.0, "w_total": 0.0},
            "torf":     {"count": 0, "sum": 0.0, "w_sum": 0.0, "w_total": 0.0},
            "torf1":    {"count": 0, "sum": 0.0, "w_sum": 0.0, "w_total": 0.0},
            "torf2":    {"count": 0, "sum": 0.0, "w_sum": 0.0, "w_total": 0.0},
        }
    })

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                try:
                    filepath = os.path.join(dirpath, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    model = data.get("model", "unknown_model")
                    acc_short = float(data.get("acc short", 0.0))
                    token_count = float(data.get("token count", 0))
                    difficulty = str(data.get("difficulty city", "easy")).lower()
                    q_type = str(data.get("type", "TorF"))

                    # 权重
                    difficulty_weight = difficulty_weights.get(difficulty, 1.0)
                    type_weight = type_weights.get(q_type, 1.0)
                    final_weight = difficulty_weight * type_weight

                    stats = model_stats[model]
                    # 总体
                    stats["count"] += 1
                    stats["acc_short_sum"] += acc_short
                    stats["acc_short_weighted_sum"] += acc_short * final_weight
                    stats["weight_total"] += final_weight
                    stats["token_count_sum"] += token_count

                    # 分组
                    ql = q_type.lower()
                    if (q_type in counting_types) or ql.startswith("count"):
                        add_group(stats["groups"]["counting"], acc_short, final_weight)
                    elif ql == "torf1":
                        add_group(stats["groups"]["torf1"], acc_short, final_weight)
                        add_group(stats["groups"]["torf"],  acc_short, final_weight)
                    elif ql == "torf2":
                        add_group(stats["groups"]["torf2"], acc_short, final_weight)
                        add_group(stats["groups"]["torf"],  acc_short, final_weight)
                    elif (q_type in torf_types) or ql.startswith("torf"):
                        # 未区分 1/2 的“笼统 TorF”，只计入总 TorF
                        add_group(stats["groups"]["torf"], acc_short, final_weight)
                    # 其他类型忽略

                except Exception as e:
                    print(f"❌ Error processing {filename} in {dirpath}: {e}")

    if not model_stats:
        print("⚠️ 没有找到任何符合格式的JSON文件")
        return

    for model, stats in model_stats.items():
        count = stats["count"]
        weight_total = stats["weight_total"]

        print(f"📊 Model: {model}")
        print(f"  Total files: {count}")
        print(f"  Average acc_short: {safe_div(stats['acc_short_sum'], count):.4f}")
        print(f"  Weighted acc_short: {safe_div(stats['acc_short_weighted_sum'], weight_total):.4f}")
        print(f"  Average token count: {safe_div(stats['token_count_sum'], count):.2f}")

        gC  = stats["groups"]["counting"]
        gT  = stats["groups"]["torf"]
        gT1 = stats["groups"]["torf1"]
        gT2 = stats["groups"]["torf2"]

        print("  [Counting]")
        print(f"    files: {gC['count']}")
        print(f"    avg acc_short: {safe_div(gC['sum'], gC['count']):.4f}")
        print(f"    weighted acc_short: {safe_div(gC['w_sum'], gC['w_total']):.4f}")

        print("  [TorF]")
        print(f"    files: {gT['count']}")
        print(f"    avg acc_short: {safe_div(gT['sum'], gT['count']):.4f}")
        print(f"    weighted acc_short: {safe_div(gT['w_sum'], gT['w_total']):.4f}")

        print("  [TorF1]")
        print(f"    files: {gT1['count']}")
        print(f"    avg acc_short: {safe_div(gT1['sum'], gT1['count']):.4f}")
        print(f"    weighted acc_short: {safe_div(gT1['w_sum'], gT1['w_total']):.4f}")

        print("  [TorF2]")
        print(f"    files: {gT2['count']}")
        print(f"    avg acc_short: {safe_div(gT2['sum'], gT2['count']):.4f}")
        print(f"    weighted acc_short: {safe_div(gT2['w_sum'], gT2['w_total']):.4f}")

        print()



if __name__ == "__main__":
    root_directory = "./results"  # Change this to your results directory
    analyze_results(root_directory)
    analyze_results_v2_weighted("./results_plus")