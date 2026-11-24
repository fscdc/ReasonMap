"""Scorer regression tests (stdlib only): ``python -m pytest tests`` or ``python tests/test_scoring.py``."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reasonmap.data import load_metro_data  # noqa: E402
from reasonmap.models import FAMILIES, detect_family, extract_final_answer  # noqa: E402
from reasonmap.scoring import (  # noqa: E402
    normalize_reference_route_name,
    parse_route_sections,
    score_long,
    score_short,
)
from reasonmap.scoring_plus import extract_plus_answer, is_correct  # noqa: E402

HANGZHOU_META = {
    "country": "china",
    "city": "hangzhou",
    "station_1": "滨康路站",
    "station_2": "之江文化中心站",
    "question_long": "... Via Stops: 某站, 某站, 某站 ...",
    "routes": {"1": [
        {"route_name": "地铁5号线(姑娘桥--南湖东)", "departure_stop": "滨康路", "arrival_stop": "长河",
         "num_via_stops": 2, "via_stops": ["江晖路", "聚才路"]},
        {"route_name": "地铁6号线(枸桔弄--桂花西路)", "departure_stop": "长河", "arrival_stop": "之江文化中心",
         "num_via_stops": 5, "via_stops": ["建业路", "诚业路", "伟业路", "中医药大学", "西浦路"]},
    ]},
}

GOLD_LONG = """Route Name: Line 5
Departure Stop: 滨康路站
Arrival Stop: 长河站
Via Stops: 江晖路站, 聚才路站
--
Route Name: Line 6
Departure Stop: 长河站
Arrival Stop: 之江文化中心站
Via Stops: 建业路站, 诚业路站, 伟业路站, 中医药大学站, 西浦路站"""


def _metro(country, city):
    return load_metro_data(country, city, ROOT / "stations")


def test_gold_answer_gets_full_marks():
    metro = _metro("china", "hangzhou")
    short = "\n".join(l for l in GOLD_LONG.splitlines() if not l.startswith("Via Stops"))
    s = score_short(short, HANGZHOU_META, metro)
    # 1 (endpoints) + 2 * 2 (line names) + 4 (stops) = 9 -> +10 for acc
    assert s["acc"] == 1 and s["map_score"] == 19 and s["route_name"] == 4 and s["stops"] == 4
    l = score_long(GOLD_LONG, HANGZHOU_META, metro)
    # 1 + 2*2 + 4 + (IoU 1.0 * 10 + 7 hits) / 2 = 17.5 -> +20 for acc
    assert l["acc"] == 1 and l["via_stops_score"] == 8.5 and l["map_score"] == 37.5


def test_via_stops_split_on_comma_space():
    secs = parse_route_sections(GOLD_LONG, via_mode="list")
    assert secs[1]["Via Stops"] == ["建业路站", "诚业路站", "伟业路站", "中医药大学站", "西浦路站"]
    assert parse_route_sections("Route Name: L\nDeparture Stop: a\nArrival Stop: b\nVia Stops: None", "list")[0]["Via Stops"] == []


def test_trailing_text_does_not_leak_into_arrival():
    metro = _metro("china", "hangzhou")
    answer = "Route Name: Line 5\nDeparture Stop: 滨康路站\nArrival Stop: 长河站\n--\nRoute Name: Line 6\n" \
             "Departure Stop: 长河站\nArrival Stop: 之江文化中心站\n\nThis route needs one transfer."
    assert score_short(answer, HANGZHOU_META, metro)["acc"] == 1


def test_reference_arrival_uses_its_own_suffix():
    meta = dict(HANGZHOU_META, routes={"1": [dict(HANGZHOU_META["routes"]["1"][0], departure_stop="滨康路站")]})
    answer = "Route Name: Line 5\nDeparture Stop: 滨康路站\nArrival Stop: 长河站"
    # departure already ends with 站: the reference arrival must still be 长河站, not 滨康路站
    assert score_short(answer, dict(meta, station_2="长河站"), _metro("china", "hangzhou"))["stops"] == 2


def test_reference_route_names_map_to_station_keys():
    cases = {
        ("america", "los_angeles", "Metro B-Line"): "Line B",
        ("america", "miami", "Metrorail Green Line"): "Metrorail Green Line",
        ("canada", "toronto", "1"): "Line 1",
        ("italy", "rome", "B1"): "Line B",
        ("italy", "rome", "Linea C"): "Line C",
        ("italy", "rome", "A"): "Line A",
        ("new_zealand", "auckland", "Sth"): "Southern Line",
        ("switzerland", "geneva", "18"): "Line 18",
        ("america", "new_york", "Q Train"): "Q",
        ("america", "new_york", "1 Line"): "1",
        ("hungary", "budapest", "M4-es metróvonal"): "Line 4",
        ("australia", "sydney", "M1"): "Line M1",
        ("united_arab_emirates", "dubai", "MRed2"): "Red Line",
        ("malaysia", "kl", "MRT Kajang Line"): "Line 9",
        ("china", "beijing", "地铁1号线八通线(环球度假区--古城)"): "Line 1",
        ("china", "beijing", "地铁房山线(阎村东--东管头南)"): "房山线",
        ("hungary", "budapest", "S50"): "S50",  # regional train, not on the metro map
    }
    for (country, city, name), want in cases.items():
        keys = list(json.load(open(ROOT / "stations" / country / f"{city}.json", encoding="utf-8")))
        assert normalize_reference_route_name(name, country, city, keys) == want, (city, name)


def test_non_china_route_name_points():
    meta = {
        "country": "new_zealand", "city": "auckland",
        "station_1": "Newmarket Station", "station_2": "Penrose Station",
        "question_long": "Number of Via Stops: 6",
        "routes": {"1": [{"route_name": "Sth", "departure_stop": "Newmarket Station", "arrival_stop": "Penrose",
                          "num_via_stops": 3, "via_stops": []}]},
    }
    answer = "Route Name: Southern Line\nDeparture Stop: Newmarket Station\nArrival Stop: Penrose Station\nNumber of Via Stops: 3"
    r = score_long(answer, meta, _metro("new_zealand", "auckland"))
    assert r["route_name"] == 2 and r["stops"] == 2 and r["num_via_stop_score"] == 4


def test_family_answer_extraction():
    assert detect_family("Qwen/QVQ-72B-Preview") == "qvq"
    assert detect_family("Skywork/Skywork-R1V2-38B") == "skywork-r1v"
    assert detect_family("ChenShawn/DeepEyes-7B") == "qwen2.5-vl"
    assert detect_family("some/new-model") == "generic"
    qvq = FAMILIES["qvq"]
    boxed = "thinking...\\boxed{\\text{Route Name: Line 1} \\text{Departure Stop: A站}}"
    assert extract_final_answer(boxed, qvq) == "Route Name: Line 1\nDeparture Stop: A站"
    assert extract_final_answer("◁think▷x◁/think▷ Route Name: L", FAMILIES["kimi-vl"]) == "Route Name: L"
    assert extract_final_answer("<think>x</think>ans", FAMILIES["skywork-r1v"]) == "ans"


def test_plus_answers():
    assert extract_plus_answer("so \\boxed{B}") == "b" and is_correct("b", 1, "Counting1")
    assert extract_plus_answer("The answer is 4.") == "4" and is_correct("4", 4, "Counting3")
    assert is_correct("yes", 1, "TorF2") and not is_correct("yes", 0, "TorF1")
    assert not is_correct("3.5", 3, "Counting2")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
