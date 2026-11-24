"""Route-planning scorer for ReasonMap (short and long questions).

Metric definitions follow the paper / original ``main.py``:

* ``acc``: the answer starts at station_1, ends at station_2, every section's line exists
  in the station table and contains both its stops, and consecutive sections share their
  transfer stop.
* ``map score`` (short, max 10 + 10): compared against the reference route plans:
  +1 correct endpoints, then per section +2 line name, +1 departure stop, +1 arrival stop;
  capped at 10, and +10 on top when ``acc`` is 1.
* ``map score`` (long, max 20 + 20): as above plus a via-stop term, either
  "Number of Via Stops" (up to 4 per section, by relative error) or "Via Stops"
  ((IoU * 10 + hits) / 2); capped at 20, and +20 on top when ``acc`` is 1.

Fixes relative to the original implementation (see the README for details):

1. Reference arrival stops were built from the *departure* stop's suffix check (and fell
   back to the departure stop name) in three of the four code paths.
2. For non-Chinese cities the line-name normalisation was applied to the *model's* answer
   instead of the reference route, so the reference line name was never read; and its
   catch-all ``[A-Z0-9]{1,2}`` regex (case-insensitive) made every rule after it dead code.
   Reference names are now mapped onto the station-table keys.
3. "Via Stops" were split on "," without stripping, so every stop after the first
   (" 某站") could never match; they are now split on , ， 、 ; and stripped.
4. Each field is read from its own line, so trailing explanations after the last section
   no longer leak into "Arrival Stop" / "Via Stops", and a final line without a trailing
   newline is still parsed.
"""

import re

from .text import clean_string

SECTION_SEPARATOR = "--"

# Reference line-name aliases that can't be derived by pattern (city -> {api name: station key}).
_ROUTE_ALIASES = {
    "auckland": {"West": "Western Line", "Sth": "Southern Line", "East": "Eastern Line"},
    "dubai": {"MRed1": "Red Line", "MRed2": "Red Line", "MGrn": "Green Line"},
    "kl": {
        "MRT Putrajaya Line": "Line 12",
        "Ampang Line": "Line 3",
        "LRT Kelana Jaya Line": "Line 5",
        "KTM Tanjung Malim - Pelabuhan Klang": "Line 2",
        "LRT Sri Petaling Line": "Line 4",
        "MRT Kajang Line": "Line 9",
        "KTM Batu Caves - Pulau Sebang/Tampin": "Line 1",
    },
    "rome": {"B1": "Line B"},  # B1 is a branch of line B
}

# Patterns that extract a line identifier from a reference (map API) route name.
_ROUTE_PATTERNS = [
    re.compile(r"^Metro\s+([A-Z0-9]{1,2})[-\s]?Line$", re.IGNORECASE),  # LA: "Metro A Line", "Metro B-Line"
    re.compile(r"^(?:Metro|Linea)\s+([A-Z0-9]{1,2})$", re.IGNORECASE),  # Rome: "Metro B", "Linea C"
    re.compile(r"^M(\d+)\b.*metróvonal$", re.IGNORECASE),  # Budapest: "M1-es Metróvonal"
    re.compile(r"^([A-Z0-9]{1,2})\s+(?:Line|Train)$", re.IGNORECASE),  # NYC: "1 Line", "Q Train"
    re.compile(r"^([A-Z0-9]{1,2})$"),  # Oslo / Toronto / Geneva "1" / "18", Rome "A", Sydney "M1"
]


# ----------------------------------------------------------------------------- parsing


def _field(section, label, exclude_prefix=None):
    """Value of ``label:`` on its own line inside ``section`` (None when absent)."""
    lookbehind = f"(?<!{re.escape(exclude_prefix)})" if exclude_prefix else ""
    match = re.search(lookbehind + re.escape(label) + r":[ \t]*([^\n]*)", section)
    if not match:
        return None
    return match.group(1).strip()


def _split_via_stops(value):
    if not value:
        return []
    items = [v.strip() for v in re.split(r"[,，、;；]", value)]
    return [v for v in items if v and v.lower() not in ("none", "n/a", "无")]


def parse_route_sections(answer, via_mode=None):
    """Parse an answer into route sections.

    ``via_mode`` is None for short questions, "num" for "Number of Via Stops" and "list"
    for "Via Stops". Sections missing one of the mandatory labels are skipped.
    """
    sections = []
    for section in (answer or "").split(SECTION_SEPARATOR):
        if not section.strip():
            continue
        if "Route Name:" not in section or "Departure Stop:" not in section or "Arrival Stop:" not in section:
            continue
        route_name = _field(section, "Route Name")
        departure = _field(section, "Departure Stop")
        arrival = _field(section, "Arrival Stop")
        info = {
            "Route Name": route_name if route_name else "Wrong",
            "Departure Stop": clean_string(departure) if departure else "Wrong",
            "Arrival Stop": clean_string(arrival) if arrival else "Wrong",
        }
        if via_mode == "num":
            value = _field(section, "Number of Via Stops") or ""
            match = re.search(r"\d+", value)
            info["Number of Via Stops"] = int(match.group()) if match else 0
        elif via_mode == "list":
            value = _field(section, "Via Stops", exclude_prefix="Number of ")
            info["Via Stops"] = [clean_string(v) for v in _split_via_stops(value)]
        sections.append(info)
    return sections


def via_mode_for_question(question_long):
    return "num" if "Number of Via Stops:" in question_long else "list"


# ----------------------------------------------------------------------------- normalisation


def normalize_answer_route_name(name):
    """Beijing sub-line aliases accepted in answers (八通线 -> Line 1, 大兴线 -> Line 4)."""
    if "八通" in name:
        return "Line 1"
    if "大兴" in name:
        return "Line 4"
    return name


def normalize_reference_route_name(name, country, city, station_keys=()):
    """Map a reference (map API) route name onto the naming used by the station tables."""
    name = (name or "").strip()
    if country == "china":
        # 地铁5号线(后关--虎滩新区) -> Line 5 ; 地铁昌平线(昌平西山口--蓟门桥) -> 昌平线
        base = re.sub(r"\(.*?\)|（.*?）", "", name)
        match = re.search(r"(\d+)", base)
        if match:
            return f"Line {int(match.group(1))}"
        return base.replace("地铁", "").strip()

    keys = set(station_keys)
    if name in keys:
        return name
    alias = _ROUTE_ALIASES.get(city, {}).get(name)
    if alias:
        return alias
    for pattern in _ROUTE_PATTERNS:
        match = pattern.match(name)
        if match:
            ident = match.group(1).upper()
            for candidate in (f"Line {ident}", ident):
                if candidate in keys:
                    return candidate
            return f"Line {ident}"
    return name  # e.g. regional trains that are not on the metro map


def reference_stop_name(name, country):
    """Reference stops come without suffix; answers use "xx站" / "xx Station"."""
    name = (name or "").strip()
    suffix = "站" if country == "china" else "Station"
    if name.endswith(suffix):
        return name
    return name + suffix if country == "china" else f"{name} {suffix}"


# ----------------------------------------------------------------------------- metrics


def _endpoints_match(route_data, station_1, station_2):
    return clean_string(route_data[0]["Departure Stop"]) == clean_string(station_1) and clean_string(
        route_data[-1]["Arrival Stop"]
    ) == clean_string(station_2)


def route_is_correct(route_data, station_1, station_2, metro_data):
    """``acc``: 1 if the answered route is a valid route between the two stations."""
    if not route_data or not _endpoints_match(route_data, station_1, station_2):
        return 0
    for i, section in enumerate(route_data):
        route_name = normalize_answer_route_name(section["Route Name"])
        if route_name not in metro_data:
            return 0
        stations = [clean_string(s) for s in metro_data[route_name]]
        departure, arrival = section["Departure Stop"], section["Arrival Stop"]
        if clean_string(departure) not in stations or clean_string(arrival) not in stations:
            return 0
        if i < len(route_data) - 1 and clean_string(arrival) != clean_string(route_data[i + 1]["Departure Stop"]):
            return 0
    return 1


def _reference_plans(routes):
    if not routes:
        return []
    return [plan for plan in routes.values() if plan]


def _compare_section(section, ref, country, city, station_keys):
    """Line-name / stop points for one answered section against one reference section."""
    ref_name = normalize_reference_route_name(ref["route_name"], country, city, station_keys)
    name_ok = clean_string(ref_name) == clean_string(section["Route Name"])
    dep_ok = clean_string(reference_stop_name(ref["departure_stop"], country)) == clean_string(section["Departure Stop"])
    arr_ok = clean_string(reference_stop_name(ref["arrival_stop"], country)) == clean_string(section["Arrival Stop"])
    return ref_name, name_ok, dep_ok, arr_ok


def score_short(answer, meta, metro_data):
    country, city = meta["country"], meta["city"]
    station_keys = list(metro_data)
    result = {"acc": 0, "map_score": 0, "departure_arrival": 0, "route_name": 0, "stops": 0}

    route_data = parse_route_sections(answer)
    if not route_data:
        return result

    acc = route_is_correct(route_data, meta["station_1"], meta["station_2"], metro_data)
    result["acc"] = acc
    endpoints_ok = _endpoints_match(route_data, meta["station_1"], meta["station_2"])

    best = None
    for plan in _reference_plans(meta.get("routes")):
        score = dep_arr = name_pts = stop_pts = 0
        if endpoints_ok:
            score += 1
            dep_arr += 1
            for section, ref in zip(route_data, plan):
                _, name_ok, dep_ok, arr_ok = _compare_section(section, ref, country, city, station_keys)
                name_pts += 2 * name_ok
                stop_pts += dep_ok + arr_ok
            score += name_pts + stop_pts
        score = min(10, score)
        if acc:
            score += 10  # a correct route always outranks an incorrect one
        if best is None or score >= best:
            best = score
            result.update(map_score=score, departure_arrival=dep_arr, route_name=name_pts, stops=stop_pts)
    return result


def score_long(answer, meta, metro_data):
    country, city = meta["country"], meta["city"]
    station_keys = list(metro_data)
    via_mode = via_mode_for_question(meta["question_long"])
    result = {
        "acc": 0,
        "map_score": 0,
        "departure_arrival": 0,
        "route_name": 0,
        "stops": 0,
        "num_via_stop_score": 0,
        "via_stops_score": 0,
    }

    route_data = parse_route_sections(answer, via_mode=via_mode)
    if not route_data:
        return result

    acc = route_is_correct(route_data, meta["station_1"], meta["station_2"], metro_data)
    result["acc"] = acc
    endpoints_ok = _endpoints_match(route_data, meta["station_1"], meta["station_2"])

    best = None
    for plan in _reference_plans(meta.get("routes")):
        score = dep_arr = name_pts = stop_pts = 0
        num_via_score = 0
        via_score = 0
        if endpoints_ok:
            score += 1
            dep_arr += 1
            via_hits = 0
            union, inter = set(), set()
            for section, ref in zip(route_data, plan):
                ref_name, name_ok, dep_ok, arr_ok = _compare_section(section, ref, country, city, station_keys)
                name_pts += 2 * name_ok
                stop_pts += dep_ok + arr_ok

                if via_mode == "num":
                    pred_n = int(section["Number of Via Stops"])
                    ref_n = int(ref["num_via_stops"])
                    error = abs(pred_n - ref_n)
                    if error == 0:
                        num_via_score += 4
                    elif ref_n != 0:
                        num_via_score += max(0, 4 - (error / ref_n) * 4)
                else:
                    ref_via = {clean_string(reference_stop_name(v, country)) for v in (ref.get("via_stops") or [])}
                    pred_via = section["Via Stops"]
                    via_hits += sum(1 for v in pred_via if v in ref_via)
                    if name_ok:
                        union.update(ref_via, pred_via)
                        inter.update(ref_via.intersection(pred_via))

            via_hits = min(10, via_hits)
            num_via_score = min(10, num_via_score)
            iou = len(inter) / len(union) if union else 0
            via_score = (iou * 10 + via_hits) / 2
            score += name_pts + stop_pts + via_score + num_via_score
        score = min(20, score)
        if acc:
            score += 20
        if best is None or score >= best:
            best = score
            result.update(
                map_score=score,
                departure_arrival=dep_arr,
                route_name=name_pts,
                stops=stop_pts,
                num_via_stop_score=num_via_score,
                via_stops_score=via_score,
            )
    return result


def score_reasonmap_record(record, metro_data):
    """Score one prediction record (see ``infer.py``) and return the per-sample result JSON.

    Field names match the files written by the original ``main.py`` so ``cal_metrics.py``
    and any existing analysis scripts keep working.
    """
    meta = record["meta"]
    short = record["generations"]["short"]
    long_ = record["generations"]["long"]
    s = score_short(short["final_answer"], meta, metro_data)
    l = score_long(long_["final_answer"], meta, metro_data)
    return {
        "model": record["model"],
        "country": meta["country"],
        "city": meta["city"],
        "question_short": meta["question_short"],
        "question_long": meta["question_long"],
        "short full answer": short["response"],
        "short token count": short["token_count"],
        "short final answer": short["final_answer"],
        "acc short": s["acc"],
        "best map api score short": s["map_score"],
        "save departure arrival score short": s["departure_arrival"],
        "save route name score short": s["route_name"],
        "save stops score short": s["stops"],
        "long full answer": long_["response"],
        "long token count": long_["token_count"],
        "long final answer": long_["final_answer"],
        "acc long": l["acc"],
        "best map api score long": l["map_score"],
        "save departure arrival score long": l["departure_arrival"],
        "save route name score long": l["route_name"],
        "save stops score long": l["stops"],
        "save num via stop score": l["num_via_stop_score"],
        "save via stops score": l["via_stops_score"],
        "difficulty city": meta.get("difficulty_city"),
        "difficulty question": meta.get("difficulty_question"),
        "city line count": meta.get("city_line_count"),
        "city transfer count": meta.get("city_transfer_count"),
        "question transfer count": meta.get("question_transfer_count"),
    }
