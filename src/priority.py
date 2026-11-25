"""Complaint priority: which problem should be solved first?

The score (0 to 100) is the sum of five kinds of evidence, each explained to the admin:

1. Category base     - some problems are urgent by nature (sewage overflow > park maintenance).
2. Risk words        - words in the text that signal danger, health risk, daily-life disruption,
                       vulnerable people, wide impact or explicit urgency. Each GROUP counts once,
                       so repeating "smell smell smell" does not inflate the score.
   A safety hazard together with vulnerable people earns an extra combined-risk bonus.
3. Duration          - how long the problem has lasted ("for two weeks", "5 din se").
4. Reports           - how many other complaints were linked to it as duplicates (added in the admin
                       view, because it changes as new complaints arrive).

Levels: Critical >= 75, High >= 55, Medium >= 35, Low < 35.

Why rules and not a trained model? The dataset has no "true priority" labels, so a model could not be
trained or evaluated honestly. Rules are transparent: every point in the score can be traced to a word
or fact, which an officer can check. Limitation: keyword rules do not understand negation
("no accident yet") or sarcasm. A trained classifier becomes possible once officers' priority
decisions are recorded and can be used as labels.
"""
import re
from dataclasses import dataclass, field

# 1. Category base score: how urgent this kind of problem is by nature.
CATEGORY_BASE = {
    "Sewage Overflow": 30, "Drainage Problems": 25, "Dead Animal Removal": 25, "Water Supply": 25,
    "Public Health and Sanitation": 25, "Stormwater Drainage": 22, "Water Leakage": 20,
    "Mosquito Control": 20, "Road Damage": 20, "Potholes": 20, "Stray Animal Problems": 20,
    "Traffic and Road Obstruction": 18, "Streetlight Problems": 15, "Public Toilets": 15,
    "Footpath Damage": 15, "Garbage Collection": 15, "Other Municipal Issues": 12,
    "Illegal Encroachment": 10, "Construction Waste": 10, "Public Park Maintenance": 8,
}
DEFAULT_BASE = 12

# 2. Risk word groups: (label shown to the admin, points, terms). English and romanised Hindi.
#    A term ending in * matches any ending (injur* -> injury, injured).
RISK_GROUPS = [
    ("Safety risk", 25, [
        "accident*", "injur*", "hurt", "fall into", "fell", "falling", "could fall", "can fall",
        "may fall", "collapse*", "electric shock", "live wire", "wires hanging", "hanging loose",
        "sparking", "fire", "set on fire", "bitten", "biting", "bite", "attack*", "chasing",
        "dangerous*", "danger", "unsafe", "sinkhole", "caved in", "open manhole", "uncovered manhole",
        "about to fall", "skid*", "gir sakta", "gir rahe", "girne wali", "khatarnak", "kaat rahe",
    ]),
    ("Health risk", 18, [
        "bad smell", "smell*", "stench", "stink*", "foul", "badbu*", "disease*", "dengue", "malaria",
        "fever", "sick", "sickness", "illness", "infection", "contaminated", "sewage", "sewer",
        "dirty water", "ganda paani", "mosquito*", "machhar", "larvae", "flies", "makkhi*", "rotting",
        "carcass", "unhygienic", "filth*", "health hazard", "health risk", "bimar*", "bukhar",
    ]),
    ("Daily life disrupted", 12, [
        "difficulty in walking", "difficulty walking", "difficult to walk", "cannot walk", "can't walk",
        "impossible to walk", "walk on the road", "no space", "blocked", "blocking", "traffic jam*",
        "ambulance*", "no water", "not getting", "no drop", "dark at night", "completely dark",
        "waterlogging", "flooded", "knee-deep", "under water", "entering homes", "into our houses",
        "cannot sleep", "can't sleep", "paani nahi", "chalne ki jagah nahi", "jam lag", "neend nahi",
        "nahana mushkil", "tanker",
    ]),
    ("Vulnerable people affected", 10, [
        "child*", "kids", "school*", "elderly", "senior citizen*", "old people", "women", "ladies",
        "pregnant", "patient*", "hospital", "bachche", "bachchon", "buzurg", "mahila*",
    ]),
    ("Wide impact or repeated", 8, [
        "whole area", "whole street", "entire", "everyone", "all houses", "every house", "many families",
        "hundreds", "residents", "again", "repeated*", "already complained", "complained twice",
        "nobody came", "no action", "nothing happens", "fed up", "poori gali", "ghar ghar", "koi nahi aaya",
    ]),
    ("Marked urgent", 6, ["urgent*", "immediately", "emergency", "asap", "turant", "jaldi"]),
]

# A safety hazard is more serious where vulnerable people are exposed to it
# (an open manhole outside a school), so the combination earns extra points.
COMBINED_RISK_POINTS = 15

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "a": 1, "ek": 1, "do": 2}
LEVELS = [(75, "Critical"), (55, "High"), (35, "Medium"), (0, "Low")]
LEVEL_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def _pattern(term: str) -> str:
    """Whole-word match; a trailing * allows any word ending."""
    if term.endswith("*"):
        return r"\b" + re.escape(term[:-1]) + r"\w*"
    return r"\b" + re.escape(term) + r"\b"


_GROUP_REGEX = [(label, pts, re.compile("|".join(_pattern(t) for t in terms), re.IGNORECASE))
                for label, pts, terms in RISK_GROUPS]
_DAYS = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|ek|do)\s*"
                   r"(day|days|din|week|weeks|hafte|hafta|fortnight|month|months)\b", re.IGNORECASE)


@dataclass
class Priority:
    score: int
    level: str
    reasons: list[str] = field(default_factory=list)

    @property
    def reason_text(self) -> str:
        return "; ".join(self.reasons)


def level_for(score: float) -> str:
    return next(name for limit, name in LEVELS if score >= limit)


def duration_days(text: str) -> int:
    """Longest duration mentioned, in days (0 if none). 'since yesterday' counts as 1."""
    best = 1 if re.search(r"\b(yesterday|kal se)\b", text, re.IGNORECASE) else 0
    if re.search(r"\bfortnight\b", text, re.IGNORECASE):
        best = max(best, 14)
    for num, unit in _DAYS.findall(text):
        n = int(num) if num.isdigit() else NUMBER_WORDS[num.lower()]
        unit = unit.lower()
        days = n * 30 if unit.startswith("month") else n * 14 if unit == "fortnight" else \
            n * 7 if unit.startswith(("week", "haft")) else n
        best = max(best, days)
    return best


def duration_points(days: int) -> int:
    return 10 if days >= 14 else 7 if days >= 7 else 4 if days >= 3 else 0


def score_complaint(text: str, category: str, extra_reports: int = 0) -> Priority:
    """Priority of one complaint. extra_reports = how many other complaints were linked to it."""
    base = CATEGORY_BASE.get(category, DEFAULT_BASE)
    score, reasons = base, [f"{category}: base {base}"]
    for label, pts, rx in _GROUP_REGEX:
        found = sorted({m.group(0).lower() for m in rx.finditer(text)})
        if found:
            score += pts
            reasons.append(f"{label} +{pts} ({', '.join(found[:4])})")
    labels = {r.split(" +")[0] for r in reasons}
    if {"Safety risk", "Vulnerable people affected"} <= labels:
        score += COMBINED_RISK_POINTS
        reasons.append(f"Safety hazard where children, elderly or patients are present +{COMBINED_RISK_POINTS}")
    days = duration_days(text)
    if duration_points(days):
        score += duration_points(days)
        reasons.append(f"Lasting about {days} days +{duration_points(days)}")
    if extra_reports > 0:
        bonus = min(15, 5 * extra_reports)
        score += bonus
        reasons.append(f"Reported by {extra_reports + 1} complaints +{bonus}")
    score = min(100, score)
    return Priority(score=int(score), level=level_for(score), reasons=reasons)


def with_reports(text_score: int, extra_reports: int) -> int:
    """Add the duplicate-report bonus to a stored text-based score."""
    return int(min(100, text_score + min(15, 5 * max(0, extra_reports))))
