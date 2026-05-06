"""
Step 16 — Aggregate and Rank
- Sort by viral score
- Group by niche
- Find top formats and hook styles
"""

from collections import Counter, defaultdict
import logging

log = logging.getLogger("aggregator")


def aggregate_and_rank(processed: list[dict]) -> dict:
    """
    Returns a ranked summary dict for pipeline output/reporting.
    """
    valid = [s for s in processed if not s.get("error")]

    if not valid:
        return {
            "top_shorts": [],
            "niche_groups": {},
            "top_formats": [],
            "top_hook_styles": [],
            "top_emotions": [],
        }

    # Sort by growth velocity first, then viral score.
    ranked = sorted(
        valid,
        key=lambda x: (x.get("views_per_hour", 0), x.get("viral_score", 0)),
        reverse=True,
    )

    # Group by niche
    niche_groups: dict[str, list] = defaultdict(list)
    for s in ranked:
        niche = s.get("niche", "other")
        niche_groups[niche].append(s)

    # Top niches by total views
    niche_summary = {}
    for niche, items in niche_groups.items():
        niche_summary[niche] = {
            "count": len(items),
            "total_views": sum(i.get("views", 0) for i in items),
            "avg_viral_score": round(
                sum(i.get("viral_score", 0) for i in items) / len(items), 2
            ),
            "top_short": items[0],  # already sorted
        }

    # Sort niches by total views
    sorted_niches = dict(
        sorted(niche_summary.items(), key=lambda x: x[1]["total_views"], reverse=True)
    )

    # Top formats
    format_counts = Counter(s.get("format", "unknown") for s in valid)
    top_formats = [{"format": f, "count": c} for f, c in format_counts.most_common(5)]

    # Top hook styles
    hook_counts = Counter(s.get("hook_style", "unknown") for s in valid)
    top_hook_styles = [{"style": h, "count": c} for h, c in hook_counts.most_common(5)]

    # Top emotions
    emotion_counts = Counter(s.get("primary_emotion", "unknown") for s in valid)
    top_emotions = [{"emotion": e, "count": c} for e, c in emotion_counts.most_common(5)]

    log.info(
        "Ranked %d shorts | Top niche: %s | Top format: %s",
        len(ranked),
        list(sorted_niches.keys())[0] if sorted_niches else "n/a",
        top_formats[0]["format"] if top_formats else "n/a",
    )

    return {
        "top_shorts": ranked[:10],  # top 10 for the report
        "niche_groups": sorted_niches,
        "top_formats": top_formats,
        "top_hook_styles": top_hook_styles,
        "top_emotions": top_emotions,
    }