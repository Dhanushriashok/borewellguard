"""
Insert clearly labelled DEMO reports so the dashboard is not empty during a demo.

Run from the BorewellRiskAI folder:
    py -m backend.seed_demo

Every location starts with "(demo)". Run it only once; it skips if the database
already has reports. Delete backend/borewellguard.db to start fresh.
"""

from contextlib import closing

from backend.app import ReportIn, connect, create_report

DEMO = [
    ("(demo) Village A - near bus stop path", "Uncovered", "Near public pathway",
     dict(nearby_population=30, children_nearby=True, abandoned=True, fenced_or_marked=False)),
    ("(demo) Village A - behind primary school", "Uncovered", "Near homes or school",
     dict(nearby_population=60, children_nearby=True, abandoned=True, fenced_or_marked=False,
          borewell_age_years=14, depth_ft=520)),
    ("(demo) Village B - farm road", "Damaged cover", "Near farmland",
     dict(nearby_population=6, children_nearby=False, abandoned=False, fenced_or_marked=False)),
    ("(demo) Village B - temple street", "Damaged cover", "Near homes or school",
     dict(nearby_population=40, children_nearby=True, abandoned=False, fenced_or_marked=False)),
    ("(demo) Village C - fenced plot", "Covered", "Restricted access",
     dict(nearby_population=3, children_nearby=False, abandoned=False, fenced_or_marked=True)),
    ("(demo) Village C - community well area", "Covered", "Near public pathway",
     dict(nearby_population=25, children_nearby=False, abandoned=False, fenced_or_marked=True)),
    ("(demo) Village D - unknown cover, lane side", "Unknown", "Near public pathway",
     dict(nearby_population=18, children_nearby=True)),
    ("(demo) Village D - abandoned field", "Uncovered", "Near farmland",
     dict(nearby_population=4, children_nearby=False, abandoned=True, fenced_or_marked=False)),
]


def main() -> None:
    with closing(connect()) as conn:
        existing = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    if existing:
        print(f"Database already has {existing} reports. Nothing added.")
        return
    for location, cover, access, extra in DEMO:
        report = create_report(
            ReportIn(location=location, cover_condition=cover, nearby_access=access, **extra)
        )
        print(f"{report['id']}  {report['risk_level']:8s} {location}")
    print(f"Added {len(DEMO)} demo reports.")


if __name__ == "__main__":
    main()