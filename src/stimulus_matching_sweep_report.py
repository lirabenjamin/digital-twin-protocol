"""Readable scenario tables for the fixed-distance penalty comparison."""
from pathlib import Path
from string import Template

from stimulus_matching_report import md_table

# Short display labels only; the original full text and options appear below.
SCENARIO_LABELS = {
    "s007": "Resort spa weekend", "s050": "New bathroom vanity and faucet",
    "s051": "Private cardiology clinic", "s076": "Premium tool kit",
    "s083": "Uber to a career fair", "s085": "New video game at release",
    "s094": "Smart-bulb kit",
    "s005": "Team dinner at a restaurant", "s014": "Premium amusement-park birthday package",
    "s034": "Lake cabin weekend with family", "s099": "Mexican restaurant with coworkers",
    "s004": "New dishwasher", "s013": "Prepared deli dinner", "s015": "Cleaning service before in-laws visit",
    "s017": "Nearby prescription pickup", "s019": "New kitchen faucet", "s022": "Temple retreat",
    "s023": "Drive and pay for parking", "s030": "Dealership oil change", "s035": "Water park with grandkids",
    "s036": "Gym membership with trainer", "s045": "Hockey game with coworkers", "s047": "Visit sister in Denver",
    "s048": "Replace failing air conditioner", "s052": "Cruise excursion booked ahead",
    "s054": "Theater show with sister", "s058": "Coffee-shop cold brew", "s061": "Name-brand standing desk",
    "s065": "Church-group dinner", "s066": "Premium painting kit", "s073": "Trampoline-park birthday party",
    "s074": "Ice cream after grandkids' play", "s081": "Premium baseball tickets with dad",
    "s082": "Party with roommates", "s087": "Travel to friend's tournament",
    "s096": "Paid design software", "s097": "Napa anniversary weekend", "s098": "New brand-name car seat",
}


def scenario_table(selected):
    rows = []
    for (_, low_id, high_id), group in selected.groupby(["focal", "low_id", "high_id"], sort=False):
        r = group.iloc[0]
        lambdas = ", ".join(f"{p:g}" for p in sorted(group.penalty))
        low = f"{SCENARIO_LABELS[low_id]} ({low_id}; {r.low_score:.2f})"
        high = f"{SCENARIO_LABELS[high_id]} ({high_id}; {r.high_score:.2f})"
        rows.append([r.moderator, lambdas, low, high, f"{r.focal_gap:.2f}", f"{r.other_adjusted:.2f}",
                     f"{r.other_mean_abs:.2f}", f"{r.other_max:.2f} — {r.worst_feature_name}"])
    return md_table(["Feature of interest", "Balance weight (λ)", "Low scenario (ID; rating)",
                     "High scenario (ID; rating)", "Gap on feature of interest", "Adjusted distance",
                     "Average gap on other features", "Largest gap on another feature"], rows)


def full_scenarios(selected, stimuli):
    lookup = {s["stimulus"]: s for s in stimuli}
    sections = []
    for sid in sorted(set(selected.low_id) | set(selected.high_id)):
        scenario = lookup[sid]
        options = " / ".join(f"{o['label']} (${o['cost']:g})" for o in scenario["options"])
        sections.append(f"### {sid}: {SCENARIO_LABELS[sid]}\n\n{scenario['scenario']}\n\nOptions: {options}\n")
    return "\n".join(sections)


def summary_table(summary):
    seven = summary.loc[summary.scope.eq("seven nominated")]
    return md_table(["Balance weight (λ)", "Average gap on feature of interest", "Average adjusted distance",
                     "Average gap on other features", "Largest gap on any other feature",
                     "Distinct scenarios in the seven pairs"],
                    [[f"{r.penalty:g}", f"{r.mean_focal_gap:.2f}", f"{r.mean_adjusted_distance:.2f}",
                      f"{r.mean_absolute_nonfocal_difference:.2f}", f"{r.largest_nonfocal_difference:.2f}",
                      str(r.unique_stimuli)] for r in seven.itertuples()])


def current_selection_table(selected, penalty=2):
    rows = []
    for r in selected.loc[selected.penalty.eq(penalty)].itertuples():
        feature = r.moderator + (" (comparison)" if r.role == "proposed_comparison" else "")
        low = f"{SCENARIO_LABELS[r.low_id]} ({r.low_id}; {r.low_score:.2f})"
        high = f"{SCENARIO_LABELS[r.high_id]} ({r.high_id}; {r.high_score:.2f})"
        rows.append([feature, low, high, f"{r.focal_gap:.2f}", f"{r.other_median_abs:.2f}",
                     f"{r.other_max:.2f} — {r.worst_feature_name}"])
    return md_table(["Feature", "Low scenario (ID; rating)", "High scenario (ID; rating)",
                     "Focal gap", "Median gap on other features", "Largest gap on another feature"], rows)


def write_report(root, selected, summary, stimuli):
    targets = selected.loc[selected.role.eq("user_nominated")]
    comparisons = selected.loc[selected.role.eq("proposed_comparison")]
    # Keep moderators together and preserve their requested order.
    order = {f: i for i, f in enumerate(selected.focal.drop_duplicates())}
    targets = targets.assign(_order=targets.focal.map(order)).sort_values(["_order", "penalty"])
    comparisons = comparisons.assign(_order=comparisons.focal.map(order)).sort_values(["_order", "penalty"])
    template = Template((Path(__file__).parent / "templates/stimulus_selection_report.md").read_text())
    report = template.substitute(
        summary_table=summary_table(summary), targets_table=scenario_table(targets),
        comparisons_table=scenario_table(comparisons), selected_count=len(selected),
        difference_count=len(selected) * 20, full_scenarios=full_scenarios(selected, stimuli),
        lambda_two_table=current_selection_table(selected),
        lambda_one_table=current_selection_table(selected, penalty=1),
    )
    path = root / "reports/stimulus_matching/lambda_sweep.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.rstrip() + "\n")
    return path
