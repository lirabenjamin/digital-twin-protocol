"""Make each option description read well in either position (2026-11-03): screen 1 now describes the options in
the same order as the answer scale, which is random, so no description may say "The other option", "Or you can",
"The second", "too", or "about the same". Also changes the closing line from "choose" to "prefer", to match the
question. Exact string replacements on layout.a / layout.b; fails loudly if a target is gone. Keeps "scenario" (the
full paragraph) equal to the layout joined in A-B order."""
import json

from common import HERE

CLOSE_OLD = "Please think about which of the two options you would choose."
CLOSE_NEW = "Please think about which of the two options you would prefer."
# stimulus -> [(field, old start or fragment, new)]
EDITS = {
    "s007": [("a", "One option is relative's backyard with catering: your aunt", "Your aunt"),
             ("b", "The other option is restaurant private room: an upscale", "An upscale")],
    "s014": [("b", "Or you can use", "You can use")],
    "s016": [("a", "The first is a Bank savings account at a local bank, which pays",
              "A savings account at a local bank pays"),
             ("b", "The second is Stock fund shares, a few shares of a broad stock fund bought in her name, which have",
              "A few shares of a broad stock fund, bought in her name, have")],
    "s025": [("b", "costs about $15 too, and it", "also costs about $15. It")],
    "s030": [("a", "One option is the cat hotel, a boarding place where", "The cat hotel is a boarding place where"),
             ("b", "The other option is the in-home sitter, who", "The in-home sitter")],
    "s031": [("a", "One option is partial shingle repair, which replaces", "Partial shingle repair replaces"),
             ("b", "The other option is full roof replacement, which puts", "Full roof replacement puts")],
    "s033": [("b", "Or you can go to", "You can go to")],
    "s037": [("a", "The first is wired LED shop-light strips, which give", "Wired LED shop-light strips give"),
             ("b", "The second is plug-in clamp lamps, which you", "Plug-in clamp lamps are lamps you")],
    "s038": [("b", "and costs about the same.", "and is priced a little under your limit.")],
    "s040": [("b", "Or you can rent", "You can rent")],
    "s049": [("a", "One choice is the library story hour, a quiet", "The library story hour is a quiet"),
             ("b", "The other choice is the nature center mud kitchen, a messy", "The nature center mud kitchen is a messy")],
    "s055": [("b", "Or you can use", "You can use")],
    "s056": [("b", "Or you can plant", "You can plant")],
    "s057": [("b", "You can also buy", "You can buy")],
    "s059": [("b", "Or you can buy", "You can buy")],
    "s060": [("b", "Or you can buy", "You can buy")],
    "s062": [("a", "Your first option is the home carrier roaming pass, which costs", "The home carrier roaming pass costs"),
             ("b", "Your second option is the local prepaid SIM, which costs", "The local prepaid SIM costs")],
    "s063": [("a", "One choice is the holiday sports camp, a nearby camp that runs", "The holiday sports camp is a nearby camp that runs"),
             ("b", "The other choice is having your retired father look after her, which means she",
              "If your retired father looks after her, she")],
    "s067": [("a", "One choice is a Dormitory loft bunk, which has", "The dormitory loft bunk has"),
             ("b", "The other choice is a Side room floor spot, which means lying", "The side room floor spot means lying")],
    "s069": [("a", "One choice is the free big-tech tier, which costs", "The free big-tech tier costs"),
             ("b", "The other choice is the paid genealogy platform plan, which costs", "The paid genealogy platform plan costs")],
    "s070": [("a", "The first is a trailing pothos in a hanging basket, which costs", "The trailing pothos comes in a hanging basket, costs"),
             ("b", "The second is a small bonsai juniper in a shallow tray, which costs", "The small bonsai juniper comes in a shallow tray, costs")],
    "s073": [("a", "One option is the pharmacy gel, a prescription-strength gel that costs", "The pharmacy gel is a prescription-strength gel that costs"),
             ("b", "The other option is weekly massage sessions, which cost", "Weekly massage sessions cost")],
    "s075": [("b", "Or you can get", "You can get")],
    "s077": [("b", "Or you can choose", "You can choose")],
    "s079": [("a", "One option is the local bank loan officer, a dedicated person", "The local bank loan officer is a dedicated person"),
             ("b", "The other option is the online lender app, which handles", "The online lender app handles")],
    "s080": [("b", "Or you can use", "You can use")],
    "s083": [("a", "One is a back seat with a power outlet, which is", "The back seat with a power outlet is"),
             ("b", "The other is a front seat with legroom, which gives", "The front seat with legroom gives")],
    "s084": [("b", "Or you can give", "You can give")],
    "s086": [("a", "One choice is the large hospital-affiliated clinic, which has", "The large hospital-affiliated clinic has"),
             ("b", "The other choice is the small independent pediatrician, which is", "The small independent pediatrician is")],
    "s087": [("b", "Or you can order", "You can order")],
    "s088": [("a", "One option is a single small dog", "You could adopt a single small dog"),
             ("b", "The other option is a pair of cats adopted from a shelter.", "You could adopt a pair of cats from a shelter.")],
    "s089": [("a", "One option is the library room, which is", "The library room is"),
             ("b", "The other option is the rented studio above the music shop, which costs", "The rented studio above the music shop costs")],
    "s091": [("a", "One option is a one-time technician spray: a pest", "With a one-time technician spray, a pest"),
             ("b", "The other option is bait stations: you set", "With bait stations, you set")],
    "s094": [("b", "Or you can use", "You can use")],
    "s095": [("b", "Or you can download", "You can download")],
    "s099": [("a", "The first is to buy bins and sort it yourself: a set", "You could buy bins and sort it yourself: a set"),
             ("b", "The second is to hire a professional organizer: it costs", "You could hire a professional organizer: it costs")],
}
stimuli = json.loads((HERE / "stimuli.json").read_text())
for s in stimuli:
    lay = s["layout"]
    for field, old, new in EDITS.get(s["stimulus"], []):
        if old in lay[field]:
            lay[field] = lay[field].replace(old, new)
        elif new not in lay[field]:
            raise SystemExit(f"{s['stimulus']}.{field}: target text not found")
    lay["after"] = lay["after"].replace(CLOSE_OLD, CLOSE_NEW)
    assert lay["after"].endswith(CLOSE_NEW), s["stimulus"]
    s["scenario"] = " ".join(v for v in lay.values() if v)
(HERE / "stimuli.json").write_text(json.dumps(stimuli, indent=1))
print(f"edited {len(EDITS)} stimuli; closing line now 'prefer' in all {len(stimuli)}")
