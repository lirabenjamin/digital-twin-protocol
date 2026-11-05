"""Write the picked stimuli as twins see them. screens.md: both full prompts per stimulus, built with
run_twins.prompt (word for word what the twin gets). screens_team.md: same screens, scenario shown once and the
two frame lines side by side, for the team to read."""
import json

from run_twins import HERE, TASK, prompt

stimuli = json.loads((HERE / "stimuli.json").read_text())
full = ["# Choice vs. rejection frame: the 100 stimuli as twins see them\n",
        "Each twin sees all 100 stimuli in both frames, and every stimulus x frame is a separate call with no memory of the others. The two frames differ only in the "
        "line on screen 2. The order of the answer options is random per twin x stimulus and the same in both "
        "conversations, and screen 1 describes the options in that same order; here the option listed first is the scale's left end (1).\n"]
team = ["# Choice vs. rejection frame: 100 stimuli for the twin study\n",
        "Each twin sees all 100 stimuli in both frames, and every stimulus x frame is a separate call with no memory of the others. Each stimulus has three screens. "
        "Screen 1 is a scenario. Screen 2 is a lead-in plus ONE line: the choice frame (the other person chose B) "
        "or the rejection frame (the other person did not choose A). The two "
        "lines are word for word the same except for the decision. Screen 3 asks for a preference on a 7-point "
        "scale (1 = strongly prefer the option listed first, 4 = no preference, 7 = strongly prefer the other); "
        "which option is listed first is random per twin x stimulus, and screen 1 describes the options in that order. Outcome: mean preference for B, by frame.\n",
        "Fixed text around each stimulus: screen 1 starts with *\"" + TASK.split("\n")[0] + "\"*; "
        "screen 3 is *\"Which of the two options would you prefer?\"* with the 7-point scale.\n"]
for s in stimuli:
    full.append(f"\n## {s['stimulus']}: {s['category']}\n")
    for frame in ("choice", "rejection"):
        body = prompt(s, frame, [0, 1])[0].split("[Next screen]")
        full.append(f"**{frame.capitalize()} frame**\n")
        full += [f"> {x}" if x else ">" for x in "\n---\n".join(b.strip() for b in body).splitlines()]
        full.append("")
    team += [f"\n## {s['stimulus']}: {s['category']}\n",
             f"**Screen 1.** {s['layout']['setup']}\n",
             f"> **{s['option_a']['label']}:** {s['layout']['a']}",
             ">",
             f"> **{s['option_b']['label']}:** {s['layout']['b']}\n",
             f"{s['layout']['after']}\n",
             f"**Screen 2.** {s['intro']}\n",
             f"- *Choice frame:* {s['choice_text']}",
             f"- *Rejection frame:* {s['rejection_text']}\n",
             f"**Screen 3.** A. {s['option_a']['label']} / B. {s['option_b']['label']} "
             "(B = the option the other person ended up with)\n"]
(HERE / "screens.md").write_text("\n".join(full))
(HERE / "screens_team.md").write_text("\n".join(team))
print(f"wrote screens.md and screens_team.md: {len(stimuli)} stimuli")
