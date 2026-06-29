# Provenance Guard — Portfolio Walkthrough Script
# ~3 minutes | Record with QuickTime Player > New Screen Recording

## Pre-recording setup (do this BEFORE hitting record)
```bash
cd ~/Desktop/Codepath/ai201-project4-provenance-guard
source .venv/bin/activate
# Open TWO terminal tabs:
# Tab 1: flask server
# Tab 2: curl commands
```

---

## [TAB 1] Start the Flask server (show this first)
```bash
python3 app.py
```
Point out: "Running on http://127.0.0.1:5001"

---

## [TAB 2] Demo commands — paste these in order

### Scene 1 — Show the architecture (~20 sec)
Open README.md in your browser (GitHub) or in VS Code.
Say: "This is Provenance Guard — a content attribution API.
A submitted piece of text passes through two detection signals,
a confidence combiner, and gets a transparency label."

### Scene 2 — Submit clearly human text (~30 sec)
```bash
curl -s -X POST http://localhost:5001/submit \
  -H "Content-Type: application/json" \
  -d '{
    "text": "ok so i finally tried that new ramen place downtown and honestly? underwhelming. the broth was fine but they put WAY too much sodium in it and i was thirsty for like three hours after.",
    "creator_id": "demo-user"
  }' | python3 -m json.tool
```
Say: "Casual human writing — LLM scores it 0.21, stylometric 0.38,
combined confidence 0.27. The label reads: Likely Human-Written."

### Scene 3 — Submit clearly AI text (~30 sec)
```bash
curl -s -X POST http://localhost:5001/submit \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits are numerous, it is equally essential to consider the ethical implications. Furthermore, stakeholders across various sectors must collaborate to ensure responsible deployment.",
    "creator_id": "demo-user"
  }' | python3 -m json.tool
```
Say: "Formulaic AI text — LLM gives 0.91. But stylometric only 0.46.
The 0.45 gap triggers disagreement dampening — score drops to 0.65,
landing in the uncertain band. This is the system being honest
about conflicting signals rather than overclaiming."

### Scene 4 — Save the content_id and file an appeal (~25 sec)
```bash
# Copy the content_id from the previous response, then:
curl -s -X POST http://localhost:5001/appeal \
  -H "Content-Type: application/json" \
  -d '{
    "content_id": "PASTE-ID-HERE",
    "creator_reasoning": "I wrote this myself. I can provide my original draft."
  }' | python3 -m json.tool
```
Say: "The creator disputes the label. The appeal captures their
reasoning and marks the submission under_review for human review."

### Scene 5 — View the audit log (~20 sec)
```bash
curl -s http://localhost:5001/log | python3 -m json.tool
```
Say: "Every decision is in the structured audit log —
content ID, confidence, both signal scores, and the appeal."

### Scene 6 — Rate limiting (~20 sec)
```bash
for i in $(seq 1 12); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:5001/submit \
    -H "Content-Type: application/json" \
    -d '{"text": "test", "creator_id": "rl-test"}'
done
```
Say: "12 rapid requests — 10 succeed, then 429. The limit is
10/minute and 100/day, calibrated to realistic creator usage."

### Scene 7 — Design decision wrap-up (~20 sec)
Say: "The key design insight: false positives on human writing
are worse than false negatives. The system uses a wide uncertain
band (0.35–0.70), disagreement dampening, and a first-class appeal
path to err on the side of giving creators the benefit of the doubt."

---

## Recording tips
- Use QuickTime Player > New Screen Recording > select terminal window
- Speak slowly — 3 minutes is plenty of time
- Don't worry about mistakes — stop, delete, re-record that segment
- Trim with iMovie or QuickTime's Edit > Trim before submitting
