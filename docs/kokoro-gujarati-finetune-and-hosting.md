# Kokoro-82M: Gujarati fine-tune + free hosting plan

Two researched plans: (1) fine-tune kokoro-82M to speak Gujarati for $0,
(2) where to host the model for free. Research verified Sept–Oct 2026.

---

## Part 1 — Fine-tune kokoro-82M for Gujarati at $0

### Verdict: viable at zero cost.

Every blocker is already solved by the community:

- **G2P frontend works locally.** espeak-ng 1.52.0 ships Gujarati (`inc/gu`),
  the same mechanism Hindi uses. No new phonemizer needed.
- **Training recipe exists.** No official hexgrad training code, but
  `semidark/kokoro-deutsch` (reverse-engineered StyleTTS2 pipeline with
  Kokoro-compatible weight patches) is the base every Indian fine-tune used.
  Step-by-step configs: `Jeevav62/tts-finetune-recipes`.
- **Gujarati training data exists** (public, CC-BY-4.0, not gated):
  - `ai4bharat/Rasa` — Gujarati F 25.4h/12,151 utts + M 26.0h/15,799 utts, studio 24kHz.
  - `ai4bharat/indicvoices_r` — Gujarati present (gated), part of 1,704h / 10,496 speakers.
- **Likely zero vocab surgery.** Marathi needed exactly one new phoneme (`ɭ`);
  Hindi needed none. Gujarati's phoneme set is essentially covered by Hindi's.
  It's a *continuation* fine-tune, so stock voicepacks (`af_heart` etc.) plug in
  as ref_s afterward — proven by the Marathi model.

**Precedent for small data:** `jeevav62/kokoro-82m-indian-en` was trained on just
**1,058 clips (~2 hours)**. We don't need Rasa's full 51h — a 1–2k clip subset of
one Gujarati speaker fits Kaggle's weekly quota.

### Free compute (verified 2026)

| Source | Free allowance | Session cap | Predictability |
|---|---|---|---|
| **Kaggle Notebooks** (anchor) | ~30 GPU-h/week (P100 or 2×T4), resets Sat 00:00 UTC | 12 h | High — shown in account settings |
| **Google Colab free** | Unpublished (T4 16GB typical) | 12 h, ~90 min idle timeout | Low — demand-driven |
| HF ZeroGPU | 5 min/day | — | Demo inference only, **not training** |

Strategy: Kaggle is the anchor (30 h/week is enough), Colab for data-prep/sanity.

### The $0 pipeline

**Phase 0 — Setup (CPU only, no GPU)**
1. Fork `semidark/kokoro-deutsch` + `Jeevav62/tts-finetune-recipes`.
2. Download a Rasa Gujarati subset (1–2k clips, one speaker, 24kHz, 3–25s) via
   the recipe's `01_prepare_dataset.py`.
3. Verify G2P with installed espeak-ng `gu`; run a vocab-diff to confirm 0 new
   phoneme slots (Marathi needed 1, `ɭ`; Hindi needed none).

**Phase 1 — Stage 1 on Kaggle**
- Sanity run: 100 clips → confirm loss drops + intelligible audio.
- Full run, T4-adjusted config: `batch_size 1–2` + gradient accumulation + fp16
  (recipe defaults are A100-tuned; this is the known 16GB OOM fix).
- **Checkpoint to HF Hub every epoch** (Hub storage free) — 12-h session kills are
  guaranteed; resume-from-Hub makes them harmless.

**Phase 2 — Stage 2 (GAN/prosody — the hard part)**
- Resume from Stage 1 Hub checkpoint across multiple short sessions.
- Follow `bol-tts-marathi` `docs/TROUBLESHOOTING.md` (written for these Indian runs).
- Fallback if Stage 2 won't converge: ship Stage 1-only (intelligible, flatter
  prosody) and iterate later.

**Phase 3 — Export & wire into our stack**
- Extract voicepacks → test on CPU with espeak gu G2P.
- Add Gujarati routing to `kokoro-tts/app.py`: script range U+0A80–U+0AFF →
  new lang_code + `gu` voice (~10 lines, mirrors the Hindi path).
- Push final weights to HF account (free).

**Honest math:** 2–4 Kaggle sessions/week × 2–3 weeks of evenings = done, $0.
Risks: quota flakiness (Kaggle is predictable), session disconnects (Hub
checkpointing), Stage 2 instability (published errata; may cost extra iterations).

---

## Part 2 — Where to host kokoro-82M for free (practice use)

### Primary host: your own computer (already doing it).

kokoro-82M is **82M params ≈ 330 MB fp32** — designed to be CPU/edge-capable
(the Indian fine-tunes literally market "edge/CPU-capable"). Your worker machine
already runs the sidecar at `localhost:8000`. For practice with no users:

| Option | Verdict | Details |
|---|---|---|
| **Your own Mac/worker** (primary) | Do this | Already working. Zero cost, zero cold start, lowest latency. Auto-restart via `launchd` (macOS) or `docker compose` `restart: always`. After fine-tuning, swap the `.pth` and restart. |
| **Oracle Cloud Always Free ARM** | Best *remote* option | A1.Flex: 2 OCPU + 12 GB RAM free forever — runs the kokoro Docker image 24/7. Caveats: signup friction (card verify, capacity errors); Oracle **reclaims always-free VMs at <20% utilization for 7 days** — a barely-used practice API risks that; a once-a-day ping helps, or accept occasional recreate. |
| Hugging Face Spaces | No longer easy | Free accounts now only create **Static Spaces** (no Python) or **2 ZeroGPU Gradio Spaces**. `cpu-basic` and Docker Spaces need a **paid plan**. (Existing CPU Spaces like `auralodyssey/kokoro-cpu-tts` are grandfathered/sleeping.) |
| Render free | Skip | Free tier is **512 MB RAM** — a torch kokoro server will OOM; also spins down after idle. Only plausible with an ONNX build + cold starts. |
| Cloudflare Workers AI | N/A for us | Serves a *stock* kokoro TTS on free tier — English-only, **not our fine-tuned weights**. |

**Recommended shape:**
- **Practice/dev (now):** your own machine, auto-restarted at boot.
- **Remote always-on demo (later, $0):** Oracle Always Free ARM running the same
  Docker container, weights pulled from your HF Hub account on first boot.

---

## Reference: existing Indian kokoro fine-tunes (for the recipe)

| Model | Language |
|---|---|
| `shreyask/bol-tts-marathi` (HF + full repo/guide) | Marathi + code-switch |
| `Sammy-Mohan/Kokoro-82M-Kannada` | Kannada |
| `jeevav62/kokoro-82m-indian-en` | Indian English |
| `BH-Builds/goonj-1-82M` | Hindi + romanized Hinglish, 15 voices |
| `Bindkushal/IndicVoice-82M` | Hindi, Punjabi, Bengali, English |

**No published Gujarati kokoro exists** — the Marathi repo lists Gujarati as an
intended follow-up. It's a known-desired, unfilled gap.
