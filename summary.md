# Kokoro TTS (goonj) — Deploy & Fine-Tune Plan (AWS Primary)

> Research date: Oct 2026. All prices verified against live AWS pricing (Mumbai `ap-south-1`, Linux, 730h/month).
> Context: Oracle/GCP/AWS cardless hosting ruled out earlier (signup KYC). AWS selected as primary provider — **signup card check is the first gate** (see Open Questions).

---

## Framing: two separate tracks

**Fine-tuning ≠ speed.** Training (fine-tuning goonj further) buys *quality* — custom voices, pronunciation quirks, code-mixed Hindi-English prosody. It never makes synthesis faster. Speed comes from optimization (quantization/ONNX) and instance choice — both cheap. The plan runs both tracks independently.

---

## Track A — Speed & efficiency ($0, do first)

Where the actual speed win lives — before paying for any hardware:

1. **ONNX Runtime + int8 dynamic quantization** of the goonj model — biggest CPU win (typically 2–4× on CPU inference; makes ARM hosting feel snappy). One-time local work, $0.
2. **FP16 on MPS** locally (Mac stays latency king at ~0.1–0.3s/sentence).
3. **Preload both variants + voices at startup** — current `prewarm()` is a no-op; first-request latency pays the model-load tax.
4. **Right-size the AWS instance** — with ONNX-int8, GPU serving likely unnecessary.

**Deliverable**: benchmark table (fp32 vs fp16 vs onnx-int8, per-sentence latency, on Mac).

---

## Track B — Fine-tuning goonj (free → paid)

**What it buys**: better pronunciation of domain terms/voices — not speed.
**What it needs first**: paired **audio + text data** (the real gate — goonj itself was trained on Hindi/Indian-English voice corpora). Without ~1–10h of transcribed target audio, fine-tuning is blocked regardless of compute. **First decision point.**

### Free → paid progression

| Stage | Where | Cost | Move on when |
|---|---|---|---|
| **1. Free** | **Kaggle** — T4×2, 30h/week (already used for Qwen) | **$0** | Weekly quota or queueing slows iteration |
| **2. Cheap paid** | AWS **g4dn.xlarge spot, Mumbai** ($0.21–0.24/hr) | **~$0.65–1 per 3h experiment** (on-demand $1.74) | When you want S3 data, longer runs, reproducibility |
| **3. Skip** | SageMaker training jobs (≈ EC2 + 21% managed markup) | Same hardware, higher price | Never — single-person project; spot EC2 or Kaggle wins |

**Verdict**: start on Kaggle (two T4s — more than AWS's one, workflow already proven). Graduate to AWS spot only when iteration speed matters.

---

## Hosting prices (AWS Mumbai `ap-south-1`, 730h/month, Linux)

| Instance | Specs | On-demand/mo | Spot/mo | Role |
|---|---|---|---|---|
| **t4g.medium** | 2 vCPU / 4 GB ARM | **$16.35** | **$7.23** | **Sweet spot** — fits torch + kokoro comfortably |
| c8g.large | 2 vCPU / 4 GB Graviton4 | $39.41 | $16.79 | Faster cores — pick if ONNX still feels slow |
| c7g.large | 2 vCPU / 4 GB Graviton3 | $35.84 | $20.37 | Middle option |
| t4g.small | 2 vCPU / 2 GB ARM | $8.18 | $2.77 | RAM-tight for torch; **750h/mo free through Dec 31 2026** (surplus CPU credits ≈ pennies) |
| g4dn.xlarge | 4 vCPU / 16 GB + **T4 GPU** | **$423** | **~$160** | Serving = overkill; **jobs only** (fine-tune) |

**Incidentals**:
- EBS gp3 ~$0.09/GB-mo (≈$2.74 for 30GB) — billed even when instance stopped (stop = no compute cost, keep EBS)
- Public IPv4: $3.60/mo if attached
- Data-out: first 100GB/mo free (WAV traffic won't dent it)
- **Region**: us-east-1 is ~9% cheaper on GPU but adds ~200ms round-trip — for voice, Mumbai's lower latency wins.

---

## AWS free tier in 2026 (changed — works in our favor)

New accounts (post Jul 2025) are **credit-based**, not the old 12-month micro-instance clock:

- **$100 instantly + up to $100 more** (5 × $20 activities: launch EC2, touch Lambda, set a Budget, etc.) = **$200 over 6 months**
- **Free plan can't bill you** — charges structurally impossible until upgrading to Paid; at 6 months the account pauses, data retained 90 days, upgrade or erased
- $200 covers t4g.medium ($16.35 × 6 ≈ $98) with room left, or t4g.large for all 6 months (~$196, tight)
- **Always-free regardless of plan**: **Lambda = 1M requests + 400,000 GB-seconds/month** — a kokoro container on Lambda at personal-use volume (~100 sentences/day + 10-min keep-warm ping) lands ~140K GB-s/month ≈ **$0 forever**. Cost: ~8–15s cold starts after idle, managed by keep-warm pings (which eat free-tier budget) — genuinely free but fiddlier than EC2
- ⚠️ **Card check**: AWS FAQ says a payment method "isn't required for most new customers" but *may* be requested for identity verification; registration page still mentions a card for account security. **Attempt signup before banking on any of the above.**

---

## The phased plan

| Phase | What | Cost |
|---|---|---|
| **0 — Now** | ONNX/int8 + fp16 + prewarm benchmark locally | **$0** |
| **1 — Fine-tune** | Data audit → train on Kaggle T4×2 → graduate to g4dn spot only if quota hurts | **$0 → ~$1/experiment** |
| **2 — Host (mo 1–6)** | AWS signup (free plan), deploy `kokoro-tts` on **t4g.medium Mumbai** from the $200 credits, budget alert at $1/$10/$50 | **$0** |
| **3 — Host (mo 7+)** | t4g.medium **on-demand $16.35** (safe for voice) or **spot $7.23** (interruptions ≈ occasional dropped sentence); c8g.large spot $16.79 if ONNX says CPU is the bottleneck | **$7–40/mo** |
| Alt — forever-free host | Lambda container, always-free tier + keep-warm | **$0**, cold-start tradeoff |

**Recommendation**: Phase 0 + Kaggle fine-tune immediately (both $0), attempt AWS signup today to clear the card question, then credits-cover t4g.medium Mumbai for 6 months — decide spot-vs-ondemand later with real latency data. **Skip GPU serving entirely**; g4dn spot is the training scalpel.

---

## Open questions (only the user can answer)

1. **Does AWS signup go through without a card?** — first gate; try it before assuming anything AWS-side works. If it demands a card at KYC, AWS is blocked like Oracle was → fall back to cardless options (Kaggle/local).
2. **Fine-tune data & goal** — is there target audio+text data, and what should it do better (a voice, pronunciation, vocabulary)?
