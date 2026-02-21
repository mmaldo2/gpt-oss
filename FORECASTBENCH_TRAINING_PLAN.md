# ForecastBench Training Plan: Fine-Tuning an Open-Source Model for Forecasting

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Understanding ForecastBench](#understanding-forecastbench)
3. [Model Selection](#model-selection)
4. [Training Strategy](#training-strategy)
5. [Data Pipeline](#data-pipeline)
6. [Phase 1: Baseline & Scaffolding](#phase-1-baseline--scaffolding)
7. [Phase 2: Supervised Fine-Tuning (SFT)](#phase-2-supervised-fine-tuning-sft)
8. [Phase 3: Reinforcement Learning for Calibration (GRPO)](#phase-3-reinforcement-learning-for-calibration-grpo)
9. [Phase 4: Inference Harness & Submission Pipeline](#phase-4-inference-harness--submission-pipeline)
10. [Phase 5: Iteration & Evaluation](#phase-5-iteration--evaluation)
11. [Hardware & Compute Requirements](#hardware--compute-requirements)
12. [Project Structure](#project-structure)
13. [Key Risks & Mitigations](#key-risks--mitigations)
14. [Learning Roadmap](#learning-roadmap)
15. [References](#references)

---

## Executive Summary

This plan outlines how to train an open-source language model to outperform on
ForecastBench, a dynamic benchmark that measures AI and human forecasting ability
on real-world events. The benchmark uses binary prediction questions sourced from
prediction markets (Manifold, Metaculus, Polymarket, INFER) and time-series
datasets (FRED, ACLED, Yahoo Finance, Wikipedia, DBnomics), scored primarily by
Brier score (lower is better).

**Current state of the art:**
- Superforecasters: ~0.081 Brier score (the target to beat)
- Best LLM (GPT-4.5): ~0.101 Brier score
- LLM-superforecaster parity projected: ~November 2026

**Our approach:** Fine-tune an open-source reasoning model using a three-stage
pipeline (SFT -> GRPO -> calibration), leveraging ForecastBench's own historical
data plus synthetic forecasting data, with a scaffolding harness that provides
the model with retrieval and search capabilities at inference time.

---

## Understanding ForecastBench

### What It Measures

ForecastBench evaluates the ability to assign calibrated probabilities to
real-world events. Every two weeks, participants receive 500 binary questions:

- **250 market questions** from prediction platforms (Manifold, Metaculus,
  Polymarket, INFER) - predict the final outcome
- **250 dataset questions** from time-series data (FRED, ACLED, Yahoo Finance,
  Wikipedia, DBnomics) - predict at up to 8 resolution dates (7 days to 10 years)

### Scoring

| Metric | Description |
|--------|-------------|
| Brier Score | Primary metric: mean of (forecast - outcome)^2. Lower is better. |
| MAE | Mean absolute error on dataset/market questions |
| BSS | Brier Skill Score relative to baselines |
| Difficulty-adjusted | Accounts for varying question difficulty across rounds |

### Key Insight: What Makes a Good Forecaster

The gap between LLMs and superforecasters is primarily about **calibration**, not
knowledge. LLMs tend to be overconfident on uncertain events and underconfident on
near-certain events. The winning strategy is:

1. Correctly identify what information is relevant
2. Retrieve up-to-date information (the model gets a 24-hour window)
3. Reason about base rates, reference classes, and update factors
4. Output a well-calibrated probability (not just "likely" or "unlikely")

### Submission Format

```json
{
  "organization": "team-name",
  "model": "model-identifier",
  "question_set": "2026-03-01-llm.json",
  "forecasts": [
    {
      "id": "question-uuid",
      "source": "manifold",
      "forecast": 0.73,
      "resolution_date": null,
      "reasoning": "Optional but helpful for debugging..."
    }
  ]
}
```

Contact forecastbench@forecastingresearch.org to register. You get a GCP bucket
and 24 hours from question release (0:00 UTC) to submit.

---

## Model Selection

### Recommended: DeepSeek-R1-Distill-Qwen-32B

After evaluating the landscape, this is the recommended base model:

| Criterion | DeepSeek-R1-Distill-Qwen-32B | gpt-oss-20b | Llama-4-Scout |
|-----------|------------------------------|-------------|---------------|
| Reasoning | Excellent (distilled from R1) | Good | Good |
| Size | 32B (fits 1x 80GB GPU w/ QLoRA) | 21B (3.6B active, MoE) | 109B MoE |
| Fine-tuning support | Excellent (standard Transformer) | Poor (MXFP4 MoE, no training code) | Moderate |
| License | MIT (code) + DeepSeek (weights) | Apache 2.0 | Llama license |
| Calibration potential | High (chain-of-thought reasoning) | Moderate | Moderate |
| Community/tooling | Large ecosystem | New, limited | Large ecosystem |

**Why not gpt-oss?** The gpt-oss models are inference-only releases with MXFP4
quantized MoE weights and a custom Harmony chat format. The repo contains no
training code, and the MoE architecture with 128 experts makes fine-tuning
substantially harder than a dense model. It is better suited as a strong baseline
for comparison or as a teacher model for synthetic data generation.

**Why DeepSeek-R1-Distill-Qwen-32B?**
- Already distilled from DeepSeek-R1's reasoning capabilities
- Chain-of-thought reasoning is critical for forecasting (working through base
  rates, evidence, and updating)
- 32B is large enough to be capable but small enough for QLoRA on a single 80GB GPU
- Standard dense Transformer architecture with excellent fine-tuning tooling support
- Qwen-2.5 base has strong multilingual and reasoning foundations

**Fallback options** (if compute-constrained):
- `DeepSeek-R1-Distill-Qwen-7B` - fits on a single 24GB GPU with QLoRA
- `DeepSeek-R1-Distill-Llama-8B` - Llama-based, slightly different strengths
- `Qwen-2.5-32B-Instruct` - if you want to start from a non-reasoning base and
  add forecasting reasoning via GRPO from scratch

---

## Training Strategy

The training pipeline has three stages, each building on the last:

```
Stage 1: SFT (Supervised Fine-Tuning)
  Input: Forecasting reasoning traces + calibrated probabilities
  Goal: Teach the model the FORMAT and REASONING PATTERN of good forecasting

Stage 2: GRPO (Group Relative Policy Optimization)
  Input: Forecasting questions with known resolutions
  Goal: Improve CALIBRATION through RL reward signals tied to Brier score

Stage 3: Calibration Post-Processing
  Input: Validation set predictions
  Goal: Apply temperature scaling / Platt scaling as a final adjustment
```

This mirrors the DeepSeek-R1 training philosophy: SFT provides a warm start,
then RL drives genuine capability improvement beyond what imitation can achieve.

---

## Data Pipeline

### Data Sources (Priority Order)

#### 1. ForecastBench Historical Data (Primary)
From `forecastingresearch/forecastbench-datasets`:

- **Question sets**: `datasets/question_sets/*.json` - historical binary
  forecasting questions with full context (background, resolution criteria,
  market freeze values)
- **Resolution sets**: `datasets/resolution_sets/*.json` - ground truth outcomes
- **Superforecaster predictions**: `datasets/human_forecasts/sf_forecasts/*.json`
  - includes **reasoning traces, search queries, and consulted URLs** from expert
  forecasters

**This is gold.** The superforecaster reasoning traces are exactly the kind of
chain-of-thought data needed for SFT.

#### 2. Synthetic Data from Strong Models
Use GPT-4.5, Claude, or gpt-oss-120b to generate forecasting reasoning traces:

- Take resolved ForecastBench questions
- Prompt the strong model to reason through each question step-by-step
- Filter to keep only traces where the model's probability was well-calibrated
  relative to the actual outcome
- This creates a large corpus of "good forecasting reasoning"

#### 3. Prediction Market Historical Data
Scrape or API-fetch historical resolved questions from:
- Metaculus (API available, thousands of resolved questions)
- Manifold Markets (API available, large question corpus)
- Polymarket (resolved markets)
- Good Judgment Open

For each, capture: question text, background, resolution criteria, resolution
date, actual outcome, and (where available) community probability over time.

#### 4. Time-Series Forecasting Data
Generate synthetic forecasting questions from:
- FRED economic indicators (historical values -> "Will X exceed Y by date Z?")
- Yahoo Finance (stock/index movement questions)
- Wikipedia page views (trending topic prediction)
- ACLED conflict data (event count predictions)

This directly mirrors how ForecastBench generates its dataset questions.

### Data Format for Training

Each training example should follow this structure:

```
<|system|>
You are an expert forecaster. For each question, reason step-by-step about:
1. Base rates and reference classes
2. Current evidence and recent trends
3. Key factors that could push the probability up or down
4. Your final calibrated probability estimate between 0 and 1

<|user|>
Question: Will the Federal Reserve cut interest rates by at least 25 basis points
at its next meeting?

Background: [relevant context...]
Resolution criteria: Resolves YES if the Federal Reserve announces a rate cut
of 25bp or more at the FOMC meeting on [date].
Current market probability: 0.65 (as of [freeze date])

<|assistant|>
<think>
Let me work through this systematically.

**Base rate**: Over the last 20 FOMC meetings, the Fed has cut rates in 6 of
them (30%). However, this base rate is less informative than recent trends.

**Current evidence**:
- Recent CPI came in at X%, below expectations
- Employment numbers show [trend]
- Fed governors have signaled [dovish/hawkish stance]

**Updating factors**:
- [Factor pushing probability up]: +0.15
- [Factor pushing probability down]: -0.05

**Calibration check**: The prediction market sits at 0.65. Markets are generally
well-calibrated for near-term policy questions. My independent estimate aligns
roughly with this.
</think>

My forecast: 0.68
```

---

## Phase 1: Baseline & Scaffolding

### Goals
- Establish baseline performance of the unmodified model on ForecastBench questions
- Build the inference harness that will be used for all subsequent evaluation
- Build the data processing pipeline

### Step 1.1: Set Up the Inference Harness

Build a Python pipeline that:

1. **Loads a ForecastBench question set** from the datasets repo
2. **For each question**, constructs a prompt with:
   - System prompt encoding forecasting best practices
   - The question text, background, and resolution criteria
   - Any retrieved context (web search results, recent data)
3. **Runs inference** through the model (local or API)
4. **Parses the output** to extract the probability forecast (0-1 float)
5. **Formats the submission** JSON per ForecastBench spec
6. **Evaluates locally** against resolved questions using Brier score

```
forecastbench-harness/
  harness/
    __init__.py
    question_loader.py      # Parse ForecastBench question JSONs
    prompt_builder.py       # Construct forecasting prompts
    inference_runner.py     # Run model inference (supports multiple backends)
    probability_parser.py   # Extract float probabilities from model output
    submission_formatter.py # Format for ForecastBench upload
    evaluator.py           # Local Brier score evaluation
    retrieval.py           # Web search / context retrieval (optional)
  configs/
    base_prompt.yaml       # System prompt templates
    model_configs.yaml     # Model-specific settings
  scripts/
    run_baseline.py        # Run baseline evaluation
    run_evaluation.py      # Evaluate against resolved questions
    submit.py             # Upload to GCP
```

### Step 1.2: Baseline Evaluation

Run the unmodified DeepSeek-R1-Distill-Qwen-32B against historical ForecastBench
questions (using resolved questions so you can score immediately).

Measure:
- Overall Brier score
- Market questions vs dataset questions (separate scores)
- Calibration curve (plot predicted probability vs actual frequency)
- Common failure modes (where does the model go wrong?)

### Step 1.3: Prompt Engineering Baseline

Before any fine-tuning, optimize the prompt. Test variations:

1. **Zero-shot**: Just the question
2. **System prompt with forecasting instructions**: Base rates, calibration checks
3. **Few-shot with superforecaster examples**: Include 2-3 examples from the
   superforecaster predictions dataset
4. **Chain-of-thought**: Explicitly request step-by-step reasoning
5. **With retrieval**: Add web search context for market questions

This gives you a strong prompt-engineering baseline to improve upon with
fine-tuning.

---

## Phase 2: Supervised Fine-Tuning (SFT)

### Goals
- Teach the model the structure of good forecasting reasoning
- Improve probability extraction reliability
- Build on superforecaster reasoning patterns

### Step 2.1: Prepare the SFT Dataset

**Source 1: Superforecaster reasoning traces**

From `forecastbench-datasets/datasets/human_forecasts/sf_forecasts/*.json`:

```python
# Pseudocode for processing superforecaster data
for forecast in superforecaster_forecasts:
    question = lookup_question(forecast["id"])
    resolution = lookup_resolution(forecast["id"])

    training_example = {
        "system": FORECASTING_SYSTEM_PROMPT,
        "user": format_question(question),
        "assistant": format_reasoning_and_forecast(
            reasoning=forecast["reasoning"],
            searches=forecast["searches"],
            forecast=forecast["forecast"],
            # Include calibration commentary based on actual resolution
            was_well_calibrated=assess_calibration(
                forecast["forecast"], resolution
            )
        )
    }
```

**Source 2: Synthetic reasoning from strong models**

```python
# Use GPT-4.5 or Claude to generate reasoning traces
for question, resolution in resolved_questions:
    for attempt in range(3):  # Multiple samples for diversity
        response = strong_model.generate(
            system=FORECASTING_SYSTEM_PROMPT,
            user=format_question(question),
            temperature=0.7
        )
        forecast_value = parse_probability(response)

        # Only keep well-calibrated responses
        brier = (forecast_value - resolution) ** 2
        if brier < 0.15:  # Threshold for "good enough" calibration
            training_data.append(format_as_training_example(
                question, response, forecast_value
            ))
```

**Source 3: Self-generated reasoning**

Run the base model on resolved questions, keep the best outputs:
```python
# "Rejection sampling" - generate many, keep the best
for question, resolution in resolved_questions:
    responses = base_model.generate(question, n=10, temperature=0.8)
    best = min(responses, key=lambda r: brier_score(parse_prob(r), resolution))
    if brier_score(parse_prob(best), resolution) < 0.12:
        training_data.append(best)
```

**Target dataset size**: 5,000-20,000 examples (mix of all three sources).

### Step 2.2: Run SFT with QLoRA

**Framework: Unsloth** (recommended for speed/memory efficiency)

```python
from unsloth import FastLanguageModel

# Load base model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    max_seq_length=8192,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,                    # LoRA rank (64 is a good starting point)
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=64,           # Scaling factor
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Memory optimization
)

# SFT training
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=forecasting_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=10,
        output_dir="outputs/sft-forecaster",
        save_strategy="steps",
        save_steps=200,
    ),
)

trainer.train()
```

**Key hyperparameters to tune:**
- `r` (LoRA rank): Start with 64, try 32 and 128
- `learning_rate`: 1e-5 to 5e-5 range
- `num_train_epochs`: 2-5 (watch for overfitting on small datasets)
- `max_seq_length`: 4096-8192 (forecasting reasoning can be long)

### Step 2.3: Evaluate SFT Model

Re-run the baseline evaluation. Compare:
- Brier score improvement over baseline
- Calibration curve shape
- Reasoning quality (manual inspection of 50+ examples)
- Format compliance (does it reliably output parseable probabilities?)

---

## Phase 3: Reinforcement Learning for Calibration (GRPO)

### Why RL After SFT?

SFT teaches the model to **imitate** good forecasting. But imitation has limits:
- The model learns the format but may not deeply internalize calibration
- It may copy reasoning patterns without truly understanding when to be confident
  vs uncertain
- RL directly optimizes for the **outcome** (calibrated probability), not just
  pattern matching

GRPO (Group Relative Policy Optimization) is the method used to train DeepSeek-R1
and is particularly suited here because:
- No critic/value model needed (saves compute)
- Works well with verifiable rewards (Brier score is perfectly verifiable)
- Proven to improve reasoning and calibration

### Step 3.1: Define the Reward Function

```python
def forecasting_reward(model_output: str, resolution: float) -> float:
    """
    Reward function for GRPO training.

    Args:
        model_output: Full model generation including reasoning
        resolution: 0.0 or 1.0 (actual outcome)

    Returns:
        reward: float, higher is better
    """
    forecast = parse_probability(model_output)

    if forecast is None:
        return -1.0  # Penalty for unparseable output

    # Negative Brier score (we want to maximize reward, minimize Brier)
    brier = (forecast - resolution) ** 2
    reward = 1.0 - brier  # Range: [0, 1], higher is better

    # Bonus for showing reasoning structure
    has_reasoning = "<think>" in model_output and "</think>" in model_output
    if has_reasoning:
        reward += 0.05

    # Bonus for considering base rates
    mentions_base_rate = any(
        term in model_output.lower()
        for term in ["base rate", "reference class", "historical", "typically"]
    )
    if mentions_base_rate:
        reward += 0.02

    return reward
```

### Step 3.2: Run GRPO Training

```python
from trl import GRPOTrainer, GRPOConfig

# Load SFT checkpoint
model = load_sft_checkpoint("outputs/sft-forecaster/best")

grpo_config = GRPOConfig(
    output_dir="outputs/grpo-forecaster",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=5e-7,          # Much lower LR for RL
    max_completion_length=4096,
    num_generations=8,            # GRPO group size (generate 8, rank them)
    temperature=0.8,
    beta=0.1,                     # KL penalty coefficient
    logging_steps=5,
    save_steps=100,
)

trainer = GRPOTrainer(
    model=model,
    config=grpo_config,
    reward_funcs=[forecasting_reward],
    train_dataset=resolved_questions_dataset,
)

trainer.train()
```

**Key GRPO parameters:**
- `num_generations`: 4-16 (more = better signal but more compute)
- `beta`: 0.01-0.1 (lower = more exploration, higher = stay close to SFT)
- `learning_rate`: 1e-7 to 1e-6 (RL requires very small steps)
- `temperature`: 0.7-1.0 (needs diversity in generations for GRPO to work)

### Step 3.3: Monitor Training

Watch for:
- **Reward increase** over training steps
- **KL divergence** staying reasonable (< 5.0 from SFT model)
- **Calibration curve** on held-out questions improving
- **No reward hacking** (e.g., model always predicting 0.5 to minimize worst-case)

---

## Phase 4: Inference Harness & Submission Pipeline

### Retrieval-Augmented Forecasting

At inference time, the model benefits greatly from current information. Build a
retrieval pipeline:

```
Question received (0:00 UTC)
  |
  v
[Question Parser] -> Extract key entities, dates, topics
  |
  v
[Web Search] -> Search for latest news, data updates
  |          -> Use search APIs (Brave, Exa, SerpAPI)
  |
  v
[Data Retrieval] -> For dataset questions:
  |               -> Fetch latest values from FRED, Yahoo Finance, etc.
  |
  v
[Context Assembly] -> Combine question + retrieved context
  |
  v
[Model Inference] -> Run fine-tuned model
  |                -> Parse probability output
  |
  v
[Ensemble / Post-Processing] -> Optional: average across multiple runs
  |                           -> Apply temperature/Platt scaling
  |
  v
[Submission Formatter] -> Generate ForecastBench JSON
  |
  v
[Upload to GCP] -> Submit within 24-hour window
```

### Calibration Post-Processing

After all predictions are generated, apply a learned calibration adjustment:

```python
from sklearn.isotonic import IsotonicRegression

# Fit on validation set (historical resolved questions)
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(validation_predictions, validation_outcomes)

# Apply to new predictions
calibrated_forecasts = calibrator.predict(raw_forecasts)
```

This is a simple but effective final step. Temperature scaling and Platt scaling
are alternatives worth testing.

### Ensembling

Run the model 3-5 times with different temperatures and average:

```python
forecasts = []
for temp in [0.3, 0.5, 0.7]:
    forecast = run_inference(question, temperature=temp)
    forecasts.append(forecast)
final_forecast = np.mean(forecasts)
```

This reduces variance and improves calibration.

---

## Phase 5: Iteration & Evaluation

### Evaluation Loop

```
For each historical ForecastBench round (biweekly):
  1. Load that round's question set
  2. Run your model's full pipeline (retrieval -> inference -> calibration)
  3. Score against known resolutions
  4. Log: Brier score, calibration curve, per-source breakdown
  5. Compare against: baseline model, superforecaster median, top LLMs
```

### Ablation Studies

Test the contribution of each component:
- Base model alone (no fine-tuning)
- + SFT only
- + SFT + GRPO
- + Retrieval augmentation
- + Calibration post-processing
- + Ensembling

### Continuous Improvement

ForecastBench runs biweekly. Use each round to:
1. Submit your current best model
2. When resolutions come in, add to training data
3. Re-train periodically with expanded dataset
4. Track your position on the leaderboard over time

---

## Hardware & Compute Requirements

### Option A: Single GPU (Budget-Friendly)

| Component | Spec | Cost Estimate |
|-----------|------|---------------|
| GPU | 1x NVIDIA A100 80GB or H100 80GB | ~$2-3/hr on RunPod/Lambda |
| RAM | 64GB+ system RAM | Included |
| Storage | 200GB+ SSD | Included |

**What fits:**
- QLoRA fine-tuning of 32B model: Yes
- GRPO of 32B model: Tight, may need 7B fallback
- Inference of 32B: Yes (with quantization)

### Option B: Multi-GPU (Recommended)

| Component | Spec | Cost Estimate |
|-----------|------|---------------|
| GPU | 2-4x A100 80GB | ~$6-12/hr on RunPod/Lambda |
| RAM | 128GB+ system RAM | Included |
| Storage | 500GB+ SSD | Included |

**What fits:**
- Full LoRA fine-tuning of 32B model: Yes
- GRPO of 32B model with larger batch sizes: Yes
- Fast iteration cycles

### Option C: Minimal (Learning-Focused)

| Component | Spec | Cost Estimate |
|-----------|------|---------------|
| GPU | 1x RTX 4090 24GB or A10G 24GB | ~$0.50-1/hr or own hardware |
| RAM | 32GB+ system RAM | - |
| Storage | 100GB+ SSD | - |

**What fits:**
- QLoRA fine-tuning of 7B model: Yes
- GRPO of 7B model: Yes
- Inference of 7B: Yes

This is sufficient for learning the full pipeline, even if the 7B model
won't be as competitive as the 32B version.

### Cloud Provider Recommendations

- **RunPod**: Best value for on-demand GPU rental
- **Lambda Labs**: Good for reserved instances
- **Google Colab Pro+**: A100 access for ~$50/month (limited hours)
- **Modal**: Pay-per-second, good for inference pipelines
- **Hugging Face Spaces**: Free T4 for inference demos

---

## Project Structure

```
gpt-oss/                              # This repo (fork)
  FORECASTBENCH_TRAINING_PLAN.md      # This plan
  forecasting/                         # New directory for our work
    README.md
    setup.py / pyproject.toml
    configs/
      model_config.yaml               # Model selection & params
      sft_config.yaml                 # SFT hyperparameters
      grpo_config.yaml                # GRPO hyperparameters
      prompt_templates.yaml           # System prompts
    data/
      prepare_sft_data.py             # Process superforecaster + synthetic data
      prepare_grpo_data.py            # Process resolved questions for RL
      synthetic_generation.py         # Generate synthetic training data
      question_templates.py           # Template-based question generation
    training/
      sft_train.py                    # SFT training script
      grpo_train.py                   # GRPO training script
      reward_functions.py             # Reward functions for RL
      calibration.py                  # Post-hoc calibration fitting
    harness/
      question_loader.py
      prompt_builder.py
      inference_runner.py
      probability_parser.py
      submission_formatter.py
      evaluator.py
      retrieval.py
    scripts/
      run_baseline.py
      run_evaluation.py
      run_sft.sh
      run_grpo.sh
      submit.py
    notebooks/
      01_data_exploration.ipynb       # Explore ForecastBench data
      02_baseline_evaluation.ipynb    # Baseline model performance
      03_sft_analysis.ipynb           # SFT results analysis
      04_grpo_analysis.ipynb          # GRPO results analysis
      05_calibration_analysis.ipynb   # Calibration curves and tuning

forecastbench-datasets/               # Companion repo (fork)
  datasets/
    question_sets/                    # Input questions (existing)
    resolution_sets/                  # Ground truth (existing)
    human_forecasts/                  # Superforecaster data (existing)
  processed/                          # New directory
    sft_training_data.jsonl           # Processed SFT examples
    grpo_training_data.jsonl          # Processed RL examples
    validation_set.jsonl              # Held-out evaluation set
    calibration_set.jsonl             # For calibration fitting
```

---

## Key Risks & Mitigations

### Risk 1: Small Dataset Size
ForecastBench has ~2 years of biweekly rounds. That's roughly 50 rounds x 500
questions = 25,000 questions, but many won't have resolved yet.

**Mitigation**: Synthetic data generation from strong models, plus external
prediction market data (Metaculus alone has 10,000+ resolved questions).

### Risk 2: Overfitting to Historical Questions
If you train on resolved ForecastBench questions and evaluate on the same
distribution, performance may not transfer to new questions.

**Mitigation**: Strict train/validation/test splits by date (train on older
rounds, evaluate on newer ones). Also evaluate on out-of-distribution forecasting
questions from Metaculus/Manifold.

### Risk 3: MoE Models Are Hard to Fine-Tune
If you want to use gpt-oss-120b or gpt-oss-20b, the MXFP4 MoE architecture
complicates fine-tuning significantly.

**Mitigation**: Use gpt-oss as a teacher model (for synthetic data generation)
rather than as the fine-tuning target. Use a dense model (DeepSeek-R1-Distill)
as the student.

### Risk 4: RL Training Instability
GRPO can be unstable, especially with small batch sizes or poorly designed rewards.

**Mitigation**: Start with SFT (which is stable and predictable), then carefully
introduce GRPO with conservative hyperparameters. Monitor KL divergence closely.
Fall back to DPO (Direct Preference Optimization) if GRPO proves too unstable --
DPO is simpler and more stable, though potentially less effective.

### Risk 5: Retrieval Quality at Inference Time
Web search quality varies. Poor retrieval can hurt forecasting more than help it.

**Mitigation**: A/B test retrieval vs no-retrieval. For dataset questions (time-
series based), prefer direct API calls to data sources (FRED API, Yahoo Finance
API) over web search. For market questions, web search is more valuable.

---

## Learning Roadmap

This project is structured so each phase teaches a distinct skill:

### Phase 1: Inference & Evaluation (Week 1-2)
**You'll learn:**
- How to run open-source LLMs locally (vLLM, Hugging Face)
- Prompt engineering for structured outputs
- Evaluation methodology (Brier scores, calibration curves)
- Working with the ForecastBench data format

**Key resources:**
- vLLM docs: https://docs.vllm.ai/
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- Brier score: https://en.wikipedia.org/wiki/Brier_score

### Phase 2: SFT Fine-Tuning (Week 3-4)
**You'll learn:**
- QLoRA / LoRA: How parameter-efficient fine-tuning works
- Data preparation for instruction tuning
- Training loop management (checkpoints, logging, evaluation)
- Unsloth framework for efficient training

**Key resources:**
- Unsloth: https://github.com/unslothai/unsloth
- LoRA paper: https://arxiv.org/abs/2106.09685
- TRL (Transformer Reinforcement Learning): https://huggingface.co/docs/trl

### Phase 3: RL Fine-Tuning (Week 5-7)
**You'll learn:**
- Reinforcement learning from human feedback (RLHF) concepts
- GRPO algorithm and implementation
- Reward function design
- Training stability monitoring

**Key resources:**
- GRPO in TRL: https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl
- DeepSeek-R1 paper: https://arxiv.org/abs/2501.12948
- DeepLearning.AI GRPO course: https://www.deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo/

### Phase 4: Production Pipeline (Week 8-9)
**You'll learn:**
- Retrieval-augmented generation (RAG)
- Calibration techniques (temperature scaling, isotonic regression)
- Ensembling methods
- Production deployment and submission automation

### Phase 5: Ongoing Competition (Week 10+)
**You'll learn:**
- Iterative model improvement
- Analyzing model failures and addressing them
- Tracking and comparing model versions
- The art of forecasting itself

---

## References

### ForecastBench
- Paper: https://arxiv.org/abs/2409.19839 (ICLR 2025)
- Website: https://www.forecastbench.org/
- Code: https://github.com/forecastingresearch/forecastbench
- Datasets: https://github.com/forecastingresearch/forecastbench-datasets
- Submission guide: https://github.com/forecastingresearch/forecastbench/wiki/How-to-submit-to-ForecastBench

### Models
- DeepSeek-R1: https://github.com/deepseek-ai/DeepSeek-R1
- DeepSeek-R1-Distill-Qwen-32B: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- gpt-oss: https://github.com/openai/gpt-oss
- Qwen-2.5: https://huggingface.co/Qwen

### Training Frameworks
- Unsloth: https://github.com/unslothai/unsloth
- TRL (GRPO, SFT, DPO): https://huggingface.co/docs/trl
- Axolotl: https://github.com/axolotl-ai-cloud/axolotl
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory

### Forecasting Methodology
- Superforecasting (Tetlock): https://en.wikipedia.org/wiki/Superforecasting
- Calibration training: https://calibrateduncertainty.org/
- Brier score: https://en.wikipedia.org/wiki/Brier_score

### Related Papers
- AIA Forecaster (LLM forecasting system): https://arxiv.org/abs/2511.07678
- Calibrating Verbalized Probabilities for LLMs: https://arxiv.org/abs/2410.06707
- Thermometer (Universal LLM Calibration): https://arxiv.org/abs/2406.15309
