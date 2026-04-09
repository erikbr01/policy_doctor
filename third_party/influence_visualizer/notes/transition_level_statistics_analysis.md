# Transition-Level Influence Statistics: Analysis and Interpretation

**Date:** 2026-01-27  
**Context:** Analysis of transition-level influence matrices showing correlations between demonstration quality, influence statistics, and rollout-demo pair characteristics.

---

## Overview

This document summarizes insights from analyzing transition-level influence matrices across all rollout-demonstration pairs. Each transition matrix represents the influence of a demonstration (timestep by timestep) on a specific rollout episode.

## Key Findings from Correlation Analysis

### 1. **Performance Influence vs. Mean Influence (r = 0.874)**

**Finding:** Very strong positive correlation between performance influence and average mean influence.

**Interpretation:**
- Demonstrations with higher average mean influence scores have much higher performance influence (contribution to successful rollouts)
- This validates that **mean influence is an excellent predictor of demonstration quality**
- The automated influence-based metric outperforms human quality labels for predicting which demos help successful rollouts

**Implication:** Mean influence is more objective and reliable than human quality assessment for identifying helpful demonstrations.

---

### 2. **Human Quality Labels Show Weak Correlations**

**Findings:**
- Q: Better ↔ Performance Influence: **0.207** (modest positive)
- Q: Worse ↔ Performance Influence: **-0.185** (modest negative)
- Q: Better ↔ Avg Std: **0.04** (essentially zero)

**Interpretation:**
- Human quality labels ("better", "okay", "worse") capture **trajectory clarity and task relevance**, not influence magnitude
- "Better" = clear, purposeful trajectory focused on task
- "Worse" = random, noisy actions unrelated to task
- "Okay" = somewhere in between

**Key Insight:** A clear, well-executed trajectory isn't automatically high-influence! Clear demos might be demonstrating the "wrong" strategy for the learned policy, or might not align with what the policy learned.

---

### 3. **Avg Std ↔ Std of Stds (r = 0.944)**

**Finding:** Very strong positive correlation between average standard deviation and the variability of that standard deviation across rollouts.

**What This Means:**
- Demos with **high average variance** (spread-out influence patterns) also show **high cross-rollout variance** (inconsistent patterns across different rollouts)
- Demos with **low average variance** (concentrated influence) show **low cross-rollout variance** (consistent patterns)

**Interpretation - Two Types of Demonstrations:**

#### Type 1: Stable, Concentrated Influence Demos
- **Low Avg Std:** Influence values are uniform/similar across the transition matrix
- **Low Std of Stds:** This pattern is consistent regardless of which rollout
- **Characteristics:** Predictable, focused influence; might be random/noisy demos that produce flat, undifferentiated influence
- **Example:** "Worse" quality demos with random actions → uniform low influence everywhere

#### Type 2: Variable, Complex Influence Demos  
- **High Avg Std:** Influence values vary widely within the transition matrix
- **High Std of Stds:** This variance pattern changes depending on the rollout
- **Characteristics:** Complex, context-dependent influence; adapts to different rollout needs
- **Example:** "Better" quality demos with clear, purposeful actions → structured influence patterns that vary by context

**Key Takeaway:** **Influence variability is a fundamental, stable property of a demonstration**, not random noise. The strong correlation (0.944) shows demos don't randomly fluctuate between concentrated and spread-out influence—they have an intrinsic "structuredness."

---

### 4. **Avg Min ↔ Avg Max (r = -0.928)**

**Finding:** Very strong negative correlation between minimum and maximum influence values.

**Interpretation:**
- Demonstrations are **predominantly one-sided** in their influence
- Demos with very negative minimum influences tend to have lower (less positive) maximum influences
- Demos with less negative minimums tend to have higher positive maximums
- This suggests demos are either "generally positive" or "generally negative" in influence, not mixed

**Implication:** Influence patterns are relatively consistent within a demonstration—a demo doesn't simultaneously strongly help some actions while strongly hurting others.

---

## Understanding the Scatter Plot Separation

The clear two-cluster separation visible in Mean vs. Std scatter plots likely represents:

### **Upper Cluster (High Std):**
- Clear, structured demonstrations with context-dependent influence
- Purposeful actions create distinctive patterns
- Some timesteps matter significantly more than others
- Influence adapts based on rollout needs

### **Lower Cluster (Low Std):**
- Random/noisy demonstrations with flat, uniform influence
- No particular timestep stands out
- Consistent (low) influence across all rollout pairings
- Undifferentiated noise that neither strongly helps nor hurts

---

## Relationship Between Quality Labels and Influence Patterns

### "Worse" Quality Demos (Random/Noisy):
- **Within matrix:** Low variance (flat influence)
- **Across rollouts:** Low cross-rollout variance (consistently flat)
- **Avg Min:** Slightly less negative (closer to 0) → random actions don't create strong negative patterns
- Random actions produce uniform, undifferentiated influence

### "Better" Quality Demos (Clear/Purposeful):
- **Within matrix:** High variance (structured influence patterns)
- **Across rollouts:** High cross-rollout variance (context-dependent adaptation)
- **Std of Means:** Higher variability (0.25 correlation) → influence changes based on what rollout needs
- Clear actions create distinctive, context-aware influence

### "Okay" Quality Demos:
- Intermediate patterns between "worse" and "better"
- Less consistent characteristics

---

## Individual Rollout-Demo Pair Analysis

The new "Individual Rollout-Demo Pairs" view allows testing hypotheses such as:

### Hypothesis: Specific Demos Have Systematically High Variance
**Test:** Check **Demo ID ↔ Std** correlation in pair-level analysis
- **Strong correlation** → certain demonstrations always produce high variance
- **Weak correlation** → variance is more dependent on rollout context

### Hypothesis: Specific Rollouts Elicit High Variance
**Test:** Check **Rollout ID ↔ Std** correlation
- **Strong correlation** → certain rollouts produce high variance with all demos
- **Weak correlation** → variance is more a property of the demo

---

## Implications for Demonstration Selection

1. **Use Mean Influence, Not Just Quality Labels**
   - Mean influence (r=0.874 with performance) is more predictive than human labels
   - Quality labels capture something different (clarity vs. helpfulness)

2. **High Variance ≠ Bad**
   - High variance might indicate structured, context-dependent influence (good)
   - Low variance might indicate flat, noisy influence (bad)
   - Context matters!

3. **Influence Structuredness is Intrinsic**
   - The 0.944 correlation shows that "how structured" a demo's influence is remains stable
   - This is a fundamental characteristic worth measuring

4. **Performance Influence Metric is Robust**
   - The automated performance influence calculation captures what matters for successful rollouts
   - More reliable than human judgment for selecting training demonstrations

---

### 5. **Rollout Success ↔ Influence Std (r ≈ -0.7)**

**Finding:** Strong negative correlation between rollout success and the standard deviation of influence in individual rollout-demo pair matrices.

**Interpretation:**
- **Successful rollouts** have **lower influence variance** (more uniform influence patterns)
- **Failed rollouts** have **higher influence variance** (chaotic, inconsistent influence patterns)

**What This Reveals About Policy Behavior:**

#### Successful Rollouts (Low Std):
- Policy receives **consistent, coherent guidance** from training data
- Influence is distributed relatively uniformly across demonstrations
- The policy has learned a **clear, stable strategy**
- Low variance indicates confidence and consistency

#### Failed Rollouts (High Std):
- Policy receives **conflicting signals** from different parts of training data
- Influence is highly variable—some demos pull strongly in one direction, others in another
- The policy is **uncertain or receiving mixed guidance**
- High variance indicates confusion or lack of coherent strategy

**Key Insight:** **Influence variance is a proxy for policy confidence and decision coherence**. High variance in influence suggests the policy hasn't learned a consistent internal representation for that rollout state.

---

## Looping Back to Interpretability & Data Curation

The strong success-std correlation opens powerful opportunities for improving training data and understanding failure modes.

### 1. **Training Data Quality Indicator**

**Insight:** High influence std in failed rollouts indicates conflicting training signals.

**Actionable Steps:**
- Identify demonstrations that contribute to high-variance influences on failed rollouts
- Review these demos for:
  - **Contradictory behaviors** (Demo A shows strategy X, Demo B shows opposite strategy)
  - **Ambiguous actions** (Same state, different actions across demos)
  - **Suboptimal patterns** (Demos showing ineffective strategies)
- Flag or remove demonstrations that create conflicting guidance

**Implementation:**
```python
for failed_rollout in failed_rollouts:
    high_std_demos = get_high_variance_influencers(failed_rollout, threshold=2.0)
    flag_for_review(high_std_demos, reason="Creates conflicting influence patterns")
```

---

### 2. **Failure Mode Taxonomy**

**Insight:** Not all failures are the same—std reveals WHY the policy failed.

**Two Types of Failures:**

#### Type A: "Confused Policy" (High Std)
- **Cause:** Received conflicting guidance from training data
- **Symptom:** Erratic influence patterns, no clear dominant strategy
- **Solution:** Need more consistent demonstrations for that scenario
- **Remedy:** Add clear, unambiguous demos or remove conflicting ones

#### Type B: "Confidently Wrong" (Low Std)  
- **Cause:** Learned a consistent but incorrect strategy
- **Symptom:** Uniform influence pointing toward wrong behavior
- **Solution:** Need demonstrations showing correct behavior for that context
- **Remedy:** Add counter-examples or correct demonstrations

**Implementation:**
```python
if rollout.failed and influence_std > threshold:
    failure_type = "uncertain_policy"  # High variance
    recommendation = "Remove conflicting demos or add consistent guidance"
elif rollout.failed and influence_std < threshold:
    failure_type = "systematic_failure"  # Low variance, wrong strategy
    recommendation = "Add demonstrations showing correct behavior"
```

---

### 3. **Demonstration Curriculum Design**

**Insight:** Some demos are "harder to learn" (create high variance) than others.

**Staged Training Approach:**

**Phase 1: Foundation (Low-Variance Demos)**
- Train on demonstrations that create consistent, low-variance influence
- These provide clear, unambiguous signals
- Build stable base strategy

**Phase 2: Refinement (Moderate-Variance Demos)**
- Add demonstrations with slightly more complex influence patterns
- Policy can now handle moderate ambiguity

**Phase 3: Edge Cases (High-Variance Demos)**
- Finally introduce complex demos that previously caused high variance
- Stable foundation allows policy to learn nuanced behaviors

**Implementation:**
```python
demos_sorted_by_learnability = sort_demos_by_avg_variance_on_rollouts(demos)
train_curriculum = [
    demos_sorted_by_learnability[:N//3],   # Low variance (easy)
    demos_sorted_by_learnability[N//3:2*N//3],  # Medium variance
    demos_sorted_by_learnability[2*N//3:],  # High variance (complex)
]
```

---

### 4. **Active Learning for Data Collection**

**Insight:** High influence std indicates states where policy is uncertain.

**Confidence-Based Data Collection:**
```python
confidence_score = 1 / (1 + influence_std)  # Low std → high confidence

for timestep in rollout:
    if confidence_score[timestep] < threshold:
        flag_for_human_demonstration(timestep)
        # Collect demo showing clear, unambiguous behavior for this state
```

**Benefits:**
- Focus human effort on states where policy is most uncertain
- Reduce wasted data collection on states policy already understands
- Systematically eliminate high-variance failure modes

**Visualization Opportunity:**
Add "uncertainty heatmap" to app showing which rollout timesteps have high influence variance.

---

### 5. **Real-Time Interpretability Dashboard**

**Feature: Highlight Uncertain Transitions**

Add to the visualizer:
```python
def flag_uncertain_transitions(rollout_idx, threshold=2.0):
    """Highlight timesteps where policy has high influence variance"""
    influence_stds = compute_per_timestep_std(rollout_idx)
    
    for timestep, std in enumerate(influence_stds):
        if std > threshold:
            mark_as_uncertain(timestep)  # Red border on frame
            
            # Show which demos are creating conflicting signals
            conflicting_demos = identify_opposing_influences(timestep)
            display_message(f"⚠️ Uncertain! Demos {conflicting_demos} provide conflicting guidance")
```

**User Experience:**
- Operator sees frames with red borders indicating policy uncertainty
- Can immediately investigate which demonstrations are causing confusion
- Enables real-time debugging of policy behavior

---

### 6. **Automated Data Curation Pipeline**

**End-to-End Workflow:**

```python
# Step 1: Identify problematic patterns
failed_high_std_rollouts = get_failed_rollouts_with_high_std()

# Step 2: Trace to responsible demos
for rollout in failed_high_std_rollouts:
    problematic_demos = get_high_variance_demos(rollout)
    
    # Step 3: Analyze conflict patterns
    for demo1, demo2 in find_conflicting_pairs(problematic_demos):
        if actions_differ_significantly(demo1, demo2):
            flag_for_review(
                demo1, demo2,
                reason=f"Conflicting strategies in similar states"
            )

# Step 4: Generate curation report
report = {
    "remove": demos_with_consistent_conflicts,
    "relabel": demos_needing_context_clarification,
    "add_more": states_with_high_uncertainty,
}

# Step 5: Retrain and validate
curated_dataset = apply_curation(report)
new_policy = retrain(curated_dataset)
assert correlation(success, std) < -0.5  # Should improve
```

**Metrics to Track:**
- Does removing high-variance demos improve success rate?
- Does the success-std correlation become more negative (stronger)?
- Do failed rollouts have lower std after curation?

---

### 7. **Theoretical Insight: "Influence Entropy" as a Training Signal**

**Observation:** Low influence std correlates with success.

**Architectural Implications:**

#### Regularization Approach:
```python
# During training, penalize high influence variance
influence_variance_loss = compute_influence_std_estimate()
total_loss = task_loss + λ * influence_variance_loss
```

**Goal:** Encourage policy to develop consistent influence patterns, which correlate with better performance.

#### Attention Mechanism Design:
```python
# Design attention that explicitly minimizes variance
attention_weights = softmax(scores)
variance_penalty = variance(attention_weights)
# Train to minimize variance_penalty
```

**Hypothesis:** Policies that learn low-variance influence patterns generalize better because they've developed coherent internal representations.

---

### 8. **Demo Ranking by "Teaching Clarity"**

**New Metric: Clarity Score**
```python
clarity_score = (
    mean_influence_on_successes 
    - influence_std_on_successes 
    + (1 / (1 + influence_std_on_failures))
)
```

**Interpretation:**
- High mean on successes → demo helps
- Low std on successes → demo helps consistently
- Low std on failures → when policy fails, it's not because of this demo's confusion

**Use Case:** Rank demonstrations by clarity score and train on top K% for fastest learning.

---

### 9. **Export Curated Dataset Feature**

**App Enhancement:** Add button to "Export Cleaned Dataset"

**Criteria:**
```python
keep_demo = (
    demo.mean_influence > threshold_mean
    and demo.avg_std_on_failures < threshold_confusion
    and demo.creates_low_variance_on_successes
)
```

**Output:** New HDF5 file with filtered demonstrations optimized for training.

---

## Immediate Next Steps for the Visualizer

**High-Priority Features to Add:**

1. **"Failure Analysis" Tab**
   - Automatically identify high-std failed rollouts
   - Show demo pairs causing conflicts
   - Generate actionable curation recommendations

2. **Color-Code Demos by "Confusion Factor"**
   - Demos creating high variance on failures → Red
   - Demos creating low variance on successes → Green
   - Helps quickly identify problematic demonstrations

3. **Demo Ranking by Teaching Clarity**
   - Sort demonstrations by clarity score
   - Filter out confusing demos
   - Export cleaned dataset

4. **Uncertainty Timeline**
   - For each rollout, show timestep-by-timestep influence std
   - Mark high-uncertainty moments for investigation
   - Link to conflicting demonstrations

5. **Curation Report Generator**
   - Automated analysis of which demos to remove/keep
   - Statistics on expected improvement
   - Export recommendations as JSON/CSV

---

## Future Analysis Directions

1. **Cluster Analysis:** Explicitly cluster demonstrations based on (Mean, Std) patterns to identify distinct demo "types"

2. **Temporal Patterns:** Analyze which timesteps within high-variance demos matter most

3. **Quality Re-annotation:** Consider re-labeling demos based on influence patterns rather than subjective trajectory quality

4. **Demo Pruning Strategy:** Remove low-mean, low-variance demos (likely unhelpful noise) while keeping high-mean or high-variance structured demos

5. **Cross-Rollout Consistency:** Investigate which demos show consistent influence across successful vs. failed rollouts

---

## Summary

The transition-level statistics reveal that:
- **Mean influence** is the best predictor of demonstration usefulness
- **Variance patterns** distinguish structured (context-dependent) from noisy (flat) influence
- **Human quality labels** capture trajectory clarity, not influence effectiveness
- **Influence structuredness** is an intrinsic, stable property of demonstrations
- The automated **performance influence metric outperforms human judgment** for demo selection
