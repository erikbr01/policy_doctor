# Interpreting Influence Heatmaps

Trajectory-level influence heatmaps visualize how specific timesteps in a **training demonstration** affect specific decision points in a **policy rollout**.

## Context: Robot Lift Task

In your specific **Lift** task, where episodes end immediately upon the block being lifted:
- **Major Sub-tasks**: Approach $\rightarrow$ Grasp $\rightarrow$ Initial Lift.
- **Termination**: The end of the rollout is a very "successful" state (block in gripper).
- **Data Quality**: "Bad/Ok" demos contain low-quality segments (stuttering approaches, missed grasps).

---

## Patterns and Interpretation

### 1. The Temporal Diagonal (Standard Imitation)
A strong green diagonal (top-left to bottom-right) indicates that the policy is successfully "tracking" the demonstration.
- **Meaning**: Demo step $t$ influenced rollout step $t$. This is the ideal behavior for simple imitation.

### 2. Horizontal/Vertical Stripes (Persistent Features)
- **Horizontal Stripe**: A single rollout step is influenced by the *entire* demonstration.
    - **Meaning**: The policy is at a "bottleneck" where it relies on global context rather than local progress. This often happens at the very start of a task.
- **Vertical Stripe**: A single demonstration step influences the *entire* rollout.
    - **Meaning**: That demonstration step contains a unique state or cue (e.g., a specific object position) that the policy uses as a reference throughout the rollout.

---

## The "Influence Reversal" Pattern

In some complex heatmaps, you may see a reversal where early demo steps (which were initially positive) become **strongly negative** at the end of a rollout.

![Influence Reversal Example](file:///Users/erik/.gemini/antigravity/brain/1d01203d-2254-4537-aa8e-24065bbd3050/uploaded_image_1769135215317.png)

### Why does this happen?

This pattern usually indicates **State Aliasing** combined with **Behavioral Conflict**.

1.  **State Aliasing (Perceptual Similarity)**: The robot's state at the *end* of the rollout (e.g., task nearly finished) looks perceptually similar to the state at the *beginning* of the demonstration (e.g., just starting).
    - *Example*: In a pick-and-place task, the gripper being "empty over the bin" at the end looks like "empty over the object" at the start.
2.  **Behavioral Conflict**: At the beginning of the demo, the correct action is to **move towards the object**. At the end of the rollout, the correct action is to **stay still or reset**.
3.  **Negative Influence**: Because the policy "remembers" the early demo actions but knows they are *wrong* for the current (late) goal, those early samples are assigned **negative influence**.

> [!NOTE]
> **Lift Example**: In a "Bad/Ok" demo, the early approach phase might be hesitant or misaligned. By the time the rollout reaches the final "Lift" phase, the policy correctly "repels" that bad approach data to ensure it stays focused on the successful pick it just achieved.
    - **Interpretation**: "The policy is actively trying *not* to do what it did at the start of Demo 88, even though the current scene looks somewhat similar."

### Troubleshooting Reversals
If negative influence is causing the policy to fail:
- **Improve State Representation**: Add features that disambiguate start vs. end (e.g., high-level progress indicators or longer history).
- **Data Augmentation**: Add more variety to the "end-of-task" states to differentiate them from "start-of-task" states.
- **Goal Conditioning**: Ensure the policy has a clear, non-aliased goal signal.

### 4. The Temporal Phase Shift (Temporal Lag)
You may see a demonstration segment transition from **strong negative influence** to **strong positive influence** as the rollout progresses.

![Temporal Phase Shift Example](file:///Users/erik/.gemini/antigravity/brain/1d01203d-2254-4537-aa8e-24065bbd3050/uploaded_image_1769135597004.png)

#### Interpretation:
This indicates a **Phase Inconsistency** or **Temporal Lag** between the demo and the rollout.

1.  **Red Phase (Premature Aliasing)**: At a certain rollout step (e.g., $t=4$), the robot's physical state *looks* like the demo state at $t=40$ (aliasing), but the robot hasn't actually finished its current sub-task. Taking the demo's action would be premature or incorrect. The negative influence confirms the policy is correctly resisting this "false cue."
2.  **Green Phase (Catching Up)**: A few steps later in the rollout (e.g., $t=6$), the robot finally reaches the stage where the actions from demo $t=40$ are actually correct. The influence flips to positive.

#### Troubleshooting:
- If the rollout eventually succeeds, this pattern is benign—it shows the robot is "out of sync" with that specific demo trajectory but managing to recover.
- If the rollout fails or stalls, it suggests the **aliasing is too strong**. The policy is so distracted by the "future" demo state that it struggles to execute the immediate "current" actions required to bridge the gap.

### 5. The Sub-task Boundary Conflict (Vertical Barrier)
You may see a sharp vertical "pillar" of negative influence at a specific demonstration step (e.g., $t=60$), followed immediately by a sharp "spike" of positive influence (e.g., $t=65$).

![Sub-task Boundary Conflict Example](file:///Users/erik/.gemini/antigravity/brain/1d01203d-2254-4537-aa8e-24065bbd3050/uploaded_image_1769135641865.png)

#### Interpretation:
This indicates a **High-Variance Decision point** or a different strategy in the demonstration.

1.  **Red Pillar (Conflicting Strategy)**: At demo step $t=60$, the human demonstrator did something that the current policy considers "bad" or "incorrect" for the early stages of this rollout. This might be a recovery behavior or a different sequence of sub-tasks.
2.  **Green Spike (The "Key" Step)**: Immediately after that barrier, the demonstration performs a specific action (at $t=65$) that is exactly what the rollout needs to progress. 

#### Why the contrast?
This pattern often appears when a demonstration is "noisy" or has a unique way of handling a transition. The negative influence acts as a **"Don't do what happened here"** signal, which helps the policy isolate the actually useful movement that follows immediately after. 

> [!NOTE]
> **Lift Example**: Step 60 might be a demo's awkward gripper adjustment that the policy dislikes. Step 65 is the final, decisive "close gripper" command. The policy uses the "red pillar" at 60 to clearly define the boundary where the "messy approach" ends and the "clean pick" begins.

#### Troubleshooting:
- If you see many of these "red pillars," it might mean your dataset contains **multimodal strategies** (different ways of solving the same task).
- If the policy is stuttering at these points, it's struggling to reconcile the "repulsion" from step 60 with the "attraction" to step 65. Adding more goal-oriented data can help smooth these transitions.
