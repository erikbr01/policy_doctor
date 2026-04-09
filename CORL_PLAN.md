# Road to CoRL

## Project objective

To demonstrate the following contributions:

1. Behavior discovery by clustering influence embeddings/explanations
2. Connecting behaviors with temporal context and semantic descriptions in a human-interpretable behavior graph
3. Guiding data collection (via targeted MimicGen data generation in simulation, via human data collector in real-world)

We are looking to show these contributions on a few tasks from Robomimic/MimicGen/Robocasa365 in simulation and in real-world experiments.
For policy learning, we will use single-task diffusion policies. The objective is not to show these results at scale (yet), but rather working for single-task setups.
The data scale should remain reasonably small to enable efficient experimentation.

## Hypotheses and experiments

We have the following questions/hypotheses we aim to answer:

Contribution 1 (behavior discovery):

- Do the automatically generated behavior annotations align with human intuition?
- Are the generated behavior clusters coherent?
- How do the answers to the above two questions vary per task?
- How does the choice of hyperparameters impact the results?

Contribution 2 (behavior graphs):

- Which properties do the graphs have (e.g. Markovian)?

Contribution 3 (insights for data collection):

- What are good heuristics to guide data collection based on the behavior graphs?
- Can we outperform current data collection methods?


## Evaluation setups

To answer the questions, we need to set up experiments accordingly.

In particular, here are drafts for three experiments:

### Evaluating cluster coherency

We generate clusters, use a VLM to annotate the behavior in the individual slices that make up the cluster. Then, we use a VLM to evaluate the coherency of the descriptions.

We can use different VLMs to do so: Qwen3-VL, Cosmos-Reason2, Gemini-ER, molmo2. It seems we need multiple camera views and high-resolution images to get good slice descriptions.

Another plan that is more expensive: pairwise comparison, given two sequences of images, the VLM should annotate the differences. However, this gets more expensive for pairwise differences for all slices assigned to a cluster.

Once we have this, sweeping over hyperparameters and tasks should be easy to do.

### Behavior graph properties

I want to know if the graphs are Markovian. We should verify the Markov assumption for the graphs.

### Data collection

For real-world experiments, the setup is a bit more straightforward. For MimicGen, there is a bit more glue needed. MimicGen by itself uses a different version of Robosuite. There is also a Robocasa365 branch of MimicGen that uses a newer version of Robosuite. However, it's unclear how well that supports the original MimicGen tasks. So we have two objectives for the diffusion policy/CUPID repository: a) add compatibility with the previous robosuite/robomimic setup from mimicgen to use that data with diffusion policy for policy learning and evaluation b) add compatibility with the robocasa robosuite setup.

On the side of attribution, we also want adapters for both mimicgen datasets and robocasa datasets. These correspond to different versions of robosuite, and possibly have different data formats. These adapters should co-exist with the current robomimic adapters from the original project. The objective: have different data adapters, without needing different policy adapters. We just do diffusion policy.





