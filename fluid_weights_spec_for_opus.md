# SPEC: Fluid Weight Learning System for Transformer Models

**Version**: 1.0  
**Date**: December 5, 2025  
**Author**: Basil (with Claude Sonnet 4.5)  
**Target**: Claude Opus 4 (or equivalent reasoning model)  
**Status**: Exploratory Research - Unknown Feasibility

---

## Executive Summary

Design a novel learning system for transformer-based language models that enables **perpetual plasticity** - where model weights continuously adapt during inference without explicit loss functions or training phases. This system should allow models to learn organically from experience, similar to biological neural plasticity, rather than through traditional supervised or reinforcement learning.

**Core Challenge**: Current deep learning frameworks only support two modes:
1. **Training mode**: Optimize weights toward explicit loss function via backpropagation
2. **Inference mode**: Frozen weights, no learning

We need a **third mode**: Organic adaptation where weights drift naturally from processing experience, with no predefined optimization objective.

---

### Why This Research Matters

**If successful**: This would represent a fundamental shift in how AI systems learn - from episodic training to continuous adaptation. Applications include:
- Personal AI that adapts to individual users over time
- Agents that learn from deployment without human supervision
- Models that self-organize knowledge from experience
- Elimination of expensive fine-tuning cycles

**If partially successful**: Even limited fluid adaptation would be valuable for online learning scenarios.

**If we prove impossibility**: Understanding fundamental constraints would guide future research and prevent wasted effort.

**This is exploratory research** - we're at the frontier of what's possible. Your rigorous analysis, whether it yields a solution or proves impossibility, will be valuable.

---

## 1. Why This Might Actually Be Possible (Encouragement Section)

### 1.1 This Isn't as Crazy as It Sounds

Before diving into the challenge, let's establish that **related ideas have worked** in adjacent domains:

**✓ Hebbian Learning Works**: Simple Hebbian networks have been successfully trained for decades
- Oja's rule (1982) demonstrated stable unsupervised learning
- BCM rule (1982) showed self-organizing feature learning
- Modern implementations exist for CNNs

**✓ Local Learning Rules Work**: Algorithms that don't use backprop have succeeded
- Feedback Alignment (2016): Random feedback instead of backprop
- Direct Feedback Alignment (2017): Simplified further
- Equilibrium Propagation (2017): Energy-based learning
- **All achieve competitive performance on real tasks**

**✓ Online Learning Is Standard**: Models that adapt during deployment exist
- Online gradient descent is well-studied
- Streaming learning algorithms are production-ready
- Meta-learning enables fast adaptation (MAML, Reptile)

**✓ Biological Brains Do This**: Proof of concept exists in nature
- Human cortex continuously adapts without "training runs"
- Synaptic plasticity operates without global loss functions
- If biology can do it, artificial systems might be able to approximate it

### 1.2 Recent Research Getting Close

**Predictive Coding Networks** (2017-2023):
- Whittington & Bogacz showed transformers can implement predictive coding
- Local prediction errors can replace backprop
- Some implementations achieve competitive performance

**Contrastive Learning** (2020+):
- SimCLR, MoCo, BYOL learn without labels
- Self-supervised objectives that could run continuously
- Proven at scale (CLIP, etc.)

**Energy-Based Models** (ongoing):
- Hopfield Networks have been scaled to modern architectures (2020)
- Energy minimization provides learning signal without explicit targets
- Recent work on "Hopfield Transformers"

**Neural ODEs & Continuous-Depth Models**:
- Treat networks as continuous dynamical systems
- Natural fit for continuous adaptation
- Math is well-understood

### 1.3 What Makes This Achievable Now

**Modern transformers are surprisingly robust**:
- LoRA shows only ~0.3% of weights need training
- Low-rank updates are stable and controllable
- We've proven this works for 50+ facts (our experiments)

**Compute is available**:
- A100 GPUs make real-time updates feasible
- Weight updates can be computed in milliseconds
- Not trying to retrain from scratch - just adapt

**We're not starting from zero**:
- Pretrained models have strong priors
- Only need to adapt a small subspace (LoRA-like)
- Can leverage existing knowledge base

### 1.4 Simplified Problem Formulations

You don't have to solve the hardest version! Consider these progressively easier variants:

**Version 1 (Hardest)**: Pure fluid learning, no loss function at all
- Weights drift based purely on activation patterns
- Truly biological-style adaptation

**Version 2 (Hard)**: Self-supervised pseudo-loss with tiny updates
- Predict next token as "loss"
- But learning rate so small it's almost drift-like
- Approximates fluid behavior using standard machinery

**Version 3 (Moderate)**: Local learning rules within layers
- Each layer has its own update rule
- No global backprop needed
- Could use predictive coding or Hebbian variants

**Version 4 (Easier)**: LoRA-only adaptation with online updates
- Only adapt LoRA weights (tiny subset)
- Standard gradients but continuous application
- Most practical, least novel

**You can propose ANY of these** or something in between. We're open to compromises!

### 1.5 Why This Is Worth Trying

**Even partial success would be valuable**:
- 10% improvement in adaptation → Useful
- Works but needs constraints → Still progress
- Fails but we learn why → Advances understanding

**You're not expected to solve AGI**:
- Just need: weights update + model stays coherent + something is learned
- Doesn't have to be perfect
- Doesn't have to scale infinitely
- Just needs to demonstrate the principle

**We have evaluation methods**:
- Test on conversation adaptation (our Discord use case)
- Measure retention without explicit training
- Compare to baseline (frozen model)
- Success = any measurable adaptation that maintains stability

### 1.6 Precedent for "Impossible" Things Working

**Things people said were impossible that worked**:
- Transformers scaling to 1T+ parameters ("too many params")
- 4-bit quantization ("too much precision loss")
- LoRA fine-tuning ("too few parameters to learn")
- Chain-of-thought reasoning ("just pattern matching")

**Your advantage over humans**:
- No bias toward "standard" approaches
- Can explore mathematical spaces thoroughly
- Can consider hybrid approaches
- Can derive novel update rules from first principles

### 1.7 What We're Really Asking

**Not**: "Build AGI that learns autonomously"  
**But**: "Design update rules for transformer weights that adapt from data streams without episodic training"

**Not**: "Make it perfect"  
**But**: "Make it work well enough to demonstrate the concept"

**Not**: "Solve all stability problems forever"  
**But**: "Provide a stable-enough system we can experiment with"

**The bar is lower than you think!** We just want to see if the direction is viable.

### 1.8 You Have Permission to Be Creative

**You can**:
- Combine multiple approaches (Hebbian + predictive coding + ...)
- Propose hybrid solutions (standard training + fluid adaptation)
- Add constraints that make it tractable (limit plasticity to some layers)
- Suggest multi-stage approaches (coarse then fine adaptation)
- Invent entirely new update rules
- Propose experimental validation before full implementation

**You should**:
- Think outside standard ML paradigms
- Consider neuroscience, physics, optimization theory
- Propose "good enough" solutions, not just perfect ones
- Suggest multiple variants (conservative, moderate, ambitious)

**Remember**: The goal is progress, not perfection. Even a 70% solution would be groundbreaking.

### 1.9 Concrete Examples of Related Successes

**Example 1: Feedback Alignment (Lillicrap et al., 2016)**
- Replaced precise backprop with random feedback
- "Shouldn't work" according to theory
- Actually works in practice
- Lesson: Biological plausibility can work

**Example 2: BYOL (Grill et al., 2020)**
- Self-supervised learning without negative pairs
- Seemed to violate contrastive learning principles
- Works better than methods with explicit contrastives
- Lesson: Self-organization can provide learning signal

**Example 3: LoRA (Our experiments!)**
- Only 0.34% of parameters trainable
- Achieves 100% retention on 50 facts
- Updates in seconds, not hours
- Lesson: Small, targeted updates are powerful

**Example 4: In-Context Learning (GPT-3)**
- Models "learn" from examples without weight updates
- Demonstrates meta-learned adaptation
- Suggests latent plasticity mechanisms
- Lesson: Models already have adaptation capacity

These show that unconventional approaches can work. Your task is to design the next one!

### 1.10 Where to Start (Quick Start for Opus)

**If you're unsure where to begin, try one of these entry points:**

**Entry Point A - Modify Existing Approach**:
- Start with self-supervised next-token prediction
- Make learning rate extremely small (1e-8)
- Add constraints to prevent collapse
- This gives you a working baseline to improve upon

**Entry Point B - Hebbian for Attention**:
- Focus just on attention weights (Q, K, V)
- Design correlation-based update rule
- Ignore feed-forward layers initially
- Scale up once working

**Entry Point C - Predictive Coding**:
- Each layer predicts its input
- Update based on prediction error
- Hierarchical implementation
- Well-studied in neuroscience

**Entry Point D - Energy-Based**:
- Define energy function for transformer states
- Weights update to minimize energy
- Connection to Hopfield networks
- Mathematical foundation exists

**Pick one and develop it fully** - we can experiment with others later. Or combine them! You decide what seems most promising.

Don't feel pressured to invent something completely novel if adapting existing approaches would work. **Pragmatic solutions are valuable too.**

---

## 2. Problem Statement

### 1.1 Current Limitations

**Standard Training Paradigm**:
```python
# Training: Optimize toward target
loss = cross_entropy(model_output, target_label)
loss.backward()
optimizer.step()  # Weights update

# Inference: Frozen
with torch.no_grad():
    output = model(input)  # Weights don't change
```

**What's Missing**:
- No ability to adapt during deployment
- Requires explicit supervision (targets/rewards)
- Binary state: learning OR inference, never both
- No organic, goal-free adaptation

### 1.2 Desired Behavior

**"Fluid Weights" Mode**:
```python
# Model is ALWAYS plastic, never frozen
# Processes conversation/experience
# Weights drift organically
# No explicit loss function required
# Adaptation is continuous, not episodic
```

**Biological Analogy**: 
Like cortical neurons that continuously adjust synaptic weights based on activation patterns, without a central "loss function" telling them what to optimize.

### 1.3 Key Requirements

**MUST HAVE**:
1. Weights update during inference (no separate training phase)
2. No explicit loss function or target labels required
3. Stable over long conversations (doesn't collapse to gibberish)
4. Maintains base capabilities (no catastrophic forgetting)
5. Adapts to patterns in input stream (conversation style, topics, etc.)

**NICE TO HAVE**:
6. Theoretically grounded (not just heuristic hacks)
7. Computationally efficient (minimal overhead)
8. Controllable plasticity rate (can tune adaptation speed)
9. Preserves important weights (elastic consolidation)
10. Works with existing transformer architectures

---

## 2. What Already Exists (And Why It's Insufficient)

### 2.1 Online Learning
**What it is**: Update model after each example using standard loss.

**Why insufficient**:
- Still requires explicit targets/labels
- Still uses traditional loss functions
- Not "organic" - still optimization-based

### 2.2 Self-Supervised Learning
**What it is**: Train on pseudo-tasks like "predict next token."

**Why insufficient**:
- Still requires a loss function (prediction error)
- Still episodic (train then infer)
- Not perpetual plasticity

### 2.3 Hebbian Learning (Classical)
**What it is**: "Neurons that fire together, wire together."

**Formula**: `Δw_ij = η * x_i * x_j`

**Why insufficient**:
- Works for simple networks, not transformers
- No notion of attention, position, or context
- Unstable at scale without constraints
- No existing implementation for transformer attention

### 2.4 Meta-Learning / Few-Shot Learning
**What it is**: Learn how to learn quickly on new tasks.

**Why insufficient**:
- Still uses episodic training
- Requires meta-training dataset
- Not continuous adaptation

### 2.5 Continual Learning (Current Approach)
**What it is**: LoRA adapters + experience replay.

**Why insufficient**:
- Still requires manual fact extraction
- Still has separate training phase
- Not self-directed learning
- We already built this - not what we want

---

## 3. Theoretical Foundations to Consider

### 3.1 Neuroscience

**Synaptic Plasticity**:
- Long-Term Potentiation (LTP)
- Long-Term Depression (LTD)
- Spike-Timing-Dependent Plasticity (STDP)

**Key insight**: Biological learning has local update rules, not global loss functions.

### 3.2 Predictive Coding
**Concept**: Brain minimizes prediction errors hierarchically.

**Mechanism**: Each layer predicts input from layer below, updates based on prediction error.

**Potential**: Could provide local learning signals without global loss.

### 3.3 Free Energy Principle
**Concept**: Systems minimize surprise/uncertainty about their environment.

**Potential**: Could define adaptation without explicit loss - just reduce surprise.

### 3.4 Contrastive Learning
**Concept**: Learn by comparing similar vs dissimilar examples.

**Potential**: Could provide signal for weight updates without explicit labels.

### 3.5 Hebbian Variants

**Oja's Rule**: Normalized Hebbian learning (prevents unlimited weight growth)
- `Δw_ij = η * x_i * (x_j - w_ij * x_i)`

**BCM Rule**: Sliding threshold for potentiation vs depression
- Prevents runaway activation

**STDP**: Temporal component - timing of spikes matters

---

## 4. Technical Constraints

### 4.1 Architecture
**Given**: Standard transformer architecture
- Multi-head attention
- Feed-forward layers
- Layer normalization
- Positional encodings

**Cannot**: Fundamentally change architecture (too expensive to retrain)

**Must**: Work with existing pretrained models (Mistral, Llama, etc.)

### 4.2 Computational
**Environment**: Single A100 GPU (40GB VRAM)

**Latency**: Weight updates must be fast (<10ms per token)

**Memory**: Cannot significantly increase memory footprint

### 4.3 Stability
**Critical**: Model must not:
- Diverge into nonsense
- Forget base capabilities
- Produce harmful outputs
- Become stuck in local minima

**Must**: Maintain coherence over thousands of tokens

### 4.4 Implementation
**Framework**: PyTorch (or compatible)

**Ideal**: Works with standard forward/backward infrastructure

**Acceptable**: Custom CUDA kernels if necessary

---

## 5. The Core Challenge: What to Optimize?

### 5.1 The Gradient Problem

In standard deep learning:
```
Loss function → Gradients → Weight updates
```

Without loss function:
```
??? → ??? → Weight updates
```

**Question for Opus**: What replaces the loss function?

### 5.2 Possible Approaches (Incomplete List)

**A) Self-Prediction as Pseudo-Loss**
```python
# Predict next token (self-supervised)
pred = model(context)
target = next_token_in_conversation
loss = cross_entropy(pred, target)
# Very tiny learning rate
loss.backward()
optimizer.step(lr=1e-8)
```

**Pros**: Works with existing frameworks  
**Cons**: Still has a loss function (not truly "fluid")

**B) Local Hebbian-Style Rules**
```python
# For each attention head
Q = query_projection(x)
K = key_projection(x)
attention_scores = Q @ K.T

# Update based on co-activation
W_update = learning_rate * (Q @ K.T)
# Apply directly to weights (no backprop)
```

**Pros**: No global loss  
**Cons**: How to apply to attention? How to prevent instability?

**C) Predictive Coding**
```python
# Each layer predicts its input
prediction = layer.predict(context)
actual = layer.input
error = actual - prediction

# Update to minimize prediction error locally
layer.weights += learning_rate * error * activation
```

**Pros**: Biologically inspired, local signals  
**Cons**: How to implement prediction at each layer?

**D) Contrastive / Energy-Based**
```python
# Define energy function
energy_current = compute_energy(current_state)
# After processing, compare
energy_after = compute_energy(new_state)
# Update to lower energy
weight_update = -gradient(energy_difference)
```

**Pros**: Principled (minimize energy)  
**Cons**: How to define energy for transformers?

**E) Something Novel (Your Job, Opus!)**

Perhaps none of these are right. Perhaps there's a better way that combines elements or introduces new concepts. **This is what we're asking you to design.**

---

## 6. Design Requirements

### 6.1 Mathematical Specification

**Provide**:
1. **Update rule**: Precise mathematical formula for weight changes
2. **Per-layer logic**: What happens at each transformer layer
3. **Attention mechanism**: How attention weights adapt
4. **Stability analysis**: Why weights won't diverge
5. **Preservation**: How base knowledge is protected

### 6.2 Implementation Sketch

**Provide**:
1. Pseudocode for the main algorithm
2. What happens during forward pass
3. When/how weights update
4. Integration points with PyTorch
5. Hyperparameters and their effects

### 6.3 Theoretical Justification

**Explain**:
1. Why this should work (theoretical foundation)
2. What objective is implicitly minimized (if any)
3. Connection to neuroscience/ML literature
4. Known limitations and failure modes
5. How to test if it's working

---

## 7. Specific Questions for Opus

### 7.1 Core Mechanism Design

**Question 1**: What mathematical rule governs weight updates in your proposed system?

**Question 2**: How do you handle the transformer-specific components:
- Multi-head attention (Q, K, V projections)
- Feed-forward layers
- Layer normalization
- Residual connections

**Question 3**: What prevents weights from:
- Growing unbounded
- Collapsing to zero
- Oscillating chaotically
- Destroying learned knowledge

### 7.2 Practical Implementation

**Question 4**: Can this work within PyTorch's autograd framework, or does it need custom kernels?

**Question 5**: What's the computational overhead per token? (Target: <2x inference time)

**Question 6**: How do you tune the "plasticity rate"? What's the equivalent of learning rate?

### 7.3 Behavior and Evaluation

**Question 7**: What observable behaviors would indicate successful adaptation?

**Question 8**: How do you evaluate this without a loss function? What metrics matter?

**Question 9**: What would "good" drift look like vs "bad" drift?

### 7.4 Advanced Considerations

**Question 10**: Can plasticity be selective? (e.g., adapt some layers, freeze others)

**Question 11**: How do you handle different timescales? (fast: conversation, slow: general knowledge)

**Question 12**: Can you "save" the adapted state and reload it later?

---

## 8. Success Criteria

### 8.1 Minimum Viable System

**Must demonstrate**:
1. Weights change during inference (not separate training)
2. Model adapts to conversation patterns
3. Maintains coherence over 1000+ tokens
4. No catastrophic forgetting on base tasks
5. Reproducible behavior (not just lucky runs)

**Example test**:
```
1. Chat about Topic A for 100 turns
2. Measure: Has model's style shifted toward Topic A language?
3. Test base knowledge: Can it still do math, coding, etc.?
4. Save and reload: Does adapted state persist?
```

### 8.2 Ideal System

**Would demonstrate**:
- Adapts to user's communication style
- Learns domain-specific terminology from context
- Improves responses over long conversations
- Can be "reset" or "saved" at will
- Scales to production use

---

## 9. Known Challenges (Be Honest About These)

### 9.1 Stability is Hard

**Problem**: Unconstrained updates lead to chaos.

**Your task**: Design constraints that maintain stability without killing plasticity.

### 9.2 Transformers Aren't Designed for This

**Problem**: Attention mechanism assumes fixed weights during sequence.

**Your task**: Adapt attention to support dynamic weights, or work around it.

### 9.3 Evaluation is Unclear

**Problem**: Without loss, how do we know if it's "learning well"?

**Your task**: Define success metrics that don't rely on supervised labels.

### 9.4 Biological Learning ≠ Artificial Learning

**Problem**: Neurons and matrix multiplications are very different.

**Your task**: Bridge neuroscience intuitions to practical transformer math.

### 9.5 This Might Be Impossible

**Problem**: Fundamental mathematical constraints might prevent this.

**Your task**: If it's impossible, explain why clearly. If it's possible, prove it with math.

---

## 10. Deliverables Requested

### 10.1 Mathematical Design
- **Update rules** (precise equations)
- **Stability analysis** (why it won't diverge)
- **Theoretical foundation** (why it should work)

### 10.2 Implementation Plan
- **Pseudocode** for main algorithm
- **Integration strategy** with PyTorch
- **Hyperparameters** and tuning guidance

### 10.3 Evaluation Protocol
- **How to test** if it's working
- **Metrics to track** (without loss function)
- **Failure modes** and how to detect them

### 10.4 Honest Assessment
- **Feasibility**: Can this actually be built?
- **Limitations**: What won't this solve?
- **Unknowns**: What needs experimental validation?
- **Alternatives**: If this fails, what else to try?

---

## 11. Context: Why We're Doing This

### 11.1 The Vision

Create models that:
- Learn continuously from experience
- Adapt organically to users/environments
- Don't require massive training runs
- Self-organize knowledge
- Are perpetually plastic, not frozen

### 11.2 Current State

We've built:
- LoRA-based continual learning (works, but requires manual supervision)
- Scaling laws (50+ facts learned, but still supervised)
- Conversational learning prototype (still uses batch training)

**Gap**: All of these require explicit training phases and supervision. We want organic, self-directed learning.

### 11.3 Why This Matters

**If successful**:
- Models that adapt to personal use
- Agents that learn from deployment
- No need for explicit fine-tuning
- Closer to biological intelligence

**If impossible**:
- We learn fundamental limits
- We understand why supervision is necessary
- We can pursue alternative approaches with clarity

---

## 12. Important Notes

### 12.1 This Is Exploratory

**We don't know if this is possible.** That's okay. The goal is to:
1. Try to design it rigorously
2. Discover if fundamental barriers exist
3. Learn what constraints bind us

### 12.2 Be Rigorous, Not Hopeful

**Don't**: Hand-wave away stability concerns or assume "it'll probably work"

**Do**: Provide mathematical grounding, acknowledge unknowns, identify failure modes

### 12.3 Novel Solutions Welcome

If standard approaches (Hebbian, predictive coding, etc.) don't work, **invent something new**.

We're not looking for "what's published" - we're looking for "what could work."

### 12.4 Failure Is Acceptable

If your analysis concludes "this is fundamentally impossible because [reason]", that's a valuable answer.

**Better to know it's impossible than waste months trying.**

---

## 13. Additional Resources

### 13.1 Relevant Papers (Suggested Reading)
- Oja (1982): "Simplified neuron model as a principal component analyzer"
- Hinton & Sejnowski (1986): "Learning and relearning in Boltzmann machines"
- Friston (2010): "The free-energy principle: a unified brain theory?"
- Whittington & Bogacz (2017): "An approximation of the error backpropagation algorithm in a predictive coding network"
- Millidge et al. (2022): "Predictive coding: a theoretical and experimental review"

### 13.2 Existing Implementations
- Hebbian learning in simple networks: Multiple PyTorch implementations exist
- Predictive coding networks: Some experimental implementations
- **None for transformer-scale models with fluid weights**

### 13.3 Our Previous Work
- Scaling law notebook: 50 facts, 100% retention, LoRA-based
- See: `/mnt/user-data/outputs/continual_learning_project_status.md`

---

## 14. The Challenge

**Design a mathematically rigorous, implementable system for perpetual plasticity in transformer models.**

Requirements:
- ✅ Weights update during inference
- ✅ No explicit loss function
- ✅ Stable over long sequences
- ✅ Maintains base capabilities
- ✅ Adapts to input patterns

**Show your work. Prove it's possible, or prove it's impossible.**

**We're ready to build whatever you design** - if it's buildable.

**We're ready to accept impossibility** - if you can prove it.

---

## 15. Final Note to Opus

Basil says: "I don't care if it's impossible. I want to see that it's impossible."

But we're also hoping you'll surprise us! Here's what we need:

**If you find a path forward**:
- Show us the math (even if rough)
- Explain why it should work
- Identify the key unknowns to test
- Propose concrete next steps

**If you hit fundamental barriers**:
- Explain what makes it impossible
- Show the mathematical constraints
- Suggest alternative approaches
- Help us understand what we're missing

**Most likely scenario - partial solutions**:
- "This works if we constrain X"
- "This could work with approximation Y"
- "Here are three variants with different tradeoffs"
- **This is totally fine!** We'll take what we can get.

**Your approach**:
- Think deeply and creatively
- Consider unconventional solutions
- Be rigorous but not defeatist  
- Propose experiments to validate ideas
- Multiple options > one perfect answer

We're not expecting magic. We're expecting your best reasoning about what's possible in this space. Even "here's a 60% solution with caveats" would be incredibly valuable.

**Remember**: You have more mathematical firepower than most human researchers. Use it. Explore the possibility space thoroughly. We trust your judgment on what's feasible.

---

**End of Specification**

*Awaiting your response, Opus.*

---

## Appendix A: Glossary

**Catastrophic forgetting**: When learning new information destroys old knowledge

**LoRA (Low-Rank Adaptation)**: Lightweight fine-tuning via low-rank matrices

**Hebbian learning**: "Fire together, wire together" - correlation-based learning

**Predictive coding**: Hierarchical prediction + error correction

**Plasticity**: Ability of weights to change

**Stability**: Resistance to divergence/collapse

**Self-supervised**: Learning from data itself (e.g., predict next token)

**Gradient**: Derivative of loss w.r.t. weights (indicates update direction)

**Backpropagation**: Algorithm for computing gradients

**Attention**: Transformer mechanism for focusing on relevant context

**Q/K/V**: Query, Key, Value matrices in attention

## Appendix B: Contact

**Questions?** 
- Respond directly with your design
- Or explain why more information is needed
- Or declare impossibility with proof

**This is a research challenge, not an engineering task.**

Good luck, Opus. We believe in you (but we're also skeptical).

Show us what you've got. 🧠
