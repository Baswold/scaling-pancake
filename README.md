# Fluid Weights: Theoretical Foundations

## 1. The Core Innovation

Traditional deep learning has a fundamental dichotomy:
- **Training**: Weights change via backpropagation toward a loss function
- **Inference**: Weights are frozen

We propose a **third mode**: **Fluid plasticity** where weights drift continuously based on activation patterns, without explicit optimization objectives.

---

## 2. Mathematical Framework

### 2.1 LoRA Formulation

For a linear layer with weight matrix `W ∈ ℝ^{d_in × d_out}`, LoRA adds low-rank adaptation:

```
y = x @ W + α * (x @ A @ B)
```

Where:
- `A ∈ ℝ^{d_in × r}` (down projection)
- `B ∈ ℝ^{r × d_out}` (up projection)
- `r << min(d_in, d_out)` (low rank, typically 8-64)
- `α` is a scaling factor

The intermediate representation `h = x @ A` lives in a compressed `r`-dimensional space.

### 2.2 The Update Rule Problem

In standard training:
```
∇_A L, ∇_B L = backprop(loss)
A ← A - η * ∇_A L
B ← B - η * ∇_B L
```

Without a loss function, we need alternative update rules. We propose **local learning rules** based on activation patterns.

---

## 3. Update Rules

### 3.1 Hebbian Learning (Baseline)

**Principle**: "Neurons that fire together, wire together"

```
ΔA = η * E[x^T @ h]
ΔB = η * E[h^T @ y]
```

Where E[·] denotes expectation over the batch.

**Problem**: Unbounded weight growth. If x and h are always positive, weights grow forever.

### 3.2 Oja's Rule (Self-Normalizing Hebbian)

**Key insight**: Add a normalization term that prevents unbounded growth:

```
ΔA = η * (x^T @ h - ||h||² * A)
ΔB = η * (h^T @ y - ||y||² * B)
```

The second term `||h||² * A` acts as a "forgetting" force that contracts weights when activations are large.

**Properties**:
- Self-normalizing: weights converge to unit norm
- Learns principal components of the input distribution
- Stable: can run indefinitely without divergence

**Derivation**: Oja's rule can be derived as gradient descent on the objective:
```
L = -h^T @ h + λ * ||W||²_F
```
Which maximizes variance of h while constraining weight magnitude.

### 3.3 BCM Rule (Sliding Threshold)

**Principle**: Introduce a dynamic threshold θ that determines potentiation vs depression:

```
ΔW = η * x * h * (h - θ)
θ = E[h²]  (running average)
```

When `h > θ`: Potentiation (strengthen connection)
When `h < θ`: Depression (weaken connection)

**Key property**: Self-stabilizing. High activity raises θ, making future potentiation harder.

For LoRA:
```
ΔA = η * x^T @ (h * (h - θ_h))
ΔB = η * h^T @ (y * (y - θ_y))
```

### 3.4 Predictive Coding

**Principle**: Each layer predicts its input from its output. Prediction error drives learning.

Given:
```
h = x @ A
y = h @ B
```

Define predictions:
```
h_pred = y @ B^T @ (B @ B^T)^{-1}  ≈ y @ B^T / ||B||²
x_pred = h @ A^T @ (A @ A^T)^{-1}  ≈ h @ A^T / ||A||²
```

Prediction errors:
```
ε_h = h - h_pred
ε_x = x - x_pred
```

Update rules:
```
ΔA = η * ε_x^T @ h  (reduce input prediction error)
ΔB = η * ε_h^T @ y  (reduce hidden prediction error)
```

**Intuition**: Weights adjust to make the layer more "predictable" - information flows coherently.

### 3.5 Energy-Based Learning

**Principle**: Define an energy function E(h, W) and update weights to minimize it.

**Energy function** (coherence):
```
E = ||h - μ_h||² / σ_h²
```
Where μ_h, σ_h are running statistics.

**Update rule**:
```
ΔW = -η * ∂E/∂W
```

For coherence energy:
```
∂E/∂A = 2 * (h - μ_h) * x
ΔA = -η * x^T @ (h - μ_h)
```

**Intuition**: Weights adjust to keep activations near their "typical" values.

---

## 4. Novel Mechanisms

### 4.1 Attention-Guided Plasticity (AGP)

**Key insight**: Attention weights encode "importance" - what the model considers relevant.

For attention `A = softmax(QK^T / √d)`:

If position i attends strongly to position j, their representations are related.
We strengthen the weights that produced this attention pattern.

**For Query/Key projections**:
```
h_attended = A @ h  (attention-weighted hidden states)
ΔW_q = η * x^T @ (h_attended - h)
```

This strengthens queries/keys that produce high-attention patterns.

**For Value projections**:
```
y_attended = A @ y  (attention-weighted outputs)
ΔW_v = η * h^T @ y_attended
```

This strengthens values that get selected by attention.

**Why it works**: Attention is self-supervised signal. High attention = useful relationship. Reinforce it.

### 4.2 Temporal Surprise Minimization (TSM)

**Principle**: Track running statistics across sequence positions. "Surprise" = deviation from expected.

```
μ_t = (1-α) * μ_{t-1} + α * h_t  (running mean)
σ_t = running std
surprise = ||h_t - μ_t|| / σ_t
```

**Update**:
```
ΔW = -η * surprise_weight * ∂surprise/∂W
```

Where `surprise_weight = σ(||surprise|| - 1)` (threshold at 1 std).

**Why it works**:
- High surprise = new information = worth learning
- Low surprise = already known = don't overfit
- Creates implicit next-token-like prediction without explicit targets

### 4.3 Contextual Homeostasis (CH)

**Principle**: Biological neurons maintain homeostatic balance. We implement this for transformers.

**Burn-in phase** (first N steps):
```
Collect target_μ, target_σ² from activation statistics
```

**Homeostasis phase** (after burn-in):
```
mean_error = μ_current - μ_target
var_ratio = √(σ²_target / σ²_current)

ΔW = -η * [mean_error * ∂μ/∂W + (var_ratio - 1) * ∂σ²/∂W]
```

**Why it works**:
- Prevents activation collapse (all zeros)
- Prevents activation explosion (unbounded growth)
- Maintains distribution stability over long sequences

---

## 5. Stability Mechanisms

### 5.1 Elastic Weight Consolidation (EWC)

**Problem**: Catastrophic forgetting - new learning destroys old knowledge.

**Solution**: Penalize changes to "important" weights.

```
L_ewc = Σ_i F_i * (θ_i - θ*_i)²
```

Where:
- `F_i` = Fisher information ≈ E[(∂L/∂θ_i)²] (importance)
- `θ*_i` = original weight value

**For fluid weights** (no loss function), we estimate importance from activation magnitudes:
```
F_A = E[h²] (outer product approximation)
```

**Constrained update**:
```
ΔA_constrained = ΔA - λ * F_A * (A - A_original)
```

### 5.2 Spectral Normalization

**Problem**: Weight matrices can amplify inputs unboundedly.

**Solution**: Constrain the spectral norm σ(W) = largest singular value.

```
if σ(W + ΔW) > max_norm:
    ΔW = ΔW * max_norm / σ(W + ΔW)
```

**Efficient computation**: Power iteration (1-3 iterations sufficient).

### 5.3 Gradient Clipping

**Problem**: Individual updates can be too large.

**Solution**: Clip update magnitude.

```
total_norm = ||ΔA||² + ||ΔB||²
if total_norm > max_norm:
    scale = max_norm / total_norm
    ΔA, ΔB = scale * ΔA, scale * ΔB
```

### 5.4 Adaptive Rate Control

**Problem**: Fixed learning rate may be too fast (unstable) or too slow (no learning).

**Solution**: Monitor stability and adjust dynamically.

```
stability_score = mean(||ΔW||) over recent window

if stability_score > target:
    η *= decay_factor
else:
    η *= increase_factor
```

---

## 6. Theoretical Guarantees

### 6.1 Bounded Weight Growth

**Claim**: Under Oja's rule with stability mechanisms, weight norms remain bounded.

**Proof sketch**:
1. Oja's rule: `ΔA = η * (x^T @ h - ||h||² * A)`
2. The term `-||h||² * A` provides contraction when ||A|| is large
3. At equilibrium: `E[x^T @ h] = E[||h||²] * A`
4. This implies ||A|| ≈ E[x^T @ h] / E[||h||²] (bounded)

Spectral normalization provides an additional hard constraint.

### 6.2 Information Preservation

**Claim**: EWC preserves important information from base model.

**Proof sketch**:
1. Fisher information F measures sensitivity: high F = important for predictions
2. EWC adds penalty λ * F * (θ - θ*)² for deviations
3. Important weights (high F) resist change
4. Unimportant weights (low F) can adapt freely

### 6.3 Convergence of Homeostasis

**Claim**: Contextual homeostasis converges to stable activation statistics.

**Proof sketch**:
1. Let μ_target, σ²_target be burn-in statistics
2. Homeostasis update: ΔW ∝ -(μ - μ_target)
3. This is gradient descent on ||μ - μ_target||²
4. With bounded learning rate, converges to μ ≈ μ_target

---

## 7. Comparison to Standard Training

| Aspect | Standard Training | Fluid Weights |
|--------|------------------|---------------|
| Loss function | Required | Not required |
| Backpropagation | Global | Local only |
| Training phase | Separate | Continuous |
| Gradient computation | O(n) backward pass | O(1) local |
| Memory | Store activations | Store statistics |
| Stability | Optimizer handles | Explicit mechanisms |

---

## 8. Limitations and Open Questions

### 8.1 Known Limitations

1. **Learning is subtle**: Much weaker than gradient descent
2. **Hyperparameter sensitivity**: Many parameters to tune
3. **Unclear optimal configuration**: Different tasks may need different settings
4. **Long-term stability untested**: Need millions of tokens to verify

### 8.2 Open Questions

1. **Optimal update rule combination**: What's the best mix?
2. **Theoretical learning capacity**: How much can fluid weights learn?
3. **Relationship to in-context learning**: Are we mimicking ICL in weights?
4. **Scaling properties**: Does this work better or worse at scale?

---

## 9. References

1. Oja, E. (1982). "Simplified neuron model as a principal component analyzer"
2. Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982). "BCM theory"
3. Kirkpatrick, J., et al. (2017). "Overcoming catastrophic forgetting (EWC)"
4. Whittington, J. C., & Bogacz, R. (2017). "Predictive coding networks"
5. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation"
6. Friston, K. (2010). "The free-energy principle: a unified brain theory"

---

## 10. Conclusion

Fluid Weights demonstrates that **perpetual plasticity is possible** with careful design:

1. **Local learning rules** (Oja, BCM, predictive coding) replace backprop
2. **Novel mechanisms** (AGP, TSM, CH) leverage transformer structure
3. **Stability mechanisms** prevent divergence and forgetting

This is exploratory research. The system works in principle, but optimal configurations and long-term behavior require further investigation.

The key insight: **Attention provides free supervision**. What the model attends to tells us what's important. We use this to guide learning without explicit labels.
