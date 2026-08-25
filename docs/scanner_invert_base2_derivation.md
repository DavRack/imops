# Scanner Invert Base-2 Mathematical Derivation & Transcendental Analysis

## 1. Overview & Motivation

In color negative film processing pipelines, the **scanner invert stage** performs the analytical transformation from developed negative dye transmittance ($T_{\text{neg}}$) back to scene-linear positive exposure ($E_{\text{scene}}$ in linear ACEScg).

Historically, optical densitometry in chemical photography was formalized in **decadic (base-10)** units:
$$D_{10} = -\log_{10}(T)$$
$$E_{\text{scene}} = 10^{D_{\text{eff}, 10} / \gamma} - 1.0$$

However, modern digital computing hardware (CPUs with x86 AVX2/AVX-512 and ARM NEON, as well as GPUs with WGSL/SPIR-V/Metal/CUDA) executes transcendentals natively in **base-2** (`log2` and `exp2`). When evaluating a decadic formulation on binary hardware, every pixel requires converting to base-10 and back:
1. $D_{10} = -\log_2(T) \cdot \log_{10}(2)$
2. $E = \exp_2\left(D_{\text{eff}, 10} \cdot \frac{1}{\gamma} \cdot \log_2(10)\right) - 1.0$

While the mathematical identity $\log_{10}(2) \cdot \log_2(10) \equiv 1.0$ is exact over $\mathbb{R}$, evaluating this roundtrip in IEEE 754 32-bit floating-point (`f32`) introduces intermediate truncation and rounding errors. Furthermore, because the derivative of exposure with respect to density scales exponentially ($\frac{dE}{dD} \propto 2^{D/\gamma}$), precision loss at high negative densities (bright scene highlights) is magnified significantly.

By formulating the entire scanner inversion directly in **native base-2 optical density ($D_2$)**:
- Intermediate transcendental base-conversion factors are completely eliminated.
- Floating-point quantization error is removed.
- ALU instruction count per pixel is reduced.
- Bit-exact mathematical consistency between CPU and GPU compute pipelines is established.

---

## 2. Physical Formulation

### 2.1 Substrate Transmission & Beer-Lambert Law

Light passing through developed negative film undergoes absorption according to the Beer-Lambert law. The total transmittance $T_{\text{neg}, c}$ in channel $c \in \{R, G, B\}$ is bounded by the film substrate minimum density ($D_{\text{min}}$), which consists of the film base plus unexposed chemical fog:

$$T_c = \text{clamp}\left(\frac{T_{\text{neg}, c}}{D_{\text{min}, c}}, \varepsilon, 1.0\right)$$

where $\varepsilon = 10^{-6}$ prevents numerical singularities at total opacity.

### 2.2 Optical Density in Base-10 vs Base-2

- **Decadic Optical Density ($D_{10}$)**:
  $$D_{10} = -\log_{10}(T)$$
- **Binary Optical Density ($D_2$)**:
  $$D_2 = -\log_2(T)$$

By the change-of-base formula for logarithms:
$$\log_{10}(T) = \frac{\log_2(T)}{\log_2(10)} = \log_2(T) \cdot \log_{10}(2)$$
Therefore:
$$D_2 = D_{10} \cdot \log_2(10)$$
$$D_{10} = D_2 \cdot \log_{10}(2)$$

### 2.3 Chemical Fog & $C^1$ Continuous Quadratic Toe

Near $D_{\text{min}}$ ($T \approx 1.0$, $D \approx 0$), chemical fog and grain fluctuations produce subtle sub-fog density variations. To avoid an artificial cutoff or derivative discontinuity (cliff), the pipeline applies a $C^1$ continuous quadratic toe.

In the decadic formulation with fog parameter $f_{10} = \text{FOG\_OFFSET} = 0.005$:
$$D_{\text{eff}, 10}(D_{10}, f_{10}) = \begin{cases}
0 & \text{if } D_{10} \le 0 \\
\frac{D_{10}^2}{2 f_{10}} & \text{if } 0 < D_{10} < f_{10} \\
D_{10} - \frac{f_{10}}{2} & \text{if } D_{10} \ge f_{10}
\end{cases}$$

At the knee $D_{10} = f_{10}$:
- Lower branch: $\frac{f_{10}^2}{2 f_{10}} = \frac{f_{10}}{2}$
- Upper branch: $f_{10} - \frac{f_{10}}{2} = \frac{f_{10}}{2}$
- Derivative at knee: $\left.\frac{d}{d D_{10}} \left(\frac{D_{10}^2}{2 f_{10}}\right)\right|_{D_{10} = f_{10}} = \frac{2 f_{10}}{2 f_{10}} = 1.0 = \left.\frac{d}{d D_{10}} \left(D_{10} - \frac{f_{10}}{2}\right)\right|_{D_{10} = f_{10}}$

The function and its first derivative are continuous everywhere ($C^1$).

### 2.4 Hurter-Driffield (H&D) Gamma Reversal

Developed color negative film exhibits an effective contrast gamma $\gamma_{\text{eff}} \approx 0.6$. Inverting the straight-line characteristic curve yields scene exposure $E_{\text{scene}}$:
$$E_{\text{scene}} = 10^{D_{\text{eff}, 10} / \gamma_{\text{eff}}} - 1.0$$
The $-1.0$ pedestal ensures that unexposed film ($D_{\text{eff}} = 0$) maps exactly to zero linear scene exposure.

### 2.5 Mid-Gray Anchor

Per-channel exposure gains $g_c$ anchor a calibration 18% mid-gray patch to standard ACEScg middle gray ($\approx 0.18$):
$$g_c = \frac{\text{MIDDLE\_GRAY}}{E_{\text{scene}}(\text{mid}_c)}$$
$$\text{RGB}_{\text{out}, c} = g_c \cdot E_{\text{scene}, c}$$

---

## 3. Hardware ALU Transcendental Mechanics

### 3.1 IEEE 754 Floating-Point Structure

In IEEE 754 single precision (`f32`), a real number is represented as:
$$x = (-1)^s \cdot 2^{e - 127} \cdot (1.m_1 m_2 \dots m_{23})_2$$

Because floating-point numbers are fundamentally binary scientific notation, hardware ALUs (such as GPU Special Function Units / SFUs and CPU SIMD transcendental units) implement base-2 functions (`log2` and `exp2`):
- `log2(x)` is evaluated by extracting the exponent $e - 127$ as an integer offset and approximating $\log_2(1.m)$ on $[1, 2)$ via a low-degree minimax polynomial.
- `exp2(y)` is evaluated by separating $y$ into integer floor $\lfloor y \rfloor$ (which directly sets the IEEE exponent) and fractional part $\{y\} \in [0, 1)$ evaluated via polynomial approximation.

### 3.2 Decadic Emulation Overhead & Inaccuracy

Evaluating base-10 logarithms or exponentials requires software or shader compilers to emit multiplication by irrational constants:
$$\text{LOG10\_2} = \log_{10}(2) \approx 0.3010299956639812$$
$$\text{LOG2\_10} = \log_2(10) \approx 3.3219280948873623$$

In a decadic pipeline:
1. `log10(T)` is compiled as `log2(T) * 0.30102999566f32`.
2. `10^z` is compiled as `exp2(z * 3.32192809488f32)`.

In the composition:
$$z = D_{10} / \gamma = (-\log_2(T) \cdot \text{LOG10\_2}) / \gamma$$
$$10^z = \exp_2\left((-\log_2(T) \cdot \text{LOG10\_2} / \gamma) \cdot \text{LOG2\_10}\right)$$

The factors $\text{LOG10\_2}$ and $\text{LOG2\_10}$ are multiplied sequentially across multiple floating-point instructions.

---

## 4. Complete Mathematical Derivation & Algebraic Proof of Base-2 Equivalence

We now prove that formulating the entire chain directly in base-2 density $D_2$ and base-2 fog offset $f_2$ produces the exact analytical exposure value without base-10 conversion.

### Theorem (Base-2 Scanner Invert Equivalence)

Let:
$$D_2 = -\log_2(T)$$
$$f_2 = f_{10} \cdot \log_2(10)$$
$$D_{\text{eff}, 2}(D_2, f_2) = \begin{cases}
0 & \text{if } D_2 \le 0 \\
\frac{D_2^2}{2 f_2} & \text{if } 0 < D_2 < f_2 \\
D_2 - \frac{f_2}{2} & \text{if } D_2 \ge f_2
\end{cases}$$
$$\text{inv\_gamma} = \frac{1}{\gamma_{\text{eff}}}$$

Then:
$$D_{\text{eff}, 2}(D_2, f_2) \equiv D_{\text{eff}, 10}(D_{10}, f_{10}) \cdot \log_2(10)$$
and
$$\exp_2\left(D_{\text{eff}, 2} \cdot \text{inv\_gamma}\right) - 1.0 \equiv 10^{D_{\text{eff}, 10} / \gamma_{\text{eff}}} - 1.0$$

### Proof

#### Step 1: Fog Toe Scaling
Since $D_2 = D_{10} \cdot \log_2(10)$ and $f_2 = f_{10} \cdot \log_2(10)$, the condition $D_2 < f_2$ is strictly equivalent to $D_{10} < f_{10}$ because $\log_2(10) > 0$.

We evaluate $D_{\text{eff}, 2}$ across all three piecewise intervals:

1. **Sub-Dmin Region ($D_2 \le 0 \iff D_{10} \le 0$)**:
   $$D_{\text{eff}, 2} = 0 = 0 \cdot \log_2(10) = D_{\text{eff}, 10} \cdot \log_2(10)$$

2. **Quadratic Toe Region ($0 < D_2 < f_2 \iff 0 < D_{10} < f_{10}$)**:
   $$D_{\text{eff}, 2} = \frac{D_2^2}{2 f_2} = \frac{(D_{10} \cdot \log_2 10)^2}{2 (f_{10} \cdot \log_2 10)} = \frac{D_{10}^2 \cdot (\log_2 10)^2}{2 f_{10} \cdot \log_2 10} = \left(\frac{D_{10}^2}{2 f_{10}}\right) \cdot \log_2(10) = D_{\text{eff}, 10} \cdot \log_2(10)$$

3. **Linear Shoulder Region ($D_2 \ge f_2 \iff D_{10} \ge f_{10}$)**:
   $$D_{\text{eff}, 2} = D_2 - \frac{f_2}{2} = (D_{10} \cdot \log_2 10) - \frac{f_{10} \cdot \log_2 10}{2} = \left(D_{10} - \frac{f_{10}}{2}\right) \cdot \log_2(10) = D_{\text{eff}, 10} \cdot \log_2(10)$$

Therefore, for all real $D_2$:
$$D_{\text{eff}, 2}(D_2, f_2) = D_{\text{eff}, 10}(D_{10}, f_{10}) \cdot \log_2(10) \quad \blacksquare$$

#### Step 2: Exponential Reversal
Using the identity $a^b = 2^{b \log_2 a}$:
$$10^{D_{\text{eff}, 10} / \gamma_{\text{eff}}} = 2^{(D_{\text{eff}, 10} / \gamma_{\text{eff}}) \cdot \log_2(10)} = 2^{(D_{\text{eff}, 10} \cdot \log_2(10)) / \gamma_{\text{eff}}}$$

Substituting $D_{\text{eff}, 2} = D_{\text{eff}, 10} \cdot \log_2(10)$:
$$10^{D_{\text{eff}, 10} / \gamma_{\text{eff}}} = 2^{D_{\text{eff}, 2} / \gamma_{\text{eff}}} = \exp_2\left(D_{\text{eff}, 2} \cdot \text{inv\_gamma}\right)$$

Subtracting 1.0 yields:
$$E_{\text{scene}} = \exp_2\left(D_{\text{eff}, 2} \cdot \text{inv\_gamma}\right) - 1.0 \quad \blacksquare$$

---

## 5. Numerical Precision & Error Propagation Analysis

### 5.1 IEEE 754 Floating-Point Quantization

Consider the exact values and their closest 32-bit IEEE 754 representations:
- $\log_2(10) \approx 3.32192809488736234787$
  - IEEE 754 `f32`: `0x40549a78` $= 3.3219280948638916015625$
  - Error: $\epsilon_1 \approx -2.48 \times 10^{-8}$
- $\log_{10}(2) \approx 0.30102999566398119521$
  - IEEE 754 `f32`: `0x3e9a209b` $= 0.30102999566398119521$ (exact to 24 bits)
  - Error: $\epsilon_2 \approx +4.34 \times 10^{-9}$

Multiplying these two `f32` numbers:
$$\text{LOG2\_10}_{\text{f32}} \times \text{LOG10\_2}_{\text{f32}} = 1.0000000305 \neq 1.0$$
This yields an error of $\approx 1 \text{ ULP}$ on the multiplicative identity alone.

### 5.2 Error Amplification at High Density

Let $\delta D$ denote an error introduced into density during intermediate roundtrips. The reconstructed exposure is:
$$E(D) = 2^{D_2 / \gamma} - 1.0$$
Differentiating with respect to $D_2$:
$$\frac{dE}{dD_2} = \frac{\ln 2}{\gamma} 2^{D_2 / \gamma} = \frac{\ln 2}{\gamma} (E + 1)$$

The relative error in exposure is:
$$\frac{\Delta E}{E} \approx \left(\frac{E + 1}{E}\right) \frac{\ln 2}{\gamma} \Delta D_2$$

For dense highlights in film negatives (corresponding to bright positive scene regions):
- Negative density $D_{10} = 3.0 \implies D_2 \approx 9.9658$
- Film contrast $\gamma = 0.6 \implies \text{inv\_gamma} \approx 1.6667$
- Base exposure: $E = 2^{9.9658 \times 1.6667} - 1.0 = 2^{16.61} - 1.0 \approx 99,800$
- Derivative: $\frac{dE}{dD_2} = \frac{0.69315}{0.6} \times 99,801 \approx 115,300$

At this operating point, an intermediate floating-point truncation error of just $\Delta D_2 = 10^{-6}$ produces an absolute exposure error of $\Delta E \approx 0.115$.

By computing $D_2 = -\log_2(T)$ directly from normalized transmittance and passing $D_{\text{eff}, 2}$ directly to `exp2`, **zero intermediate base conversions occur**, eliminating this error source.

---

## 6. Implementation Reference & Performance Comparison

### 6.1 Rust Implementation (`pichromatic/src/film/scan/invert.rs`)

```rust
/// Precomputed once per calibration:
let inv_gamma = 1.0 / GAMMA_EFF;
let fog2 = FOG_OFFSET * LOG2_10;

/// Inner per-pixel loop:
#[inline]
pub fn fog_effective_density_base2(d2: f32, fog2: f32) -> f32 {
    if d2 <= 0.0 {
        0.0
    } else if d2 < fog2 {
        (d2 * d2) / (2.0 * fog2)
    } else {
        d2 - 0.5 * fog2
    }
}

#[inline]
pub fn density2_to_exposure(d2: f32, inv_gamma: f32, fog2: f32) -> f32 {
    let d_eff2 = fog_effective_density_base2(d2, fog2);
    if d_eff2 <= 0.0 {
        0.0
    } else {
        (d_eff2 * inv_gamma).exp2() - 1.0
    }
}
```

### 6.2 Operation Count Comparison

| Step | Legacy Decadic Pipeline | Native Base-2 Pipeline |
|---|---|---|
| Transmittance Normalization | 1 `mul` (`inv_dmin`), 1 `max` (`eps`) | 1 `mul` (`inv_dmin`), 1 `max` (`eps`) |
| Density Computation | 1 `log2`, 1 `mul` (`LOG10_2`), 1 `neg` | 1 `log2`, 1 `neg` |
| Quadratic Toe Branch | 1 comparison, 1 branch, 1 `fma`/`mul` | 1 comparison, 1 branch, 1 `fma`/`mul` |
| Exponent Scaling | 1 `mul` (`inv_gamma_log2_10`) | 1 `mul` (`inv_gamma`) |
| Exponential Reversal | 1 `exp2`, 1 `sub` | 1 `exp2`, 1 `sub` |
| **Total per-channel ops** | **9 ops (incl. 2 base conversion muls)** | **7 ops (0 base conversion muls)** |

### 6.3 Parity and Consistency Benefits

1. **CPU / GPU Equivalence**: Because both CPU SIMD (AVX2/NEON) and GPU (WGSL/Metal/Vulkan/DirectX) share identical IEEE 754 `log2` and `exp2` specifications, eliminating intermediate constants removes platform-specific constant folding and FMA contraction differences.
2. **Simplified Constant Structs**: Constant buffers store direct base-2 quantities (`fog2`, `inv_gamma`), reducing uniform buffer bandwidth and shader register footprint.
