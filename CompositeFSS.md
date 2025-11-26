## 0. 先定清楚目标：你要超越谁、在哪些指标上超越

对照一下现状：

* **Sigma**：

    * 专注半诚实，所有非线性都拆成一串原始 FSS 原语（DReLU、TR/ARS、LUT、selectlin 等) 的调用。
    * 每个 GeLU / SiLU / Softmax / LayerNorm 都是“多个 FSS 协议+少量 LUT”串联，优化了 LUT 尺寸（GeLU 只用 2⁸ table，SiLU 用 2¹⁰ 等）。
    * 但 key 大体是 **“每个原语一个 key”** 的模式。

* **SHARK**：

    * 在 Sigma 等半诚实 FSS 协议之上，引入 **interactive FSS** + 混合 Boolean/arith 认证，得到恶意安全，同时把之前 Eurocrypt’20 mixed‑mode FSS 那种 “n+s 位输入” 的开销降回到 n 位。
    * 对 ReLU、ARS、ReluARS、splines、reciprocal 都有针对性的 IFSS 协议，已经在 keysize、AES 调用和通信上明显优于 Escudero 等。

**你想做的事情**：

* 仍然是 2PC + preprocessing 模型；
* 仍然用 FSS / PDPF 作为主要密码内核（Crypto’20 一脉 + 你自己的 PDPF 实现）；
* 但在 *协议层* 给出一个新的 **Composite‑FSS 抽象 + 实现**：

    * 每个“复合门/层”只需要 **1–2 个 PDPF 程序 key**；
    * 离线 key 量严格少于 SHARK / Sigma 对应功能；
    * 在线 PRG/AES 次数也严格减少；
    * 抽象本身足够 general，能统一 ReLU/ReluARS、GeLU/SiLU、Softmax、LayerNorm、Reciprocal 等。

换句话说：你要做的不是“再提一个更好的 ARS 协议”，而是提出一个 **新的组织方式**，把 SHARK / Sigma 这些“一个功能 = 多个 FSS 原语组合”的做法收编成 “一个 Composite‑FSS gate = 一小撮 PDPF 程序 + 少量线性逻辑”。

---

## 1. 抽象层：给自己发明一个“Composite‑FSS for Structured Nonlinearities”

### 1.1 功能族：一维分段算子 + 线性层融合

观察 SHARK / Sigma 所有非线性：

* ReLU / DReLU / GEZ
* ARS / TR / ReluARS（带 gap）
* 一维分段多项式：Sigmoid / tanh / CELU / splines（SHARK 已有 spline）
* GeLU / SiLU：本质是 ReLU + 在小区间上的一个“修正 δ(x)” LUT。
* Reciprocal / rsqrt：裁剪+规格化+小 LUT + 一两个牛顿步。
* Softmax：max + nExp + 除以 sum；其中 nExp/reciprocal 部分又是上面的元素函数。

抽象出一个统一的函数族：

> 所有非线性都属于：
>
> * **Affine + 少量阈值 + 少量 LUT + 低次多项式** 的组合；
> * 所有阈值和多项式分段 **都在一维标量 x 上**，只是在模型中被大量复制。

于是可以定义一个新对象：

> **结构化一维 Composite‑FSS (SC‑FSS)**：
> 对于函数族
> [
> \mathcal{G}=\left{g(x) : g(x)=A x + B +\sum_{i=1}^m 1[x\in I_i]\cdot p_i(x) + \sum_{j=1}^u 1[x\in J_j]\cdot c_j \right}
> ]
> （I_i、J_j 是区间，p_i 为低次数多项式），给出一个 FSS 方案：
>
> * KeyGen 只生成 **O(m+u)** 大小的 key；
> * Eval 在固定 x 上，既返回 g(x) 的分享，也返回所有需要的“控制比特/索引”。

你可以在 paper 中把 ReLU/ReluARS、splines、GeLU/SiLU、Reciprocal 都写成这个形式，证明它们属于同一个函数族。SHARK 的 spline / reciprocal、Sigma 的 GeLU / SiLU 其实都已经“半只脚”在这里了，只是他们依然做成很多 FSS 原语串联。

### 1.2 Cryptographic primitive：PDPF 作为唯一内核

这里沿用你前面那篇“programmable DPF = 通用 FSS 核”的抽象，但这次要多做一层事：

1. **统一接口**：

    * PDPF.Setup(pp₀,pp₁)；
    * PDPF.ProgGen(desc(f))→(k₀,k₁)；
    * PDPF.Eval(pp_b,k_b,·)。

2. **Composite‑FSS = 只用 PDPF，但把“多个 PDPF 功能”融合成一个**：

    * 对于每个复合门 τ（ReluARS, GeLU, SoftmaxPiece, ReciprocalPiece, …），定义一个 PDPF 程序
      [
      f_{\tau,r_{\text{in}}}(\hat x)\in G_\tau
      ]
      其中输出 group G_τ 封装：

        * 所有比较位 (w,t,d,…)；
        * 小 LUT 的 index；
        * 可能还包括一组区间的多项式系数。

3. **Composite key**：

    * 每个门只存：

        * 掩码 share；
        * 若干 PDPF key (常数个)；
        * 少量门本身的参数（如 truncation f、 LUT ID 等）。
    * Eval 的在线步就是：

        * 调 PDPF.Eval 一两次；
        * 再做固定 pattern 的线性运算和少量乘法（三元组）。

这一步其实就是**把 SHARK / Sigma 的“FSS‑pipeline”折叠成一个 PDPF 调用 + 线性逻辑**。

---

## 2. 真正能降 keysize 和在线复杂度的三个关键点

仅仅“把多个协议折叠到一个 PDPF 程序里”还不一定 beating Sigma/SHARK——你需要抓住他们没充分 exploit 的三个结构：

### 2.1 **横向打包**：同一类 gate 批量用一个 PDPF

Sigma / SHARK 的典型模式：

* 每个标量 x 调一个 DPF key（DReLU / TR / LUT），即使参数一样，只是输入值不同。
* 他们会在 **同一个门内部** 复用 key（比如 Sigma 的 GPU‑GeLU 会用同一个 DReLU key 在 y, y+255, y−256 上 eval 三次），
  但不会把 **同一 layer 的很多 GeLU** 打包成“一个大的 FSS”。

你的 Composite‑FSS 可以做一件更 aggressive 的事：

> 为整层（甚至多个 layer）中所有“形同” non‑linear gates，生成 **一个 batched PDPF 程序**，domain = {0,1}^logM × {0,1}^n：
> [
> f((i,x_i)) = \text{所有该 gate 需要的辅助比特/系数}
> ]

* 这里 i 是 gate index（或者 token index + head index + neuron index），x_i 是对应的 masked 输入；
* PDPF 的树结构在不同 i 上自然共用，大量内部 node 的 seeds/纠正值可以被批量生成；
* 密钥长度 ~ O(log(M·2ⁿ)) * security_param + O(#leaf_updates)；
* 相比“每个 gate 一棵树”，你节省了一倍以上的 per‑tree metadata，尤其是：

    * 路径上的 tweak / correction 不再被重复存储；
    * 许多 gate 因为有 **相同的阈值集合 / 区间划分**，可以共用整段叶子结构。

这其实是一个 **结构化 multi‑point PDPF**：你把“index i + 值 x_i”合在一起，通过描述函数族的结构，让 dealer 只负责为“少数有改变的 leaf”打 patch。

**复杂度直觉**：

* 假设一层有 M=10⁶ 个 GeLU，每个要 3 个比较位 + 1 LUT index；
* SHARK / Sigma：每个 x 单独调几次 DPF，keysize ~ O(M·λ·n)；
* 你的 batched PDPF：

    * 内部树只建一次（高度 log(M·2ⁿ)~logM + n）；
    * 每个 gate 只需要在叶子或 near‑leaf 处存一个小的 “payload patch”；
    * keysize ~ O(λ·(logM+n) + M·(payload_bit))；
    * online 评估：可以用 batched eval，在 GPU 上 re‑use 每一层的 AES 扩展。

这在实践上给的是“同样量级的 PRG 调用，但 per gate 常数 *明显* 降低”，而 keysize 则会减少一个看得见的因子。

### 2.2 **纵向融合**：把 ReLU / ARS / ReluARS / GeLU 拆成同一组辅助比特

SHARK 已经有一条很好的公式，把 ReluARS 写成三个比特 w,t,d 再挂一个小 LUT：

* w = 1[x < r_in]
* t = 1[x mod 2^f < r_in mod 2^f]
* d = “sign after shift”的某种组合（MSB & w）。

Sigma 这边也有一条自己的“DReLU + TR + Clip + LUT” pipeline。

你可以做的 **更 radical**：

> 定义一个统一的辅助向量
> [
> \mathbf{b}(x)=(\text{msb}(x+r_\text{offset}),\ \sigma_{1}(x),\dots,\sigma_{L}(x))
> ]
> 其中 σ_i(x) 覆盖：
>
> * 所有你会在 ReLU/ARS/ReluARS 用到的比较；
> * GeLU / SiLU 的 clipping / |·| / “是否落在修正区间”；
> * Reciprocal / rsqrt 里的区间定位（e 区间 / m 区间）；
> * Softmax 的 “是否超出 cut‑off”。

然后：

* **所有这些 σ_i** 统一通过 **一个 PDPF 程序** 计算出来；
* 每个 gate 自己只从中拿出它需要的那几个 bit + index；
* 这就把原来“每种功能都配一套 DPF”的世界，变成“整个网络共用一两个全局 PDPF 程序”。

这样你在抽象层上可以主张：

> “我们展示了一个 unified helper‑bit FSS：对固定 bitwidth n，它一次性支持所有 truncation/ReLU/spline/reciprocal 等辅助比特的生成，并且 keysize 只线性依赖于 ‘总的不同阈值和区间个数’，而不是功能数×层数×神经元数。”

这个是比较容易被 reviewer 认可为 **non‑incremental** 的 conceptual contribution。

### 2.3 **LUT 收缩 + PDPF 内部编码**

Sigma 把 LUT 做得已经很小：GeLU 的 δ(x) 只需要 2⁸ entries，SiLU 需要 2¹⁰，inverse / rsqrt 在 2¹³–2¹⁶ 之间。

你的空间可以在两点做得更 agressive：

1. **把 LUT 也塞进 PDPF 的 payload**：

    * 而不是“一个单独的 LUT 协议（自己要一个 DPF key）”，
    * 直接让 PDPF 输出：

        * `control bits` + `在这一段的多项式系数`；
    * 即 GeLU/SiLU/Reciprocal 的 δ(x) 不用 LUT，而是用 **分段多项式 + 低 bitwidth 多项式系数** 编码在 PDPF 中。

2. 对于某些函数（如 SiLU / Sigmoid / CELU），你可以沿用 Sigma 的“对称性 + clipping 把区间收窄到 [-16,16] + even/odd 函数”的 trick，
   但**真正** LUT 的尺寸只存在在「编译到 PDPF 程序时」，在线时 parties 根本看不到 LUT，只看到来自 PDPF 的系数 shares。

这块的卖点是：

* **keysize 维度**：你不再需要 `DPF_LUT` 这种 2^{n-ν-1} 次 AES 的通用 LUT 方案（Pika / Sigma 用的），
* 而是用“结构化 multipoint PDPF”来实现“区间→系数”的映射；
* 对于给定非线性，keysize ~ O(#intervals × coef_bits)，与 bitwidth n 几乎解耦（只在比较时依赖 n）。

---

## 3. 具体 gate 设计示例：如何写到 paper 里

我用两个核心例子说明怎么“落地”成 Composite‑FSS 方案：一个是 ReluARS，一个是 GeLU。你可以按同样套路再写 Softmax / Reciprocal / LayerNorm。

### 3.1 Composite‑FSS for ReluARS (对比 SHARK)

回顾 SHARK 的 ReluARS IFSS：它已经把 ReLU 和 ARS fuse 成一个 2‑round 的 IFSS，keysize ~ (n+f)(λ+κ)，AES 调用 4n+2f，明显优于逐个调用 ARS + ReLU。

#### 3.1.1 你的抽象

你可以沿用你前面那套：

* 输入：masked (\hat x = x + r_{\text{in}})；
* 要输出：(\hat y = \mathrm{ReluARS}*{n,f}(x) + r*{\text{out}})；
* 用三比特 (w,t,d)：

    * w = 1[x < r_in]；
    * t = 1[x mod 2^f < r_in mod 2^f]；
    * d = sign(ARS(x)) 的变体；
* 有一个 2³=8 entry 的小表 T[d,w,t]，给出所有“mask 校正项”（你在前一个回答已经写过）。

现在把这些全部 encode 到 **一个 PDPF 程序 f_ReluARS** 中：

* 输出 group G = Z₂³ × Z_{2^n}^8；
* Eval(pp_b,k_b, \hat x) → (w_b,t_b,d_b, T_b[0..7])；
* parties 聚合得到：

    * (w,t,d) 的 shares；
    * 表 T[d,w,t] 的 shares（只需要访问对应 index）。

在线算法：

1. 公开算 u = LRS_{n,f}(\hat x)；
2. 用 (d) 转换成 arith share；
3. 线性算：
   [
   \hat y_b = d_b \cdot u + T_b[d,w,t].
   ]

**离线复杂度**：只需要 1 个 PDPF.ProgGen 调用；keysize∝(λ·(n+f) + 8·n)。

**对比 SHARK**：

* SHARK ReluARS 的 key：

    * 需要两套 DCF key：一个 n 位比较，一个 f 位比较 + 一些表的认证 share；
* 你的方案：

    * 只需要一个 PDPF key；DPF 的“内部结构”可以同时涵盖高位/低位比较（通过 output group packing）；
    * 不需要额外的 LUT 协议或者 B2A primitive，所有 mask 操作都在 PDPF 生成的表里完成。

理论上可以 argue：对相同安全参数 λ,κ，你的 keysize 至少可以减少一个常数因子（大约是 SHARK 里 “两个分离 DCF vs 一个联合 PDPF”的差异），AES 调用次数则接近“1×DPF.eval vs 2×DCF.eval”。

### 3.2 Composite‑FSS for GeLU (对比 Sigma)

Sigma 的 CPU‑GeLU：ReLU + Clip + Abs + TR + LUT，GPU 版又做了一些 key reuse 的 trick。

你可以重新组织为：

> [
> \mathrm{GeLU}(x) = \mathrm{ReluARS}_{n,f}(x) - \delta(x)
> ]
> 其中 δ(x) 只在小区间 (-C,C) 内非零（C≈2⁴），在这一区间上用一段 degree‑d 的 spline 逼近。

Composite‑FSS 做法：

1. 复用上面的 **ReluARS Composite‑FSS**：得到 y₁ = ReluARS_{n,f}(x) + r_out¹；

2. 定义一个新的 PDPF 程序 f_δ：

    * 输入：同样的 masked (\hat x)；
    * 输出：

        * 比特 `in_range` = 1[|x|<C]；
        * 区间 index idx(x)；
        * 对应多项式系数向量 β_idx (用于近似 δ(x))。

3. 在线阶段：

    * Eval PDPF f_δ 得到 (in_range, idx, β) 的 shares；
    * 公开算 z = (x 的“中心化实数表示”，例如 (\hat x-r_\text{in}))；
    * 用少数乘法 triple 算 δ(x) 的 share；
    * 用 in_range 乘上 δ(x)（区间外 δ=0）；
    * 输出：
      [
      \hat y = y_1 - \delta(x) + r_{\text{merge}}.
      ]

如果你愿意更激进一点，完全可以把 f_ReluARS 和 f_δ **合并为一个大 PDPF 程序**，这样整条 GeLU 都只需要一次 PDPF.Eval。

**对 Sigma 的优势**：

* Sigma 的 GeLU 至少需要：

    * 1× DReLU 协议，1× TR 协议，1× LUT 协议，外加若干 selectlin 调用；
    * keysize ≈ key(DReLU) + key(TR) + key(LUT) + 常数×n；
* 你的方案：

    * 1× ReluARS‑Composite‑FSS PDPF + 1× δ‑spline PDPF（甚至可以合并成 1 个）；
    * keysize 只和“ReluARS 的几个阈值 + spline interval 数 × (degree+1)”成正比；
    * 在线 AES 调用次数 ≈ 1–2 次 PDPF.eval 的成本，而不是多个不同 FSS 协议的叠加。

这里的 conceptual 升级是：

> “我们证明，对于所有 ‘ReLU+局部修正’ 型激活（包括 GeLU/SiLU/CELU/Sigmoid），可以统一成一个 Composite‑FSS gate，其 FSS key 长度只与修正区间的分段数和多项式次数有关，与模型大小、层数、bitwidth 基本解耦。”

---

## 4. 系统层 / layer‑fusion：把 Composite‑FSS 做成“Layer‑Level 原语”

有了单 gate 的 Composite‑FSS，你可以进一步做两件能写进 paper 的系统层贡献：

### 4.1 MatMul + Nonlinearity + Truncation 一次性融合

类似 Sigma 里“effective bitwidth”的 observation：线性层后面必然有 truncation，于是非线性输入的有效 bitwidth 是 n-f，而不是 n。

你可以在 Composite‑FSS 层面做更大胆的 fusion：

> 一个大 gate：
> [
> (X,W,b)\mapsto \mathrm{Nonlin}\big(\mathrm{Trunc}(XW+b)\big)
> ]
> 其中 Nonlin 是 Composite‑FSS 支持的任何一维算子（ReluARS / GeLU / SiLU）。

具体做法：

* 线性层依旧用 SPDZ‑style Beaver 乘法（不涉及 FSS）；
* 但你**不暴露**中间值 Z = XW+b，而是只在内部保持 masked (\hat Z = Z + r_{\text{in}})；
* 对每个坐标 zᵢ，调用同一个 “Nonlin+Trunc Composite‑FSS” PDPF 程序：

    * 由 PDPF 直接负责：

        * truncation 的比较位（w,t 等）；
        * Nonlin 所需的 sign / interval index / LUT index；
    * 只需要 1–2 个 global PDPF key + per‑neuron 的很小 meta（比如 truncation f、Nonlin 类型 ID）。

这对 Sigma / SHARK 的区别在于：

* 他们是“线性层 → faithful truncation 协议 → DReLU/GeLU 等 FSS”，各部分 key 分开；
* 你是“线性层一出 Z，立刻交给一个大的 Composite‑FSS 程序做 trunc+nonlin 一条龙处理”，在 FSS 层面只看见一个 gate。

### 4.2 软最大 / LayerNorm：整 block Composite‑FSS

Softmax 和 LayerNorm 看起来是多维操作，但需要 FSS 的核心仍然是一维 exp / reciprocal / rsqrt：

* Softmax: y_i = exp(x_i)/Σ exp(x_j)；
* LayerNorm: (x_i - μ)/sqrt(Var[x] + ε)。

你的 Composite‑FSS 可以做成两级：

1. 一级：用 PDPF 实现 exp / reciprocal / rsqrt 的 **structured univariate Composite‑FSS**，支持 clip + 分段 + 修正；
2. 二级：在协议层，用 standard MPC（不需要 FSS）求 sum / mean / variance，再调用一级 Composite‑FSS 来算 reciprocal / rsqrt；

此外还可以加入 Sigma 已经用过的 transformer‑specific 优化，如有效 bitwidth、注意力 mask 等等。

区别在于：你的抽象把这些都放在一个 **统一的 Composite‑FSS API** 下面，而不是一堆 ad‑hoc 的 FSS 协议。

---

## 5. 安全性与“非增量性”怎么写

### 5.1 安全性

由于你整个框架是：

* “PDPF + additive secret sharing + 少量 Beaver triple”，
* 没有引入新型密码假设，安全性证明其实非常干净：

1. 定义 Composite‑FSS 的功能性：

    * 对每种 gate τ，定义 offset 功能 (F_{\tau,\gamma}^{[r_{\text{in}},r_{\text{out}}]})；
2. 证明：给定 PDPF 是安全的 FSS 核，那么 Composite‑FSS 满足标准 simulation‑based 安全（单方腐化时存在 simulator）；
3. 对整个网络使用 sequential composition 直接得到端到端安全。

如果你想进一步对标 SHARK，可以加一段讨论：

* 如何把 Composite‑FSS 升级到 **interactive FSS + authenticated shares**；
* 基本上是重复 SHARK 的 lifting 技巧，只是把 FSS 内核换成你的 PDPF‑Composite‑FSS；
* 强调：你在 malicious 模型下仍然可以保持“keysize 相对 SHARK 有优势、online AES 次数不增加太多”。

### 5.2 为什么“不是增量工作”

这块 reviewer 会看得很细，所以你可以从以下几个角度强调：

1. **抽象层新东西**：

    * 提出一个统一的 **Structured Composite‑FSS** 定义，
    * 并证明多个看似不同的非线性（ReluARS、GeLU、SiLU、Reciprocal、Sigmoid、CELU 等）都属于同一函数族；
    * 给出一个通用的“从 piecewise polynomial 描述编译到 PDPF 程序”的 compiler。

2. **跨函数、跨层的 helper‑bit 统一 FSS**：

    * Sigma / SHARK 的 helper bits 是 per‑function 设计的：DReLU 的比较位、TR 的 w,t、spline 的 interval 指示等各玩各的；
    * 你直接定义一个 global helper vector，把所有相关比较合在一个 PDPF 程序里，这是概念上大的统一。

3. **batched / structured PDPF**：

    * 提出把“整层同类 gates”一起用一个 PDPF 的想法，并在复杂度分析里证明 keysize、预处理复杂度可以降一个可观的因子（因子依赖于层大小 / gate 数）；
    * SHARK / Sigma 都只在 single‑gate 范围内做 fusion（比如 ReluARS），没做到 layer‑level 的 FSS batched。

4. **近似层的统一 recipe + FSS 实现**：

    * 虽然 Sigma 已经有“函数特定近似”的 recipe（GeLU/SiLU/Softmax/LayerNorm），但它们在协议层仍然是 ad‑hoc 的组合；
    * 你的 Composite‑FSS 提供一个统一的“分段+多项式+PDPF 编码”的 template，可以重用到新函数（比如 CELU/SELU/Swish）的 FSS 协议中。

---

## 6. 真正落成一篇 paper 你还需要做什么

最后我给你一个比较务实的 checklist，方便你规划论文和实验：

1. **Formalization**

    * 写出 Composite‑FSS 的 formal definition（Gen_τ, Eval_τ,b、correctness、privacy）；
    * 定义 Structured‑Univariate‑Function 家族，证明常见非线性都在里面；
    * 给出“从函数描述到 PDPF 程序”的通用编译器规范。

2. **协议细节**

    * 完整写出几个核心 gate 的 Gen/Eval 伪代码：

        * ReluARS（带 gap / 无 gap）；
        * GeLU / SiLU；
        * Reciprocal / rsqrt；
        * Softmax‑block（分为 Max + nExp + Normalize + maybe clipping）；
    * 对每个 gate，给出 keysize / online AES call /通信量的封闭形式表达，并把它们与 SHARK / Sigma 对应协议表格对比（类似他们 Table 1 & 2）。

3. **理论对比**

    * 对于固定 n=64, f=12, common parameters，算一个具体数字：

        * keysize_Composite‑ReluARS vs keysize_SHARK‑ReluARS；
        * keysize_Composite‑GeLU vs keysize_Sigma‑GeLU；
    * 证明至少在这些配置下你有 X× 的节省（哪怕是 2–4× 这种级别也够 paper）。

4. **实现 & 实验**

    * 在你已有的 PDPF 库上实现一个 prototype：

        * 先实现单 gate Composite‑FSS（ReluARS / GeLU）；
        * 再尝试 batched PDPF 和 layer‑fusion。
    * 选择一两个模型做 demo：

        * BERT‑tiny / base（方便和 Sigma 对比）；
        * 一两个 CNN（方便和 SHARK 对比）。
    * 报告：

        * per‑gate latency vs SHARK / Sigma；
        * end‑to‑end latency & communication；
        * key 生成时间 & 预处理 material 大小。

5. **讨论**

    * 安全扩展到 malicious：引用 SHARK 的 interactive FSS 方法，说明如何在你的 Composite‑FSS 上 plug‑in；
    * 限制 & 未来工作：

        * batched PDPF 在极大网络/长序列上的内存占用；
        * 更复杂激活（比如 attention 中某些自定义函数）如何适配。

---

如果你愿意，下一步我可以帮你把 **某一个具体 gate（比如“Composite‑ReluARS”或者“Composite‑GeLU”）的 Gen/Eval 写成正式伪代码 + 复杂度表**，直接可以搬到论文里当 protocol figure & theorem。
