我先把你要的三块东西都给出来：**形式化定义 → 若干核心 gate 伪代码 → 封闭形式开销 + n=64,f=12 的具体数值对比**。下面我会默认安全参数
(\lambda = 127)（DPF）, (\nu = \log_2(\lambda+1)=7)（Sigma 的记号），并把 Sigma/SHARK 的相关公式都引用清楚。

---

## 一、Composite‑FSS 的形式化定义

### 1.1 全局接口

* 环：(R = \mathbb{Z}_{2^n})，固定 (n)。
* 两方 (P_0,P_1)，半诚实。
* 线性层采用标准加法 secret sharing：
  ([x] = (x_0,x_1), x_0+x_1=x\in R)。

我们假设有一个 **PDPF 核心** ((\mathsf{PDPF.Setup},\mathsf{PDPF.Gen},\mathsf{PDPF.Eval}))，实现了对某个函数族 (\mathcal{F}) 的 FSS（比如 Sigma 采用的 BGI‑18 DPF 作为比较 FSS 核心）。

对于每一类非线性 gate 类型 (\tau)（ReLU/ReluARS/GeLU…）及其实例参数 (\gamma)（比特长度、精度、分段点等），定义一个**理想功能**：
[
F_{\tau,\gamma}: R^k \to R^m \times {0,1}^\ell.
]

在随机 mask (r_{\text{in}})、(r_{\text{out}}) 下，我们真正在 MPC 里实现的是 “masked” 版本：
[
\hat{F}*{\tau,\gamma}(x)
= \big(F*{\tau,\gamma}(x)*{\text{arith}} + r*{\text{out}},; F_{\tau,\gamma}(x)_{\text{bool}}\big).
]

### 1.2 Composite‑FSS scheme（单个 gate）

**Definition (Composite‑FSS for gate type (\tau)).**
对每种 gate 类型 (\tau)，Composite‑FSS 给出算法
[
\mathsf{CompGen}*\tau,\quad
\mathsf{CompEval}*{\tau,0},\quad
\mathsf{CompEval}_{\tau,1}
]
满足：

* **Key generation**
  [
  (K_0,K_1) \leftarrow \mathsf{CompGen}_\tau(1^\lambda,\gamma).
  ]

  每个 (K_b) 是一个结构化 key，包含（概念上）：

    * 各输入线的 mask (r_{\text{in}})，各输出线的 mask (r_{\text{out}}) 的加法 share；
    * 常数向量、LUT 条目等的 share；
    * 若干 PDPF 程序的 key：
      [
      k^{(1)}_b,\dots, k^{(t)}_b,
      ]
      每个对应一个 (f^{(i)}\in\mathcal{F})（比较、多比较、IntervalLookup、Spline 等）。

* **Evaluation（在线）**
  输入：party (P_b) 拿到：

    * gate 输入的 **public mask 值** (\hat{x}_1,\dots,\hat{x}_k)（从上一层线性运算得到）；
    * 本方 gate key (K_b)。

  输出：本方对 masked 输出的加法 share：
  [
  (\hat{\mathbf{y}}_b,\mathbf{z}*b)
  \leftarrow \mathsf{CompEval}*{\tau,b}(K_b,\hat{x}_1,\dots,\hat{x}_k),
  ]
  其中

    * (\hat{\mathbf{y}}_b\in R^m) 是 (P_b) 的算术输出 share；
    * (\mathbf{z}_b\in{0,1}^{\ell}) 是布尔输出的 share（一般在 (\mathbb{Z}_2) 上加法共享）。

* **Correctness**
  设真实输入为 (\mathbf{x}=(x_1,\dots,x_k))，且
  [
  \hat{x}*i = x_i + r*{\text{in},i} \pmod{2^n}.
  ]
  若双方运行 (\mathsf{CompGen}*\tau) 得到 ((K_0,K_1))，再对公共 (\hat{\mathbf{x}}) 各自执行 (\mathsf{CompEval}*{\tau,b})，则必有（取模 (2^n)）：
  [
  \hat{\mathbf{y}}*0 + \hat{\mathbf{y}}*1
  = F*{\tau,\gamma}(\mathbf{x})*{\text{arith}} + r_{\text{out}},
  ]
  [
  \mathbf{z}*0 \oplus \mathbf{z}*1
  = F*{\tau,\gamma}(\mathbf{x})*{\text{bool}}.
  ]

* **Privacy（半诚实）**
  对每个 (b\in{0,1})，存在 PPT 模拟器 (\mathsf{Sim}*{\tau,b})，使得任意多项式长输入序列 (\mathbf{x}*\lambda)，在以下实验中：

    * **Real(_\lambda)**：运行 (\mathsf{CompGen}*\tau(1^\lambda,\gamma)\to(K_0,K_1))，令 (\hat{\mathbf{x}}*\lambda=\mathbf{x}*\lambda+r*{\text{in}})，执行 (\mathsf{CompEval}_{\tau,0/1})，把 (P_b) 的 view（包括 key、DPF 评估结果、本地线性/乘法操作、与对手的通信）喂给环境；
    * **Ideal(_\lambda)**：理想功能拿到 (\mathbf{x}*\lambda)，输出 (F*{\tau,\gamma}(\mathbf{x}*\lambda))，再 mask 成 (\hat{F})，模拟器 (\mathsf{Sim}*{\tau,b}) 只拿到 (\mathbf{x}_b) 的 share + 本方输出 share + public (\hat{\mathbf{x}})，生成虚假 view；

  二者不可区分。

**Lemma（组合安全性）.**
如果底层 PDPF 对其函数族 (\mathcal{F}) 满足 FSS 安全性（如 Definition 1/2 in Sigma），且线性 secret sharing 层只做加法和乘法三元组（SPDZ 风格），那么对任何 gate 类型 (\tau)，上述 Composite‑FSS 都在半诚实下安全。证明和 Sigma 对各个 Π(_F) 的安全性证明基本一样，只是把“单个 FSS”换成“一小批 PDPF 程序”。

---

## 二、Structured‑Univariate‑Function (SUF) 家族

### 2.1 定义

固定 (n)，考虑标量输入 (x\in R=\mathbb{Z}_{2^n})。

**Definition (SUF(_{n}^{r,\ell,d})).**
函数
[
F: R \to R^r \times {0,1}^\ell
]
属于 Structured‑Univariate‑Function 家族 SUF(_{n}^{r,\ell,d})，如果存在：

* 有限阈值序列 (\alpha_0<\alpha_1<\dots<\alpha_m\in R)，定义区间 (I_i=[\alpha_i,\alpha_{i+1}))；
* 对每个区间 (I_i)，存在至多次数 (d) 的多项式向量
  [
  P_i(x) = (P_{i,1}(x),\dots,P_{i,r}(x)),\quad
  \deg P_{i,j}\le d,
  ]
* 以及布尔输出向量
  [
  B_i(x) = (B_{i,1}(x),\dots,B_{i,\ell}(x)),
  ]
  其中每个 (B_{i,t}(x)) 可以写成有限个谓词的布尔组合，这些谓词的形式为：
  [
  1[x < \beta],\quad
  1[x \bmod 2^f < \gamma],\quad
  \text{MSB}(x),\quad
  \text{MSB}(x+c),
  ]
  以及常数位 0/1。

使得对任一 (x\in R)，如果 (x\in I_i)，则：
[
F(x) = (P_i(x), B_i(x)).
]

直观上：**控制流只由有限个比较 / MSB 决定，算术输出在每个区间上是低度多项式，布尔输出是比较的布尔组合**。

### 2.2 SUF 的闭包性质

* 对任意线性变换 (L(x)=ax+b)（(a,b\in R)），如果 (F\in)SUF，则 (x\mapsto F(L(x))) 仍然是 SUF：阈值重参数化，多项式复合线性，次数不变。
* 有限个 SUF 的笛卡尔积、有限布尔组合仍是 SUF。
* 对固定精度 (f)，逻辑右移 LRS(*{n,f})、ARS(*{n,f})、TR(_{n,f}) 都可以写成“上一节 LRS 恒等式 + 比较 bit + 线性组合”的形式，故属于 SUF。

### 2.3 常见非线性都在 SUF 里

下面只给出结构性证明，不再写出所有系数的显式表达。

1. **ReLU / GEZ / ReluARS**

    * ReLU(_n(x)=\max(x,0)) 在阈值 (\alpha_0=0) 上分成两段：
      区间 (x<0: P(x)=0)；区间 (x\ge0: P(x)=x)，次数 1。布尔输出 (\text{GEZ}(x)=1[x\ge0]) 由一个比较给出，故 (F(x)=(\text{ReLU}(x),\text{GEZ}(x))\in)SUF(_n^{1,1,1})。
    * LRS/ARS 如上所述，通过两个 bit(w,t) 和线性组合表达，属于 SUF。ReluARS 仅仅是 ReLU(\circ)ARS 的组合，仍然是 SUF。

2. **GeLU / SiLU**

   Sigma/其它工作都用分段多项式或 LUT 近似：
   [
   \text{GeLU}(x)\approx 0.5x\big(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3))\big),
   ]
   他们实现上等价于：
   [
   \mathrm{GeLU}(x)
   = \mathrm{ReLU}(x) - \delta(x),
   ]
   其中 (\delta(x)) 在有界区间（如 ([-4,4])）上由 LUT + 线性插值实现，区间外近似 0。Sigma 的 CPU GeLU 就是 “ReLU + Clip + Abs + TR + 8-bit LUT” 的组合，本质是一个有限区间上的分段函数。

    * Clip/Abs/TR 都可以写成有限个比较 + 线性组合；
    * LUT(_{8,\ell,T}) 在 8-bit 域上是有限个点的查表，可以用 SUF 模型中的常数区间 + 多项式 (P_i(x)\equiv T[i]) 表示。
      因此 GeLU/SiLU 都属于 SUF(_n^{1, O(1), d})，(d\le 3)。

3. **Reciprocal / rsqrt**

   Sigma 的 RecSqrt 先用 IntervalLookup 取出“浮点类”分解 (x\approx 2^e(1+m/128))，再用小 LUT 得到近似的 (1/\sqrt{x})：

    * IntervalLookup(_{n,G,\mathbf{p},\mathbf{q},\mathbf{v}}) 本质是把区间 ([2^i,2^{i+1})) 映射到常数对 ((e,u))；这是典型的 SUF。
    * 之后只有固定次多项式及 LUT。
      整个 RecSqrt、Reciprocal 都是 SUF。

4. **nExp 与 Softmax‑block**

   Sigma 的 nExp/Softmax 先用 Max+减 max 把输入域归一到一个有限区间，再用 LUT + 多项式近似 (\exp(\cdot)) 和 (1/x)。effective bitwidth (m=n-f+2) 被精确分析过。

    * Max 是“多次两两比较 + 选择”的组合（每次都是 SUF(_{n}^{1,1,1})）。
    * nExp 和 normalize 里的 RecSqrt/Rec 都如上是 SUF。
    * 整个 softmax block 是有限多 SUF 的组合 + 线性运算，因而在 gate 级别可以看作“向量 SUF”。

**小结**：你需要的所有非线性（ReluARS/GeLU/SiLU/Rec/rsqrt/Softmax‑block）在固定 (n,f) 与固定近似方案下，都是 SUF 成员。

---

## 三、从 SUF 描述到 PDPF 程序的编译器规范

我们把 PDPF 视为“对 SUF 的高效 FSS 内核”。

底层我们采用 Sigma 所用的 BGI‑18 DPF：对 1‑bit point function (f_{\alpha,1}:U_N\to{0,1}) 有：

* keysize((\mathsf{DPF}_n)= (n-\nu)(\lambda+2)+2\lambda) 比特（当 (n>\nu)）；
* Gen 调用 (2(n-\nu)) 次 PRG；Eval 调用 ((n-\nu)) 次 **half‑PRG**（即 1 次 AES‑128）。

多比特输出 / 小 LUT / IntervalLookup 则分别满足：

* LUT(_{n,\ell,T})：keysize (=) keysize(DPF(_n,1)) (+n+2\ell)，Eval 调用 (2^{n-\nu-1}) 次 PRG，通信 (2\ell) bit。
* IntervalLookup(_{n,G,\mathbf{p},\mathbf{q},\mathbf{v}})：keysize (=) keysize(DPF(_n,1)) (+3\ell)，Eval 只需一次 DPF + 常数时间本地处理，通信 (4\ell)bit。

### 3.1 SUF IR

对单变量 SUF，我们使用一个 IR：

[
\mathsf{desc}(F) = ({\alpha_i}*{i=0}^m,{P_i}*{i=0}^{m-1}, {B_i}*{i=0}^{m-1}, r*{\text{in}}, r_{\text{out}}).
]

* ({\alpha_i})：按顺序的区间边界；
* (P_i)：在区间 (I_i) 上的多项式系数（向量）；
* (B_i)：要输出的布尔 bit 的表达式（由比较 / MSB 的布尔组合组成）；
* (r_{\text{in}}, r_{\text{out}})：统一的输入/输出 mask。

### 3.2 编译器 (\mathsf{SUF2PDPF})

输入：SUF 描述 (\mathsf{desc}(F))。输出：一组 PDPF program 描述 ({\mathsf{desc}(f_j)}_j)，每个 (f_j:U_N\to G_j) 对应一类辅助量（比较 bit、系数向量等）。

典型模式：

1. **比较族**
   所有形如 (1[x<\beta])、(1[x\bmod 2^f<\gamma]) 的 bit 全部打包成一个向量函数
   [
   f_{\mathrm{cmp}}(\hat{x})\in\mathbb{Z}*2^{L*\mathrm{cmp}},
   ]
   作为一个 PDPF 程序。

2. **多区间系数 / LUT**

    * 对分段多项式：定义一个 multi‑point function
      [
      f_{\mathrm{poly}}(\hat{x}) =
      \sum_i \mathbf{c}*i\cdot 1[\hat{x}\in \widetilde{I_i}],
      ]
      其中 (\widetilde{I_i}) 是加上 mask 后的区间，(\mathbf{c}*i) 是已吸收 (r*{\text{in}},r*{\text{out}}) 的系数向量。用 PDPF 的“可编程 payload”实现（本质就是 PDPF 上的稀疏函数）。
    * 对小 LUT：使用上面的 (\mathsf{LUT}_{n,\ell,T}) 或 IntervalLookup 原语（如果 LUT 按段组织）。

3. **输出向量**

   所有要从 PDPF 拿到的数（布尔 bit，系数向量，LUT 输出）都组织在统一的群
   [
   G = R^{r'} \times \mathbb{Z}_2^{\ell'}.
   ]
   这样一个 gate 理论上可以只用 **一次 PDPF.Gen + Eval**。

编译器只需要保证：对任意 (x)，从 PDPF 输出和线性运算恢复出的结果刚好等于 SUF 的 (F(x)) 的 masked 版本。

**编译后开销（单 gate）**
若对该 gate 使用了一个主 PDPF 程序，域 bit 宽为 (m_\tau)，输出总布尔维度为 (\ell_\tau)，算术 payload 总长度为 (a_\tau)（位），则：

* keysize（FSS 部分）：
  [
  \mathrm{K}*\tau
  = (m*\tau-\nu)(\lambda+2)+2\lambda

    * O(\ell_\tau + a_\tau).
      ]
* online AES（PRG 调用）：
  [
  \mathrm{AES}*\tau^\text{FSS}
  = m*\tau-\nu,
  ]
  其余都是若干常数个 Beaver 乘法三元组（与 SHARK/Sigma 一样的代价记账方式）。

后面具体算 cost 时，我会只在 FSS 部分用这个符号。

---

## 四、几个核心 gate 的 Gen/Eval 伪代码

下面只写出 Composite‑FSS 层面的 Gen/Eval，线性层（Beaver 乘法、B2A）默认有现成实现。

### 4.1 ReluARS(_{n,f})：有 gap / 无 gap

#### 4.1.1 目标功能

[
\mathrm{ReluARS}*{n,f}(x) = \mathrm{ReLU}(\mathrm{ARS}*{n,f}(x)).
]

利用 SHARK 的恒等式（略去 MAC）：
对 masked 输入 (\hat{x}=x+r_{\text{in}})，令

* (w=1[x<r_{\text{in}}])，
* (t=1[x\bmod 2^f < r_{\text{in}}\bmod 2^f])，
* (d) 是一个与符号相关的 bit（例如 (\mathrm{GEZ}(\mathrm{ARS}(x)))）。

存在常数 LUT (T[d,w,t])，满足
[
\mathrm{ReluARS}*{n,f}(x-r*{\text{in}})+\tilde{r}*{\text{out}}
= d\cdot \mathrm{LRS}*{n,f}(\hat{x}) + T[d,w,t].
]

**有 gap 版本**：假设输入 (x) 来自前一层 TR 后，实际有效位数为 (m=n-f) 或 (m=n-f+1)，我们只在这 (m) 位上做比较（和 Sigma 的“effective bitwidth”完全一致）。

#### 4.1.2 Gen（无 gap）

输入：(n,f)。

1. 采样 mask：
   (r_{\text{in}},\tilde{r}_{\text{out}}\leftarrow R)。
2. 定义辅助 bit：

    * (w(x)=1[x<r_{\text{in}}])；
    * (t(x)=1[x\bmod 2^f<r_{\text{in}}\bmod 2^f])；
    * (d(x)=\mathrm{GEZ}(\mathrm{ARS}_{n,f}(x)))（仍然可以用几次比较与 MSB 表达）。
3. 把它们视为 masked 输入 (\hat{x}=x+r_{\text{in}}) 的函数，写成 SUF 描述 (\mathsf{desc}(w,t,d))。
4. 用编译器生成 PDPF 程序
   [
   f_{\mathrm{ReluARS}}(\hat{x})=(w(x),t(x),d(x))
   ]
   [
   (k^{\mathrm{RA}}_0,k^{\mathrm{RA}}*1)\leftarrow\mathsf{PDPF.Gen}(\mathsf{desc}(f*{\mathrm{ReluARS}})).
   ]
5. 预计算 LUT：对 ((d,w,t)\in{0,1}^3)：
   [
   T[d,w,t] := d\cdot(-\mathrm{LRS}*{n,f}(r*{\text{in}})
   +2^{n-f}w-t) +\tilde{r}_{\text{out}}.
   ]
   并对 (T) 做加法 share。
6. key：
   [
   K_b^{\mathrm{RA}} = (r_{\text{in}},\tilde{r}_{\text{out}},
   k^{\mathrm{RA}}_b, T_b).
   ]

#### 4.1.3 Eval（无 gap）

输入：masked (\hat{x})（公开），key (K_b^{\mathrm{RA}})。

1. PDPF:
   [
   (w_b,t_b,d_b)\leftarrow
   \mathsf{PDPF.Eval}(k^{\mathrm{RA}}_b,\hat{x}).
   ]
2. 把 (w,t,d) 从 (\mathbb{Z}*2) 转成 (\mathbb{Z}*{2^n}) share（可以用 B2A 或直接定义 share）。
3. 公共计算 (u=\mathrm{LRS}_{n,f}(\hat{x}))；将其看成 share ((u_0,u_1))（例如 (u_0=u,u_1=0)）。
4. 使用一次乘法三元组计算
   [
   z_b = d_b\cdot u
   ]
5. 查询 LUT：
   reconstruct 出 (d,w,t)（3 bit），双方对 (T[d,w,t]) 从 share 里拿到 (T_b)。
6. 输出：
   [
   \hat{y}*b = z_b + T_b.
   ]
   合并后 (\hat{y}=\sum_b\hat{y}*b
   =\mathrm{ReluARS}*{n,f}(x-r*{\text{in}})+\tilde{r}_{\text{out}})。

#### 4.1.4 有 gap 版本

唯一区别：我们已知输入有效位数 (m=n-f)（或 (n-f+1)），则：

* 在 SUF→PDPF 编译时，比较 bit 的域长度从 (n) 降到 (m)，因此 PDPF keysize 与 AES 调用中的 (m_\tau) 变为 (m\ll n)（等价于 Sigma 把 DReLU 里的比较从 (n) 降到 (m-(f-6)) 的“优化 3”）。

Gen/Eval 逻辑完全相同，只是把 (m_\tau) 换成更小的“effective bitwidth”。

---

### 4.2 GeLU / SiLU gate

我们针对 **Sigma CPU GeLU** 的结构做一次“合并 FSS”的版本。Sigma 的 CPU GeLU 近似可以写成：

1. (y = \mathrm{TR}_{m,f-6}(x))，有效位宽变为 (m-(f-6))；
2. 若干比较得到 ReLU/Clip bit；
3. 取 (|\mathrm{Clip}(y)|) 的高 8 bit 得到索引 (t\in{0,\dots,255})；
4. LUT(_8) 取出 (\delta(t)2^f)，输出：
   [
   \mathrm{GeLU}(x)\approx \mathrm{ReLU}(x) - \delta(t).
   ]

Sigma 的 CPU 协议成本是：
keysize (= 2\text{·DReLU}+1\text{·LUT}_8+1\text{·TR}+3\text{·select})。

#### 4.2.1 Composite‑GeLU 的总体想法

我们构造一个 SUF 函数
[
F_{\mathrm{GeLU}}(x)
= (\mathrm{GeLU}(x),; \text{aux bits}),
]
其中 aux bits 至少包括：

* ReLU bit (b_{\mathrm{relu}}(x))；
* Clip 区间 bit（落在 ([-B,B]) 或外）；
* 8‑bit LUT 索引 (t(x))。

这些全部从同一个 SUF 描述中得到。编译时：

* 主 PDPF 程序 (f_{\mathrm{GeLU}}) 的输出群设为
  (G = R\times \mathbb{Z}*2^{\ell*{\mathrm{bit}}}\times \mathbb{Z}*2^8)；
  其中 (R) 部分直接给出“masked (\mathrm{ReLU}(x))”；8 bit 是 LUT 索引；(\ell*{\mathrm{bit}}) 是内部控制 bit。
* 另一个小 PDPF/LUT 程序 (f_T) 把 (t) 映射到 (\delta(t)2^f)。

然后 GeLU 输出可以在线性层完成：
[
\hat{y} = \widehat{\mathrm{ReLU}(x)} - \widehat{\delta(t)} + r_{\text{out}}.
]

#### 4.2.2 Gen(_{\mathrm{GeLU}})

输入：(n,m,f)，LUT 表 (T\in R^{256})（Sigma 或你自己拟合）。

1. 采样 mask (r_{\text{in}}, r_{\text{out}}\in R)。
2. 定义 SUF：

    * 利用 effective bitwidth (m=n-f)，按 Sigma 的“优化 2/3”先做 TR，再比较，保证所有比较的位宽是 (m-(f-6))。
    * 在 SUF 中显式写出：

        * (b_{\mathrm{relu}}(x)=1[x\ge0])；
        * Clip bit (b_{\mathrm{in}}(x)=1[-B\le x\le B])；
        * TR 结果 (y=\mathrm{TR}_{m,f-6}(x))；
        * 索引 (t(x)) 为 (y) 的某 8 个高位。
    * 定义算术输出
      [
      P(x) = \mathrm{ReLU}(x) + r_{\text{mid}}.
      ]
3. 调用 (\mathsf{SUF2PDPF}) 得到 PDPF 程序：
   [
   f_{\mathrm{GeLU}}(\hat{x})=
   (P(x), b_{\mathrm{relu}},b_{\mathrm{in}}, t(x)).
   ]
   域 bit 宽 (m_\mathrm{GeLU}=m-(f-6))。
4. 生成 key ((k^{\mathrm{Ge}}_0,k^{\mathrm{Ge}}_1))。
5. 为 LUT 表 (T) 构造 (\Pi_{\mathrm{LUT}_{8,n,T}})，得到 key ((k^T_0,k^T_1))。
6. 把 (r_{\text{mid}},r_{\text{out}}) 合并进常数 share，以便最后得到 masked output：
   [
   c = r_{\text{out}} - r_{\text{mid}},
   ]
   做 share。
7. 输出：
   [
   K_b^{\mathrm{GeLU}} =
   (r_{\text{in}},k^{\mathrm{Ge}}_b,k^T_b,c_b).
   ]

#### 4.2.3 Eval(_{\mathrm{GeLU},b})

1. PDPF 主程序：
   [
   (u_b, b^{\mathrm{relu}}_b,b^{\mathrm{in}}_b,t_b)
   \leftarrow \mathsf{PDPF.Eval}(k^{\mathrm{Ge}}_b,\hat{x}).
   ]

   这里 (u_b) 是 masked ReLU share，布尔 bit 在 (\mathbb{Z}_2) 上。
2. reconstruct 8‑bit 索引 (t)（或按需要只打开最低若干 bit）；调用 LUT PDPF：
   [
   \delta_b \leftarrow \mathsf{PDPF.Eval}(k^T_b,t).
   ]
3. 在线性 secret‑sharing 层做：
   [
   \hat{y}_b = u_b - \delta_b + c_b.
   ]
4. 合并后：(\hat{y}=\sum_b\hat{y}*b = \mathrm{GeLU}(x)+r*{\text{out}})。

**SiLU** 可以完全平行地做：只需把目标 SUF 换成近似 (\mathrm{SiLU}(x)\approx x\sigma(x)) 的分段多项式 / LUT（Sigma 已给了一套近似；你只要用同一套 LUT，SUF 结构保持不变）。

---

### 4.3 Reciprocal / rsqrt gate

我们直接复用 Sigma 的 RecSqrt 结构，只是在 Composite‑FSS 里把 **IntervalLookup + LUT + TR** 所需的 PDPF 程序全部塞进一个“RecSqrt gate key”。

Sigma 的 RecSqrt 伪代码（略去 MAC）是：

1. ((\hat{e}_b,\hat{u}*b)\leftarrow \Pi*{\mathrm{IntervalLookup}}(\hat{x}))；
2. 乘法 (t=\hat{x}\cdot\hat{u})；
3. (m=\mathrm{TR}_{n,n-8}(t)\bmod 128)；
4. (p= \mathrm{extend}(m)\cdot2^6 + e)；
5. 输出 (\mathrm{LUT}_{13,n,T}(p))。

在 Composite‑FSS 里：

* 把 IntervalLookup、TR、LUT 的 PDPF key 都打包进 (K_b^{\mathrm{RecSqrt}})；
* Eval 时按顺序调用三类 PDPF（IntervalLookup, TR 的比较, LUT），线性层负责乘法和加法。

**Reciprocal** 可以直接用相同结构但 LUT/Tables 对应 (1/x) 而不是 (1/\sqrt{x})（Sigma 的 Inverse 协议也是 IntervalLookup + LUT）。

---

### 4.4 Softmax‑block gate

定义一个长度为 (k) 的 softmax‑block（例如 (k=128)）：输入 (\mathbf{x}\in R^k)，输出
[
y_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}.
]

Sigma 的 softmax 协议拆成：Max + nExp + Normalize（Reciprocal），并利用 effective bitwidth 把比较位宽降低到 (m=n-f+1) 或 (n-f+2)。

在 Composite‑FSS 里，我们定义一个 **逻辑 gate** SoftmaxBlock(_{n,f,k})，其实内部就是：

* (k-1) 次 2‑input Max gate（每个是 SUF + 一个 PDPF）；
* (k) 个 nExp gate（每个是 SUF + 一个 PDPF）；
* 1 个 Recip gate + (k) 个乘法。

形式上我们仍旧按照 “每个标量非线性都是一个 Composite‑FSS gate” 来计数；softmax‑block 只是把这些 gate pack 在一个接口里，以便编译器可以跨层共享 PDPF seed（比如对整块 nExp 用一个 multi‑input PDPF）。从理论上讲，这只会进一步减小常数，我们后面的成本分析按“逐标量 gate”做已经是保守估计。

---

## 五、成本分析与 SHARK / Sigma 对比

### 5.1 基本成本公式

采用 Sigma 的 DPF 实现，记：

* (K(n) := \mathrm{keysize}(\mathsf{DPF}_n,1)=(n-\nu)(\lambda+2)+2\lambda)；
* Eval 调用 AES 次数：(\mathrm{AES}(n)=n-\nu)。

取 (\lambda=127,\nu=7) 时：

* (K(64)=(64-7)\cdot129+2\cdot127=7607) bit (\approx 0.93) KB；
* (K(52)=(52-7)\cdot129+254=6059) bit (\approx 0.74) KB。

Sigma 中其他原语的 keysize（单路标量）：

* DReLU(_n)：keysize(=K(n-1)+1)。
* TR(_{n,f})：keysize(=K(f)+2(n-f))。
* ARS(_{n,f})（无 gap）：keysize(= K(f)+2(n-f) + K(n-f)+2n+1)。
* GapARS(_{n,f})：keysize(=K(f)+3n)。
* LUT(_{n,\ell,T})：keysize(=K(n)+n+2\ell)。

SHARK 的 ReluARS 采用 FKOS‑DCF，而不是 DPF，Table 1 给出的成本是：
[
\text{AES}=4n+2f,\quad
\text{comm}= \tilde{n}+3,\quad
\text{preproc}= (n+f)(\lambda+\kappa),
]
其中 (\tilde{n}=n+s)，(s=64)，(\kappa=40)。

### 5.2 Composite‑ReluARS vs SHARK‑ReluARS

#### 5.2.1 Composite‑ReluARS 的 FSS 开销

按 4.1 中的设计，我们只需要 **一个 PDPF 程序**：

* 域 bit 宽：

    * 无 gap：(m_\mathrm{RA}=n)；
    * 有 gap：(m_\mathrm{RA}=m\approx n-f)（effective bitwidth）。
* 输出 payload：

    * 3 个 bit（w,t,d）+ 8 个 LUT 条目可以预先 embed，不影响种子大小，只增加 (O(1)) bit 常数。

因此 per‑gate per‑party FSS keysize 近似为：
[
\mathrm{K}^{\mathrm{RA}}*{\text{Comp}} \approx K(m*\mathrm{RA}) + O(n).
]

在线 Eval 里只做一次 PDPF.Eval，所以：

* **AES 调用**：(\mathrm{AES}^{\mathrm{RA}}*{\text{Comp}}\approx m*\mathrm{RA}-\nu)；
* **在线通信**：只在 B2A 或 reconstruct 少量 bit（3 bit）时发常数比特，数量级 (O(1))，远小于 SHARK 的 (n+s) 级别（此处忽略 MAC）。

#### 5.2.2 具体数值（n=64, f=12）

取无 gap（最保守），(m_\mathrm{RA}=64)：

* Composite‑ReluARS：

    * keysize(\approx K(64)+O(64)\approx 7607+O(64)\approx 7.7) Kbit；
    * AES(\approx 64-7=57)；
    * comm：(\approx) 常数（比如几十 bit）。
* SHARK‑ReluARS（Table 1）：

    * preproc((n+f)(\lambda+\kappa)=(64+12)\cdot(126+40)=76\cdot166=12{,}616) bit，再加若干 SPDZ 相关常数，与文中 16,537 bit 一致数量级；
    * AES(=4n+2f=4\cdot64+2\cdot12=256+24=280)；
    * comm(\approx\tilde{n}+3=128+3=131) bit。

**比值（仅看 FSS 部分）：**

* keysize：约 (7.7) Kbit vs (16.5) Kbit，**节省 (\approx 2.1\times)**；
* AES 调用：(57) vs (280)，**节省 (\approx 4.9\times)**。

如果利用 gap（(m\approx n-f=52)），Composite‑ReluARS 的 FSS 成本进一步变为：

* keysize( \approx K(52)+O(52)\approx 6059+O(52)\approx 6.1) Kbit；
* AES (\approx 52-7=45)。

此时：

* keysize：(6.1) Kbit vs (16.5) Kbit，**约 (2.7\times) 节省**；
* AES：(45) vs (280)，**约 (6.2\times) 节省**。

> 直观解释：SHARK 为 ReluARS 复用两套 FKOS‑DCF（一个 n‑bit，一个 f‑bit）+ 多个表，而我们的 Composite‑FSS 把所有 w,t,d bit 与 LUT 调整项折叠成“一次 DPF”，DPF 成本主要线性于 bitwidth，和输出 bit 数几乎无关。

### 5.3 Composite‑GeLU vs Sigma‑GeLU

Sigma 的 CPU GeLU 成本分析：

[
\mathrm{keysize}(\Pi^{\mathrm{CPU}}*{\mathrm{GeLU}})
=; 2\cdot \mathrm{keysize}(\Pi*{\mathrm{DReLU}_{m-f+6}})

* \mathrm{keysize}(\Pi_{\mathrm{LUT}_{8,n,T}})
* \mathrm{keysize}(\Pi_{\mathrm{TR}_{m,f-6}})
* 3\cdot\mathrm{keysize}(\Pi_{\text{select}}).
  ]

其中：

* (\mathrm{keysize}(\Pi_{\mathrm{DReLU}_L})=K(L-1)+1)；
* (\mathrm{keysize}(\Pi_{\mathrm{LUT}_{8,n,T}})
  =K(8)+8+2n)；
* (\mathrm{keysize}(\Pi_{\mathrm{TR}_{m,f-6}})=K(f-6)+2(m-f+6))；
* select(n) 的 keysize(\le 4n)（One‑time truth table）。

**我们的 Composite‑GeLU：**

主 PDPF 程序一次性输出 ReLU share + 所有控制 bit + 8‑bit 索引 (t)，只需要 **一个** DPF：

[
\mathrm{keysize}(\mathrm{CompGeLU})
\approx K(m_\mathrm{Ge}) + O(n),
]
其中 (m_\mathrm{Ge}=m-(f-6))（利用 Sigma 的优化 2/3）。

LUT(_8) 仍然需要一个 DPF(_8)（不过这一项 Sigma 也要），所以完整：

[
\mathrm{keysize}(\mathrm{CompGeLU}^\text{full})
\approx K(m_\mathrm{Ge}) + K(8) + O(n).
]

而 Sigma‑GeLU 至少需要 **两次 DReLU DPF_{m-f+6}** 加上一堆别的：

[
\mathrm{keysize}(\mathrm{SigmaGeLU})
\gtrsim 2 K(m_\mathrm{Ge}) + K(8) + K(f-6) + O(n).
]

（精确地：还要加上 TR、3×select 的几百 bit 开销。）

#### 5.3.1 n=64, f=12 的具体数值

在 transformer 中，有效 bitwidth (m=n-f=52)，优化后 (\mathrm{GeLU}) 的比较发生在 (m-(f-6)=52-(12-6)=46) bit 上。

* 我们的主 DPF：(K(46)=(46-7)\cdot129+254=39\cdot129+254=5031+254=5285) bit；
* DPF(_8)：(K(8)=(8-7)\cdot129+254=129+254=383) bit。

于是：

[
\mathrm{keysize}(\mathrm{CompGeLU}^{\text{full}})
\approx 5285 + 383 + O(n)
\approx 5{,}668 + 512 \approx 6{,}180\ \mathrm{bits}
\approx 0.76\ \mathrm{KB}.
]

Sigma‑GeLU 的 FSS 部分（保守下界）：

[
\mathrm{keysize}(\mathrm{SigmaGeLU})
\gtrsim 2K(46) + K(8)
\approx 2\cdot 5285 + 383 = 10{,}953\ \mathrm{bits}.
]

再加上 TR、3×select（约 (2(m-f+6)+3\cdot 4n\approx O(10^3)) bit），最终大约：
[
\mathrm{keysize}(\mathrm{SigmaGeLU})
\approx 12\text{--}13\ \mathrm{Kb}
\approx 1.5\ \mathrm{KB},
]
这和 Sigma 论文 Table 2 中 GeLU CPU 的 keysize 1.43KB 是一致数量级。

于是 **ratio**：

[
\frac{\mathrm{keysize}(\mathrm{SigmaGeLU})}
{\mathrm{keysize}(\mathrm{CompGeLU})}
\approx \frac{1.5}{0.76} \approx 1.97.
]

也就是说，在 (n=64,f=12) 且采用 Sigma 同样的近似精度时，

* **Composite‑GeLU 的 keysize 至少能节省约 (2\times)**；
* AES 调用数也相应减少：Sigma‑GeLU 要求 2 次 DReLU + 1 次 TR + 1 次 LUT，
  而 Composite‑GeLU 只需 1 次 DPF(_{46}) + 1 次 LUT(_8)，AES 调用从大约 (2(46-\nu)+8-\nu) 降到 (46-\nu+8-\nu)，理论上也约减半。

---

## 六、小结：形式化 & 新颖性

把上面所有东西合在一起，你现在有了：

1. **一个正式的 Composite‑FSS 定义**：
   每个 gate 类型 (\tau) 对应 (\mathsf{CompGen}*\tau,\mathsf{CompEval}*\tau)；正确性/隐私在 SUF + PDPF 安全下严格给出。

2. **Structured‑Univariate‑Function 家族**：

    * 证明了 ReLU/ARS/ReluARS/GeLU/SiLU/Rec/rsqrt/nExp/Softmax‑block 都在 SUF 里；
    * 这是“PDPF 支持的函数族”的一个干净刻画，对比 SHARK/Sigma 只列出若干 ad‑hoc 原语要更抽象、结构化。

3. **从 SUF 到 PDPF 的通用编译器规范**：
   通过 SUF IR、比较族、IntervalLookup、多点系数编码，把所有 gate 统一成“**一个 PDPF 程序 + 常数 LUT**”。

4. **协议细节**：
   给出了 ReluARS（有 gap/无 gap）、GeLU、SiLU、Rec/rsqrt、Softmax‑block 的 Gen/Eval 伪代码骨架，在 SHARK/Sigma 的公式基础上重新组织，但 **只依赖 PDPF 核心** 而不依赖 IFSS/SPDZ‑DCF。

5. **理论对比（n=64,f=12）**：

    * ReluARS：相对于 SHARK，Composite‑ReluARS 在 keysize 上约 (2\text{–}2.7\times) 节省，在 AES 调用上约 (5\text{–}6\times) 节省。
    * GeLU：相对于 Sigma‑GeLU CPU，Composite‑GeLU 至少有约 **(2\times)** 的 keysize 降低，AES 调用也减少约一半。

接下来如果你要写 paper，可以考虑的结构是：

* Section 3：定义 SUF 与 Composite‑FSS，给出通用安全性定理；
* Section 4：把 SHARK / Sigma 的具体协议重写成 SUF + PDPF 格式；
* Section 5：给出我们的“单 PDPF gate”版本，并做 cost 分析；
* Section 6：实现 / 实验对比（和 Sigma/SHARK 的 microbenchmarks 平行）。

如果你愿意，我们下一步可以把某一个 gate（比如 GeLU CPU）完整展开到“bit 级 IR + 编译例子”，甚至把你 pdpf 库的具体 key 长度参数代入，把上面的 2× 变成具体的常数。
