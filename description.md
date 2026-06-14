你是一个资深软件架构师、符号计算系统开发者、桌面应用开发者，并且熟悉基础数学、高等数学、张量代数、连续介质力学和有限变形理论。请帮助我设计并逐步实现一个名为 **TensorForge** 的桌面端符号计算软件。

# 1. 项目定位

我要开发一个可通过 **Homebrew** 发行的 **desktop app**，名字暂定为：

**TensorForge**

它不是普通计算器，也不是单纯的连续介质力学小工具，而是一个面向：

- 基础数学符号运算
- 高等数学符号运算
- 张量代数
- 连续介质力学张量运算

的专用符号计算系统。

最终目标是做成一个能够严格、可验证地处理连续介质力学张量推导的软件，尤其是二阶张量、四阶张量、张量对张量求导、指标缩并检查、张量表达式化简和 LaTeX / Markdown 显示。

核心计算必须是严格符号运算，不能依赖 AI 猜测。

# 2. 软件形态

TensorForge 应该是一个 desktop app，并支持 Homebrew 安装和发行。

交互方式优先考虑：

- 公式编辑器风格
- 左侧输入 DSL / 公式表达式
- 右侧显示数学排版结果
- 支持 LaTeX / Markdown 渲染
- 支持手动展开张量分量
- 支持保存为自定义文本格式

保存文件扩展名建议为：

```text
.tens
```

例如：

```text
start.tens
```

# 3. DSL 语法设计

TensorForge 使用函数式声明语法，而不是直接使用 LaTeX 或 Python。

基本风格如下：

```text
mu = Scalar("\mu")
lambda = Scalar("\lambda")

F = Tensor("\bm F", order=2, dim=3)
I = Tensor("\bm I", order=2, dim=3, identity=true)
```

这种语法需要满足：

- 易解析
- 易扩展
- 数学对象类型明确
- 适合符号计算
- 适合显示为 LaTeX / Markdown

# 4. 数学对象模型

软件需要明确区分以下对象：

```text
Scalar
Vector
Matrix
Tensor
Expression
Function
Derivative
```

其中：

- Matrix 和 Tensor 必须区分
- Vector 和一阶 Tensor 必须区分
- 四阶张量不需要单独成为独立类，可以作为 Tensor(order=4)
- 张量属性作为 Tensor 的 metadata 或 properties

张量对象应支持以下属性：

```text
order
dim
name
latex_display
identity
symmetric
antisymmetric
orthogonal
isotropic
```

示例：

```text
C = Tensor("\bm C", order=2, dim=3, symmetric=true)
Q = Tensor("\bm Q", order=2, dim=3, orthogonal=true)
I = Tensor("\bm I", order=2, dim=3, identity=true)
A = Tensor("\mathbb A", order=4, dim=3)
```

# 5. 张量显示规则

二阶张量默认显示为粗体符号，例如：

```latex
\bm C
```

四阶张量默认显示为黑板粗体或类似符号，例如：

```latex
\mathbb A
```

软件不应该默认自动展开张量分量，因为展开结果可能很大。

需要通过显式函数手动展开：

```text
display(C, mode=symbol)
display(C, mode=components)
display(C, mode=matrix)
display(A, mode=symbol)
display(A, mode=block_components)
```

# 6. 四阶张量显示方式

四阶张量不优先使用 Voigt notation。

对于四阶张量：

```latex
\mathbb A = A_{iJkL}
```

希望支持将其按照后两个指标 `k,L` 展开为 9 个二阶张量块：

```text
display(A, mode=block_components)
```

显示结构类似：

```latex
\mathbb A =
\begin{bmatrix}
\bm A^{11} & \bm A^{12} & \bm A^{13}\\
\bm A^{21} & \bm A^{22} & \bm A^{23}\\
\bm A^{31} & \bm A^{32} & \bm A^{33}
\end{bmatrix}
```

其中每一个块都是一个二阶张量：

```latex
\bm A^{kL}_{ij} = A_{ijkL}
```

这个显示方式用于保持真实四阶张量结构，而不是过早压缩成 Voigt 矩阵。

# 7. 运算符约定

TensorForge 的张量运算应支持以下运算：

```text
*        标量乘法、二阶张量乘法
:        双指标缩并的显示符号
&        张量积
dot()    单指标缩并
ddot()   双指标缩并
contract(A, B, indices=...) 一般缩并
```

内部语法可以支持：

```text
C = F.T * F
A = F & F
s = ddot(A, B)
R = contract(A, B, indices=...)
```

但数学显示时优先显示：

```latex
\bm A \bm B
```

```latex
\bm A : \bm B
```

```latex
\bm A \otimes \bm B
```

其中 `A & B` 显示为：

```latex
\bm A \otimes \bm B
```

# 8. 张量表达式示例

TensorForge 应支持如下表达式：

```text
mu = Scalar("\mu")
lambda = Scalar("\lambda")

F = Tensor("\bm F", order=2, dim=3)
I = Tensor("\bm I", order=2, dim=3, identity=true)

C = F.T * F
J = det(F)
I1 = tr(C)

W = mu/2 * (I1 - 3) - mu * log(J) + lambda/2 * log(J)^2

dCdF = diff(C, F)
dJdF = diff(J, F)
P = diff(W, F)
A = diff(P, F)

C = simplify(C, rules=continuum)
P = simplify(P, rules=continuum)
A = simplify(A, rules=continuum)

display(C, mode=symbol)
display(C, mode=components)
display(dCdF, mode=components)
display(dJdF, mode=symbol)
display(P, mode=symbol)
display(A, mode=block_components)

export(A, format=latex)
export(A, format=markdown)
```

# 9. 自动属性推断

软件应支持严格的自动属性推断。

例如：

```text
C = F.T * F
```

软件应该在可严格证明的情况下推断：

```text
C.order = 2
C.dim = 3
C.symmetric = true
```

但属性推断必须保守：

- 只能在数学上严格成立时自动推断
- 不能因为两个张量都是对称张量，就错误推断它们的乘积也是对称张量
- 用户手动声明属性应作为已知条件进入化简系统

# 10. 求导规则

TensorForge 必须支持以下类型的符号求导：

```text
diff(scalar, scalar)
diff(scalar, vector)
diff(scalar, tensor)
diff(vector, vector)
diff(tensor, tensor)
diff(second_order_tensor, second_order_tensor)
diff(scalar_energy, second_order_tensor)
diff(second_order_stress, second_order_tensor)
```

核心目标包括：

```latex
\frac{\partial C_{ij}}{\partial F_{mn}}
```

```latex
\frac{\partial J}{\partial F_{ij}}
```

```latex
\frac{\partial \bm C}{\partial \bm F}
```

```latex
\frac{\partial \bm S}{\partial \bm C}
```

```latex
\frac{\partial^2 W}{\partial \bm C \partial \bm C}
```

```latex
\frac{\partial \bm P}{\partial \bm F}
```

若：

```text
P = diff(W, F)
A = diff(P, F)
```

且 `P` 和 `F` 都是二阶张量，则 `A` 是四阶张量。

默认指标顺序约定为：

```latex
A_{iJkL} = \frac{\partial P_{iJ}}{\partial F_{kL}}
```

也就是：

```text
diff(P[i,J], F[k,L]) -> A[i,J,k,L]
```

规则原则：

```text
结果指标 = 被求导对象指标 + 求导变量指标
```

# 11. 连续介质力学化简规则

TensorForge 需要内置一整套 continuum mechanics simplification rules。

例如：

```latex
\bm F^{-T}\bm F^T = \bm I
```

```latex
\det(\bm F)\bm F^{-T} = \operatorname{cof}\bm F
```

```latex
\frac{\partial J}{\partial \bm F} = J\bm F^{-T}
```

```latex
\operatorname{tr}(\bm A\bm B) = \operatorname{tr}(\bm B\bm A)
```

```latex
\bm C^{-1}\bm C = \bm I
```

```latex
\bm C = \bm F^T\bm F
```

```latex
J = \det \bm F
```

```latex
I_1 = \operatorname{tr}\bm C
```

软件应允许用户调用：

```text
simplify(expr)
simplify(expr, rules=algebra)
simplify(expr, rules=tensor)
simplify(expr, rules=continuum)
```

其中 `rules=continuum` 使用连续介质力学相关规则。

# 12. 指标缩并

用户一般不希望直接用指标记号进行主要计算，而是希望用粗体张量符号进行计算。

但是软件内部必须能够正确处理指标、缩并和分量展开。

需要支持：

- 检查重复指标
- 检查自由指标
- 检查指标维度
- 防止非法三重重复指标
- 在分量显示时生成正确的指标表达式
- 在张量对张量求导时生成正确的 Kronecker delta 表达式

例如：

```text
C = F.T * F
dCdF = diff(C, F)
display(dCdF, mode=components)
```

应能得到类似：

```latex
\frac{\partial C_{ij}}{\partial F_{mn}}
= \delta_{im}F_{nj} + \delta_{jm}F_{ni}
```

具体指标命名可由软件内部统一管理。

# 13. 谱分解

暂时不需要支持曲线坐标、协变/逆变指标、Christoffel 符号、流形等微分几何功能。

谱分解不作为内置的 opaque 对象,而是用 set 显式手写(`spectral(C)` 内置函数已移除):

```text
lambda = ScalarSet("\lambda", dim=3)
N = VectorSet("\bm N", dim=3)
C = sum(lambda[a]^2 * N[a] & N[a], a)
```

```latex
\bm C = \sum_{a=1}^3 \lambda_a^2 \, \bm N_a \otimes \bm N_a
```

基于谱分解的张量函数(`sqrt(C)`、`log(C)` 等)同样用 set 形式显式书写,
例如 `sum(log(lambda[a]) * N[a] & N[a], a)`。

# 14. 不需要优先考虑的功能

以下功能暂时不是核心：

- 内置完整标准连续介质力学变量体系
- 自动生成所有 kinematic variables
- 坐标系和基底
- 曲线坐标
- 协变/逆变张量
- Christoffel 符号
- 有限元代码生成
- Abaqus UMAT / UEL 代码生成
- FEniCS / Firedrake 代码生成

未来可以扩展，但当前重点是：

```text
基础符号系统 + 张量对象系统 + 连续介质力学化简 + 二阶/四阶张量求导 + 显示
```

# 15. 目标示例任务

TensorForge 最终应能完成以下代表性任务。

输入：

```text
mu = Scalar("\mu")
lambda = Scalar("\lambda")

F = Tensor("\bm F", order=2, dim=3)
I = Tensor("\bm I", order=2, dim=3, identity=true)

C = F.T * F
J = det(F)
I1 = tr(C)

W = mu/2 * (I1 - 3) - mu * log(J) + lambda/2 * log(J)^2

dCdF = diff(C, F)
dJdF = diff(J, F)
P = diff(W, F)
A = diff(P, F)

display(C, mode=symbol)
display(C, mode=components)
display(dCdF, mode=components)
display(dJdF, mode=symbol)
display(P, mode=symbol)
display(A, mode=block_components)

export(P, format=latex)
export(A, format=markdown)
```

期望输出：

1. 自动判断 `C = F.T * F` 是对称二阶张量；
2. 求出 `∂C_ij / ∂F_mn`；
3. 求出 `∂J / ∂F`；
4. 求出一阶 Piola 应力 `P = ∂W / ∂F`；
5. 求出材料切线 `A = ∂P / ∂F`；
6. 用 `\bm C`、`\bm F`、`\mathbb A` 显示张量；
7. 将四阶张量 `A_{iJkL}` 展开为 9 个二阶张量块；
8. 输出 LaTeX 和 Markdown 结果。

# 16. 技术路线要求

请比较以下技术路线，并给出推荐方案：

1. Rust + Tauri
2. TypeScript + Electron
3. Python + Qt
4. Julia + GUI
5. C++ / Rust 核心 + 桌面前端
6. 基于现有符号库，如 SymPy / Symbolics.jl，还是自研符号引擎

比较时需要考虑：

- Homebrew 发行便利性
- 桌面 App 体验
- 跨平台能力
- 打包体积
- 数学符号系统自由度
- 张量符号系统扩展性
- 长期维护成本
- 性能
- LaTeX / Markdown 渲染能力
- 是否适合自己实现 DSL 和解释器

请给出推荐架构。

# 17. 架构偏好

目前不强制拆成独立 core 和 frontend 两个仓库，但内部代码结构应该清晰。

建议至少包含：

```text
parser
ast
symbolic_engine
tensor_engine
simplifier
continuum_rules
differentiation
renderer
desktop_ui
file_io
exporter
```

即使在一个项目中，也要保持模块清晰。

# 18. 需要你输出的内容

请基于以上需求，输出：

1. TensorForge 的完整产品定义；
2. 推荐技术路线；
3. 总体架构设计；
4. DSL 语法规范；
5. AST 设计；
6. 数学对象类型系统设计；
7. 张量对象系统设计；
8. 张量运算规则；
9. 求导规则；
10. 连续介质力学化简规则；
11. LaTeX / Markdown 渲染规则；
12. `.tens` 文件格式设计；
13. 代表性示例 `start.tens`；
14. 初始开发里程碑；
15. 可以执行的第一阶段实现计划；
16. 第一阶段最小可运行原型的代码结构；
17. 后续扩展路线图。

请优先保证设计严谨、可实现、可扩展，不要只给空泛概念。
