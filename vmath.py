# -*- coding: utf-8 -*-
"""
vmath.py —— Computational LSNet 的向量表达式求值器
=================================================
设计目标（相对 more_math 的 Float math / String math）：
  * V0..V9 / F0..F3 全部是 "*"(Any) 端口 —— 可以直接连向量(TENSOR)、标量(FLOAT/INT)、字符串；
    more_math 的 V 是 String、F 是 Float，向量只能塞进 stack，这是本模块要解决的问题。
  * 表达式语言是 Python 数学语法的安全子集（ast 白名单，不用 eval），支持多语句 + 赋值 + 注释。
  * 张量与标量自然地混合运算（torch 广播），返回值末尾语句的值。

支持（大小写不敏感）：
    变量   a b c d w x y z  = V0..V7（F0..F3 可覆盖 w x y z）；V0..V9；V(所有已连接 Vn 的列表)；Vcnt
    常量   pi, e, inf, nan
    运算   + - * / // % ** @ (矩阵乘) -(负) <,>,==,!=,<=,>=  and/or/not  条件表达式 a if c else b
    取值   x[i]  x[i:j]  x.shape  x.shape[0]  len(x)
    函数   见 FUNCTIONS（含 norm/normalize/cos_sim/dot/softmax/topk/argmax/clip/lerp/remap/
           mean/std/sum/var/cat/stack/reshape/permute/index/slice/shape/numel ...）
"""

import ast
import math
import operator as _op

import torch

__all__ = ["evaluate", "run_expression", "parse_vector_text", "to_tensor", "describe"]

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------------------
# 基础类型处理
# --------------------------------------------------------------------------------------
def to_tensor(x, dtype=torch.float32):
    """把任意输入转成 float32 torch.Tensor（不做形状规范化）。"""
    if torch.is_tensor(x):
        return x.detach().to(dtype=dtype)
    if isinstance(x, bool):
        return torch.tensor(float(x), dtype=dtype)
    if isinstance(x, (int, float)):
        return torch.tensor(float(x), dtype=dtype)
    if isinstance(x, (list, tuple)):
        items = [to_tensor(v, dtype) for v in x]
        if not items:
            return torch.zeros(0, dtype=dtype)
        return torch.stack([i.reshape(-1) for i in items]) if any(i.ndim > 0 for i in items) \
            else torch.stack(items)
    raise TypeError(f"无法把类型 {type(x).__name__} 当作数值/向量使用")


def _num(x):
    """尽力转成 python 标量。"""
    if torch.is_tensor(x):
        if x.numel() == 1:
            return float(x.reshape(-1)[0])
        raise TypeError("需要一个标量，但拿到的是长度 %d 的向量（可用 mean()/norm()/sum() 先降维）" % x.numel())
    return x


def _flat(x):
    return to_tensor(x).reshape(-1)


def _t(*args):
    """把 1 个或多个输入统一成同设备张量列表。"""
    return [to_tensor(a) for a in args]


# --------------------------------------------------------------------------------------
# 函数库
# --------------------------------------------------------------------------------------
def _elementwise(fn):
    def g(x, *a):
        return fn(to_tensor(x)) if not a else fn(to_tensor(x), *[_num(i) for i in a])
    return g


def _reduce(red, pair):
    def g(*args):
        if len(args) == 1:
            return red(_flat(args[0]))
        out = to_tensor(args[0])
        for a in args[1:]:
            out = pair(out, to_tensor(a))
        return out
    return g


def _binary(fn):
    def g(a, b):
        return fn(to_tensor(a), to_tensor(b))
    return g


def _combine(args, divide):
    """sum()/mean() 的语义：
       单个张量  -> 全元素归约成标量（sum(x) 求和, mean(x) 求均值）
       多个参数或列表（如 V = 所有已连接 Vn）-> 逐元素相加/平均，保持向量维度
    """
    items = []
    for a in args:
        if isinstance(a, (list, tuple)):
            items.extend(list(a))
        else:
            items.append(a)
    if not items:
        return torch.zeros(())
    if len(items) == 1:
        v = _flat(items[0])
        s = torch.sum(v)
        return (s / v.numel()) if (divide and v.numel()) else s
    tensors = [to_tensor(i) for i in items]
    out = tensors[0].clone()
    for t in tensors[1:]:
        out = out + t
    return out / len(tensors) if divide else out


def _norm(x, p=2.0):
    v = _flat(x)
    if float(p) == 2.0:
        return float(torch.linalg.vector_norm(v))
    return float(torch.linalg.vector_norm(v, ord=float(p)))


def _normalize(x, p=2.0):
    v = to_tensor(x)
    n = torch.linalg.vector_norm(v.reshape(-1), ord=float(p))
    return v / n.clamp_min(1e-30)


def _snorm(x):
    v = _flat(x)
    return (v - v.min()) / (v.max() - v.min()).clamp_min(1e-30)


def _zscore(x):
    v = _flat(x)
    return (v - v.mean()) / v.std(unbiased=False).clamp_min(1e-30)


def _dot(a, b):
    return float(torch.dot(_flat(a), _flat(b)))


def _cos_sim(a, b):
    a2, b2 = _flat(a), _flat(b)
    return float(torch.dot(a2, b2) / (a2.norm() * b2.norm()).clamp_min(1e-30))


def _dist(a, b, p=2.0):
    return float(torch.linalg.vector_norm(_flat(a) - _flat(b), ord=float(p)))


def _softmax(x, T=1.0):
    return torch.softmax(_flat(x) / float(T), dim=-1)


def _topk(x, k):
    k = max(1, int(_num(k)))
    return torch.topk(_flat(x), k=min(k, _flat(x).numel())).values


def _topk_ind(x, k):
    k = max(1, int(_num(k)))
    return torch.topk(_flat(x), k=min(k, _flat(x).numel())).indices.to(torch.float32)


def _argmax(x, dim=None):
    v = to_tensor(x)
    if v.numel() == 1:
        return 0
    if dim is None:
        return int(torch.argmax(v.reshape(-1)))
    return torch.argmax(v, dim=int(_num(dim))).to(torch.float32)


def _cat(*args):
    parts = []
    for a in args:
        if isinstance(a, (list, tuple)):
            parts.extend(_flat(v) for v in a)
        else:
            parts.append(_flat(a))
    return torch.cat(parts, dim=0)


def _stack(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = tuple(args[0])
    parts = [_flat(a) for a in args]
    if len({p.numel() for p in parts}) != 1:
        return torch.cat(parts, dim=0)          # 长度不一致时退化为拼接
    return torch.stack(parts, dim=0)


def _shape(x):
    return list(to_tensor(x).shape)


def _reshape(x, *shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    return to_tensor(x).reshape(tuple(int(_num(s)) for s in shape))


def _index(x, i):
    return to_tensor(x)[int(_num(i))]


def _slice(x, i, j):
    return to_tensor(x)[int(_num(i)):int(_num(j))]


def _permute(x, *order):
    if len(order) == 1 and isinstance(order[0], (list, tuple)):
        order = tuple(order[0])
    return to_tensor(x).permute(tuple(int(_num(o)) for o in order))


def _lerp(a, b, t):
    t = float(_num(t))
    return to_tensor(a) * (1.0 - t) + to_tensor(b) * t


def _remap(x, lo, hi):
    v = _flat(x)
    lo, hi = float(_num(lo)), float(_num(hi))
    return (v - lo) / (hi - lo) if hi != lo else v * 0.0


def _where(c, a, b):
    cc = to_tensor(c) > 0 if not torch.is_tensor(c) else to_tensor(c) > 0
    return torch.where(cc, to_tensor(a), to_tensor(b))


def _rand_like(dist, args):
    """randn / randu 的统一实现（随机分布）。

    参数形式（与 more_math 的 randn(seed,[shape]) 习惯一致）：
      randn([S,D])         无随机种：用全局 RNG，每次调用都不同
      randn(S, D)          无随机种，逐维给尺寸
      randn(seed, [S,D])   有随机种：同一 seed 每次得到同一张张量（可复现）
    有随机种时用独立的 torch.Generator，不污染全局 RNG 状态。
    """
    args = list(args)
    seed = None
    if len(args) == 2 and isinstance(args[1], (list, tuple)):
        seed, args = args[0], list(args[1])
    elif len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = list(args[0])
    shape = tuple(max(0, int(_num(s))) for s in args)
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(_num(seed)) % (2 ** 63 - 1))
    fn = torch.randn if dist == "randn" else torch.rand
    return fn(shape, generator=gen) if shape else fn((), generator=gen)


def _randn(*args):
    return _rand_like("randn", args)


def _randu(*args):
    return _rand_like("randu", args)


def _seedmix(seed, k=0):
    """基准随机种 × 轮次 -> 新随机种（同一 base+轮次 永远得到同一值，保证每轮噪声不同且可复现）。

    用途：遗传算法里每轮迭代需要一个"新的、但仍然可复现"的随机种，
          写法：randn(seedmix(本节点随机种, 迭代次数), [S,D])
    """
    s = int(_num(seed)) & 0x7FFFFFFFFFFFFFFF
    kk = int(_num(k)) & 0x7FFFFFFFFFFFFFFF
    x = (s * 6364136223846793005 + kk * 1442695040888963407 + 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF
    x ^= x >> 29
    x = (x * 0xBF58476D1CE4E5B9) & 0x7FFFFFFFFFFFFFFF
    x ^= x >> 27
    return int(x)


def _rownorm(x, p=2.0):
    """每行（最后一维）的范数，保留维度：[N,D] -> [N,1]（可直接与 [N,D] 广播）。"""
    v = to_tensor(x)
    if v.ndim <= 1:
        return torch.linalg.vector_norm(v.reshape(-1)).reshape(1)
    return torch.linalg.vector_norm(v, ord=float(p), dim=-1, keepdim=True)


def _normclip(x, lo=0.5, hi=2.0, ref=None):
    """逐行把范数钳制到 [lo*ref, hi*ref]：只改模长、不改方向（对应权重版的 clamp）。

    ref 是参考范数（标量，通常取初始画风向量的 norm）；缺省时用第一行的范数。
    """
    v = to_tensor(x)
    n = _rownorm(v)
    r = float(_num(ref)) if ref is not None else float(n.reshape(-1)[0])
    target = torch.clamp(n, float(_num(lo)) * r, float(_num(hi)) * r)
    return v * (target / n.clamp_min(1e-30))


def _ints(s, first=1):
    """'1,2,4' -> tensor([0,1,3])：序号串 -> 下标张量（默认 1 起始，转成 0 起始）。"""
    import re
    if torch.is_tensor(s) or isinstance(s, (bool, int, float)):
        s = str(_num(s))
    toks = [t for t in re.split(r"[,;，、\s]+", str(s).strip()) if t]
    vals = []
    for t in toks:
        try:
            vals.append(float(t) - float(_num(first)))
        except ValueError:
            pass
    if not vals:
        raise ValueError(f"序号串里没有可解析的数字: {s!r}（例如 '1,2,4'）")
    return torch.tensor(vals, dtype=torch.float32)


def _reprows(x, n):
    """向量 -> [n, D]：把初始画风向量复制成 n 行（初代种群：同一初始条件复制 种群数量 份）。"""
    v = to_tensor(x).reshape(1, -1)
    return v.repeat(max(1, int(_num(n))), 1)


def _take(x, idx):
    """按行取值：take(M, 3) -> 第 3 行；take(M, [0,1,3]) -> 选中 3 行（下标自动取整）。"""
    v = to_tensor(x)
    i = to_tensor(idx)
    if i.numel() == 1:
        return v[int(_num(i))]
    return v[i.reshape(-1).long()]


def _arange(n, start=0.0, step=1.0):
    """arange(n) -> tensor([0,1,...,n-1])（float32，常用于做行掩码：reshape(arange(P),[P,1]) == 0）。"""
    a, s = float(_num(start)), float(_num(step))
    return torch.arange(a, a + float(_num(n)) * s, s, dtype=torch.float32)


def _setrow(M, i, v):
    """把矩阵第 i 行整行替换成 v（浮点/负下标自动取整，返回新张量，不改原张量）。

    初代种群用法：setrow(normclip(reprows(f, P) + σ*randn(...)), 0, f)
                 —— 全体变异，再把第一行还原成初始向量（与 v1.0 的 A[0]=初始权重 一致）。
    """
    m = to_tensor(M)
    out = m.clone()
    if out.ndim == 1:
        out = out.reshape(1, -1)
    idx = int(_num(i)) % out.shape[0]
    val = to_tensor(v).reshape(-1)
    if val.numel() != out.shape[1]:
        raise ValueError(f"setrow: 第 {idx} 行需要 {out.shape[1]} 个数，收到 {val.numel()} 个")
    out[idx] = val
    return out


def _catrows(*args):
    """按行拼接（保持 2 维结构）：catrows(A[Q,D], B[C,D]) -> [Q+C, D]。"""
    parts = []
    for a in args:
        if isinstance(a, (list, tuple)):
            parts.extend(to_tensor(v) for v in a)
        else:
            parts.append(to_tensor(a))
    return torch.cat(parts, dim=0)


def _clip(x, lo, hi):
    return torch.clamp(to_tensor(x), float(_num(lo)), float(_num(hi)))


FUNCTIONS = {
    # 一元
    "abs": lambda x: torch.abs(to_tensor(x)),
    "neg": lambda x: -to_tensor(x),
    "sqrt": lambda x: torch.sqrt(to_tensor(x).clamp_min(0)),
    "square": lambda x: to_tensor(x) ** 2,
    "exp": lambda x: torch.exp(to_tensor(x)),
    "ln": lambda x: torch.log(to_tensor(x).clamp_min(1e-30)),
    "log": lambda x: torch.log(to_tensor(x).clamp_min(1e-30)),
    "log10": lambda x: torch.log10(to_tensor(x).clamp_min(1e-30)),
    "log2": lambda x: torch.log2(to_tensor(x).clamp_min(1e-30)),
    "sign": lambda x: torch.sign(to_tensor(x)),
    "floor": lambda x: torch.floor(to_tensor(x)),
    "ceil": lambda x: torch.ceil(to_tensor(x)),
    "round": lambda x, d=0: torch.round(to_tensor(x) * 10 ** int(_num(d))) / 10 ** int(_num(d)),
    "trunc": lambda x: torch.trunc(to_tensor(x)),
    "relu": lambda x: torch.relu(to_tensor(x)),
    "sigmoid": lambda x: torch.sigmoid(to_tensor(x)),
    "sigm": lambda x: torch.sigmoid(to_tensor(x)),
    "tanh": lambda x: torch.tanh(to_tensor(x)),
    "sin": lambda x: torch.sin(to_tensor(x)),
    "cos": lambda x: torch.cos(to_tensor(x)),
    "tan": lambda x: torch.tan(to_tensor(x)),
    "asin": lambda x: torch.asin(to_tensor(x).clamp(-1, 1)),
    "acos": lambda x: torch.acos(to_tensor(x).clamp(-1, 1)),
    "atan": lambda x: torch.atan(to_tensor(x)),
    "sinh": lambda x: torch.sinh(to_tensor(x)),
    "cosh": lambda x: torch.cosh(to_tensor(x)),
    "erf": lambda x: torch.erf(to_tensor(x)),
    "frac": lambda x: to_tensor(x) - torch.floor(to_tensor(x)),
    "flatten": lambda x: _flat(x),
    "sort": lambda x: torch.sort(_flat(x)).values,
    "argsort": lambda x: torch.argsort(_flat(x)).to(torch.float32),
    "flip": lambda x: torch.flip(_flat(x), dims=[0]),
    "unique": lambda x: torch.unique(_flat(x)),
    "cumsum": lambda x: torch.cumsum(_flat(x), dim=0),
    "cumprod": lambda x: torch.cumprod(_flat(x), dim=0),
    "clamp01": lambda x: torch.clamp(to_tensor(x), 0.0, 1.0),
    "step": lambda x: (to_tensor(x) >= 0).to(torch.float32),
    "ones_like": lambda x: torch.ones_like(to_tensor(x)),
    "zeros_like": lambda x: torch.zeros_like(to_tensor(x)),
    # 归一化 / 度量
    "norm": _norm,
    "magnitude": _norm,
    "length": lambda x: int(_flat(x).numel()),
    "normalize": _normalize,
    "tnorm": _normalize,
    "snorm": _snorm,
    "zscore": _zscore,
    "dot": _dot,
    "cos_sim": _cos_sim,
    "cossim": _cos_sim,
    "cosine": _cos_sim,
    "dist": _dist,
    "softmax": _softmax,
    "topk": _topk,
    "topk_ind": _topk_ind,
    "argmax": _argmax,
    "argmin": lambda x, dim=None: int(torch.argmin(_flat(x))) if dim is None else
            torch.argmin(to_tensor(x), dim=int(_num(dim))).to(torch.float32),
    # 归约 / 组合
    "sum": lambda *a: _combine(a, False),
    "mean": lambda *a: _combine(a, True),
    "std": lambda x: torch.std(_flat(x), unbiased=False),
    "var": lambda x: torch.var(_flat(x), unbiased=False),
    "median": lambda x: torch.median(_flat(x)).values,
    "prod": lambda x: torch.prod(_flat(x)),
    "min": _reduce(lambda v: torch.min(v), _binary(torch.minimum)),
    "max": _reduce(lambda v: torch.max(v), _binary(torch.maximum)),
    "cat": _cat,
    "stack": _stack,
    "shape": _shape,
    "size": _shape,
    "numel": lambda x: int(to_tensor(x).numel()),
    "count": lambda x: int(_flat(x).numel()),
    "reshape": _reshape,
    "permute": _permute,
    "unsqueeze": lambda x, d=0: to_tensor(x).unsqueeze(int(_num(d))),
    "squeeze": lambda x, d=None: to_tensor(x).squeeze() if d is None else to_tensor(x).squeeze(int(_num(d))),
    "index": _index,
    "slice": _slice,
    "clip": _clip,
    "clamp": _clip,
    "lerp": _lerp,
    "mix": _lerp,
    "remap": _remap,
    "where": _where,
    "matmul": _binary(torch.matmul),
    "outer": _binary(torch.outer),
    "pow": _binary(_op.pow),
    "powe": _binary(_op.pow),
    "randn": _randn,
    "noise": _randn,
    "randu": _randu,
    "rand": _randu,
    "seedmix": _seedmix,
    "rownorm": _rownorm,
    "normclip": _normclip,
    "ints": _ints,
    "parse_ints": _ints,
    "reprows": _reprows,
    "tilerows": _reprows,
    "take": _take,
    "setrow": _setrow,
    "rowset": _setrow,
    "arange": _arange,
    "catrows": _catrows,
    "vstack": _catrows,
    # 三角函数补充（配合归一化向量做角度/插值）
    "angle": lambda x: torch.atan2(to_tensor(x)[1], to_tensor(x)[0]) if to_tensor(x).numel() >= 2 else torch.zeros(()),
    "deg2rad": lambda x: to_tensor(x) * (math.pi / 180.0),
    "rad2deg": lambda x: to_tensor(x) * (180.0 / math.pi),
}


# --------------------------------------------------------------------------------------
# 表达式求值（ast 白名单，不使用 eval）
# --------------------------------------------------------------------------------------
_BINOPS = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul, ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv, ast.Mod: _op.mod, ast.Pow: _op.pow, ast.MatMult: _op.matmul,
    ast.BitAnd: _op.and_, ast.BitOr: _op.or_, ast.BitXor: _op.xor,
}
_CMPOPS = {
    ast.Eq: _op.eq, ast.NotEq: _op.ne, ast.Lt: _op.lt, ast.LtE: _op.le,
    ast.Gt: _op.gt, ast.GtE: _op.ge,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "inf": float("inf"), "nan": float("nan"), "tau": _TWO_PI}
_ALLOWED_ATTRS = {"shape", "numel", "ndim", "T"}


class _EvalError(ValueError):
    pass


def _binop(op, l, r):
    if isinstance(l, (list, tuple)) or isinstance(r, (list, tuple)):
        raise _EvalError("列表（例如 V = 所有已连接 Vn 的集合）不能直接参与算术；"
                         "请用 stack(V) / cat(V) / mean(V) 先合并，或直接对 V0/V1 单独运算")
    if isinstance(l, str) or isinstance(r, str):
        raise _EvalError("字符串不能参与算术运算（向量请从 TENSOR 端口连入）")
    try:
        return _BINOPS[type(op)](l, r)
    except KeyError:
        raise _EvalError(f"不支持的运算符 {type(op).__name__}") from None
    except Exception as e:
        raise _EvalError(f"运算失败: {e}") from None


class _Evaluator:
    def __init__(self, variables):
        self.vars = variables

    # --- 单节点求值 -----------------------------------------------------------------
    def value(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.vars:
                return self.vars[node.id]
            raise _EvalError(f"未定义的变量/函数名: {node.id}（可用: {', '.join(sorted(self.vars)[:14])} ...）")
        if isinstance(node, ast.BinOp):
            return _binop(node.op, self.value(node.left), self.value(node.right))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -self.value(node.operand)
            if isinstance(node.op, ast.UAdd):
                return self.value(node.operand)
            if isinstance(node.op, ast.Not):
                return not self.value(node.operand)
            if isinstance(node.op, ast.Invert):
                return ~self.value(node.operand)
            raise _EvalError("不支持的一元运算符")
        if isinstance(node, ast.BoolOp):
            vals = [self.value(v) for v in node.values]
            if isinstance(node.op, ast.And):
                out = vals[0]
                for v in vals[1:]:
                    out = torch.minimum(to_tensor(out), to_tensor(v))
                return out
            out = vals[0]
            for v in vals[1:]:
                out = torch.maximum(to_tensor(out), to_tensor(v))
            return out
        if isinstance(node, ast.Compare):
            left = self.value(node.left)
            out = None
            cur = left
            for op, comp in zip(node.ops, node.comparators):
                right = self.value(comp)
                try:
                    r = _CMPOPS[type(op)](cur, right)
                except KeyError:
                    raise _EvalError(f"不支持的比较符 {type(op).__name__}") from None
                r = to_tensor(r) if torch.is_tensor(r) else (1.0 if r else 0.0)
                out = r if out is None else torch.minimum(to_tensor(out), to_tensor(r))
                cur = right
            return out
        if isinstance(node, ast.IfExp):
            c = self.value(node.test)
            c = bool(c) if not torch.is_tensor(c) else bool(c.any())
            return self.value(node.body if c else node.orelse)
        if isinstance(node, ast.Subscript):
            base = self.value(node.value)
            idx = node.slice
            if isinstance(idx, ast.Slice):
                i = self.value(idx.lower) if idx.lower else 0
                j = self.value(idx.upper) if idx.upper else None
                return to_tensor(base)[int(_num(i)):] if j is None else to_tensor(base)[int(_num(i)):int(_num(j))]
            key = self.value(idx)
            if torch.is_tensor(key) and key.dtype != torch.bool:
                key = key.long()          # 浮点下标（如 floor(randu*S)）自动取整
            return to_tensor(base)[key]
        if isinstance(node, ast.Attribute):
            base = self.value(node.value)
            if node.attr not in _ALLOWED_ATTRS:
                raise _EvalError(f"不允许访问属性 .{node.attr}（仅支持 {sorted(_ALLOWED_ATTRS)}）")
            if node.attr == "numel":
                return int(to_tensor(base).numel())
            return getattr(base, node.attr)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise _EvalError("只支持直接调用内置函数")
            name = node.func.id
            fn = FUNCTIONS.get(name) or FUNCTIONS.get(name.lower())
            if fn is None:
                raise _EvalError(f"未知函数: {name}（可用: {', '.join(sorted(FUNCTIONS)[:18])} ...）")
            args = [self.value(a) for a in node.args]
            kwargs = {k.arg: self.value(k.value) for k in node.keywords if k.arg}
            try:
                return fn(*args, **kwargs)
            except _EvalError:
                raise
            except Exception as e:
                raise _EvalError(f"调用 {name}() 失败: {e}") from None
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.value(e) for e in node.elts]
        if isinstance(node, ast.Starred):
            return self.value(node.value)
        raise _EvalError(f"不支持的语法: {type(node).__name__}")

    # --- 语句序列求值 ---------------------------------------------------------------
    def run(self, text):
        text = "\n".join(line.split("#")[0] for line in str(text).splitlines())
        if not text.strip():
            raise _EvalError("表达式为空")
        try:
            tree = ast.parse(text, mode="exec")
        except SyntaxError as e:
            raise _EvalError(f"语法错误: {e.msg}") from None
        result = None
        for st in tree.body:
            if isinstance(st, ast.Assign):
                if len(st.targets) != 1 or not isinstance(st.targets[0], ast.Name):
                    raise _EvalError("只支持简单赋值，例如  t = a + b")
                val = self.value(st.value)
                self.vars[st.targets[0].id] = val
                result = val
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                val = self.value(st.value) if st.value is not None else None
                self.vars[st.target.id] = val
                result = val
            elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                cur = self.vars[st.target.id]
                val = _binop(st.op, cur, self.value(st.value))
                self.vars[st.target.id] = val
                result = val
            elif isinstance(st, ast.Expr):
                result = self.value(st.value)
            elif isinstance(st, ast.If):
                cond = self.value(st.test)
                cond = bool(cond) if not torch.is_tensor(cond) else bool(cond.any())
                for s2 in (st.body if cond else st.orelse):
                    if isinstance(s2, ast.Assign) and isinstance(s2.targets[0], ast.Name):
                        result = self.value(s2.value)
                        self.vars[s2.targets[0].id] = result
                    else:
                        result = self.value(s2.value) if isinstance(s2, ast.Expr) else result
            else:
                raise _EvalError(f"不支持的语句: {type(st).__name__}")
        if result is None:
            raise _EvalError("表达式没有产生结果")
        return result


def run_expression(expression, variables):
    """求值，返回 (result, variables_after)。result 可能是 torch.Tensor 或 python 标量/列表。"""
    return _Evaluator(dict(variables)).run(expression), None


# 兼容函数名：evaluate 直接返回结果
def evaluate(expression, variables=None, **kw):
    vars_ = dict(variables or {})
    vars_.update(kw)
    return _Evaluator(vars_).run(expression)


# --------------------------------------------------------------------------------------
# 向量 <-> 文本
# --------------------------------------------------------------------------------------
def describe(value, precision=6, limit=0):
    """把结果序列化成一段人类可读文本（不省略，除非 limit>0）。"""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(describe(v, precision, limit) for v in value) + "]"
    if torch.is_tensor(value):
        v = value.detach().float().reshape(-1)
        n = v.numel()
        with torch.no_grad():
            parts = [f"{x:.{precision}g}" for x in v.tolist()]
        head = parts if limit <= 0 else parts[:limit] + ([f"... ({n} 项已截断)"] if n > limit else [])
        extra = f"  |  shape={tuple(value.shape)} numel={n} ‖x‖={float(torch.linalg.vector_norm(v)):.{precision}g}"
        if n and n <= 8:
            return "[" + ", ".join(head) + "]" + extra
        mean = float(v.mean()) if n else 0.0
        std = float(v.std(unbiased=False)) if n else 0.0
        return "[" + ", ".join(head) + f"]{extra} mean={mean:.{precision}g} std={std:.{precision}g}"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def vector_to_text(value, fmt="plain", precision=8, limit=0):
    """把张量/标量转成可完整保存、可再解析的文本。fmt: plain | comma | json | b64 | lines"""
    import json
    if torch.is_tensor(value):
        v = value.detach().float()
    elif isinstance(value, (list, tuple)):
        v = to_tensor(value)
    else:
        v = torch.tensor(float(value))
    v = v.reshape(-1)
    if fmt == "b64":
        import base64
        head = json.dumps({"dtype": "float32", "shape": list(value.shape if torch.is_tensor(value) else [v.numel()])})
        return "b64:" + head + ":" + base64.b64encode(v.numpy().astype("float32").tobytes()).decode("ascii")
    nums = v.tolist()
    if fmt == "json":
        return json.dumps([round(x, precision) if precision else x for x in nums])
    if fmt == "comma":
        return ",".join(f"{x:.{precision}g}" for x in nums)
    if fmt == "lines":
        return "\n".join(f"{x:.{precision}g}" for x in nums)
    return " ".join(f"{x:.{precision}g}" for x in nums)


def parse_vector_text(text):
    """把文本/文件内容解析回 1 维 float32 张量。自动识别 b64 / json / python repr / 分隔符格式。"""
    import base64
    import json
    import re
    s = (text or "").strip()
    if not s:
        raise ValueError("文本为空")
    if s.startswith("b64:"):
        idx = s.rindex(":")            # meta 是 JSON（本身含':'），payload 是 base64（不含':'）
        meta = json.loads(s[4:idx])
        raw = base64.b64decode(s[idx + 1:])
        t = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
        shape = meta.get("shape")
        if shape:
            try:
                t = t.reshape(tuple(shape))
            except Exception:
                pass
        return t
    # tensor([...]) / array([...]) / 裸列表
    s = re.sub(r"^\s*(tensor|array)\s*\(", "", s, flags=re.I).strip()
    s = re.sub(r"\)\s*,?\s*(dtype=.*)?$", "", s, flags=re.I).strip()
    s = s.replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ")
    s = s.replace("...", " ").replace("nan", "0").replace("inf", "0")
    toks = re.split(r"[\s,;]+", s.strip())
    nums = []
    for tk in toks:
        tk = tk.strip().strip("'\"")
        if not tk:
            continue
        try:
            nums.append(float(tk))
        except ValueError:
            nums.append(float("nan"))
    if not nums:
        raise ValueError("没有解析到任何数值")
    return torch.tensor(nums, dtype=torch.float32)
