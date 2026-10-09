# -*- coding: utf-8 -*-
"""
Computational LSNet —— 画师风格张量的计算工具箱
=================================================
把 LSNet（Kaloscope）的画师风格张量从"只能看"变成"能算、能存、能读回"。

包含 7 个节点（分类 Computational LSNet）：
  1. LSNet Features → Artist Tags   画风张量 → 画师串（不改权重，直接用分类头）
  2. LSNet Artist Tags → Features   画师串 → 画风张量（分类头权重方向）
  3. Vector Math                    向量/标量混合表达式运算（V0..V9 / F0..F3 都是 Any 端口）
  4. Vector → Text                  完整序列化 + 存 txt + 摘要（不再出现 tensor([...]) 省略）
  5. Text → Vector                  从 txt / 字符串读回向量
  6. Artist Tags → Prompt           反推结果 → 提示词画师串（含 @ 前缀等）
  7. Text Join                       拼接提示词片段（支持 STRING 列表）

依赖：只需要 torch。不修改 lsnet 插件，不修改任何模型权重。
"""

import glob as _glob
import json
import os
import re
import time

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# ComfyUI 运行时（standalone 测试时优雅降级）
# --------------------------------------------------------------------------------------
try:                                     # pragma: no cover
    import folder_paths
except Exception:                        # standalone
    folder_paths = None

# vmath 子模块：正常包加载失败时（例如目录名带连字符），退回按文件路径加载
try:
    from . import vmath
except Exception:                                                   # pragma: no cover
    import importlib.util as _ilu
    import sys as _sys
    _spec = _ilu.spec_from_file_location(
        "clsn_vmath", os.path.join(os.path.dirname(os.path.abspath(__file__)), "vmath.py"))
    vmath = _ilu.module_from_spec(_spec)
    _sys.modules["clsn_vmath"] = vmath
    _spec.loader.exec_module(vmath)


class AnyType(str):
    """始终相等的类型字符串 -> ComfyUI 的 '*'（Any）端口。"""
    def __ne__(self, other):
        return False


ANY = AnyType("*")


def _output_root():
    if folder_paths is not None:
        try:
            return folder_paths.get_output_directory()
        except Exception:
            pass
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(root, exist_ok=True)
    return root


def _resolve_input_path(name, max_depth=3):
    """把用户给的路径/文件名解析成真实路径。

    依次尝试：绝对路径 -> output/<name> -> input/<name> -> output 与 input 目录下按文件名递归查找（新的优先）。
    这样 Vector → Text 存到 output/lsnet_vectors/xxx.txt 后，直接填 xxx.txt 就能读回。
    """
    name = (name or "").strip().strip('"')
    if not name:
        return None
    cands = [name]
    roots = [_output_root()]
    if folder_paths is not None:
        try:
            roots.append(folder_paths.get_input_directory())
        except Exception:
            pass
    if not os.path.isabs(name):
        for r in roots:
            cands.append(os.path.join(r, name))
    for c in cands:
        if os.path.isfile(c):
            return c
    # 递归兜底：按文件名在 output/input 下查找，最近修改的优先
    base = os.path.basename(name)
    hits = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for depth in range(1, max_depth + 1):
            pat = os.path.join(r, *(["*"] * depth), base)
            hits.extend(g for g in _glob.glob(pat) if os.path.isfile(g))
    if hits:
        hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return hits[0]
    return None


def _count_unescaped(s, ch):
    """统计未被反斜杠转义的字符个数（识别 "(name:weight)" 的外层括号用）。"""
    c, i = 0, 0
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == ch:
            c += 1
        i += 1
    return c


def _strip_parens_weight(s):
    """'(name:0.8)' -> 'name:0.8'（只剥最外层、未转义的括号）。"""
    s = str(s).strip()
    if s.startswith("(") and s.endswith(")") and _count_unescaped(s, "(") == 1 \
            and _count_unescaped(s, ")") == 1:
        return s[1:-1].strip()
    return s


def _norm_tag(s):
    """提示词式名字与类别名对齐用：去掉空格/下划线/括号/转义/大小写差异。"""
    return re.sub(r"[\s_()\[\]\\'\".]+", "", str(s)).lower()


def _as_tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.detach().to(dtype=dtype)
    if isinstance(x, (int, float, bool)):
        return torch.tensor(float(x), dtype=dtype)
    if isinstance(x, (list, tuple)):
        return torch.tensor([float(v) for v in x], dtype=dtype)
    raise TypeError(f"无法把 {type(x).__name__} 当成向量/标量")


# ======================================================================================
# 1. 风格张量 -> 画师串
# ======================================================================================
class LSNetFeaturesToTags:
    """用 LSNet 的分类头把画师风格张量直接翻译成画师串（无需图像、不改权重）。

    head 是独立子模块（BN_Linear），forward_features 的输出就是它的输入，所以
    head(f) 与“图像 -> 分类”完全等价；eval 下 BN/Linear 均为仿射，因此
    head(mean f) == mean head(f)（多图均值张量再分类 == logit 空间集成）。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "features": ("TENSOR",),
                "model": ("LSNET_MODEL",),
                "top_k": ("INT", {"default": 10, "min": 1, "max": 2000}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "aggregate": (["mean_logits", "mean_probs"], {"default": "mean_logits"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1e6, "step": 0.01}),
                "l2_normalize": ("BOOLEAN", {"default": False,
                                             "tooltip": "⚠ 实测：归一化会把 ‖f‖≈49 压到 1，而 head.bn.running_mean 的范数本身就是 48.13，"
                                                        "图像信息被 BN 常数项淹没，所有图片都会退化成同一个画师（同一张图归一化后 top1 从 poki_(...) 变成 uni_mate）。"
                                                        "除非你同时把尺度乘回去，否则请保持关闭"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "TENSOR", "TENSOR")
    RETURN_NAMES = ("tag_string", "json_output", "probs", "logits")
    FUNCTION = "process"
    CATEGORY = "Computational LSNet"
    OUTPUT_NODE = True

    def process(self, features, model, top_k, threshold, aggregate, temperature, l2_normalize):
        bundle = model
        net = bundle["model"]
        device = bundle["device"]
        cmap = bundle.get("class_mapping", {})
        head = net.head
        feat_dim = head.bn.num_features if hasattr(head, "bn") else head[0].num_features

        f = _as_tensor(features)
        if f.numel() == 1 and feat_dim != 1:
            raise ValueError(f"传入的是标量，不是 {feat_dim} 维画师风格张量"
                             "（如果这是表达式结果，请检查它是否被 mean()/norm() 归约成了标量）")
        # 一行 = 一个风格向量；多行 = 多图/多向量（由上面的 aggregate 决定怎么聚合）。
        # 注意必须按 feat_dim 整分：以前这里直接 reshape(-1)，导致任何多行输入都会被拍平成
        # N*feat_dim 个元素后报"维度不匹配"，aggregate 实际上永远走不到。
        if f.numel() % feat_dim != 0:
            raise ValueError(f"特征维度不匹配: 输入共 {f.numel()} 个元素，head 期望 {feat_dim} 的整数倍"
                             "（必须是 projection 之后的特征，即 LSNet Common Features 的输出）")
        f = f.reshape(-1, feat_dim)
        f = f.to(device)
        if l2_normalize:
            f = F.normalize(f, dim=-1)

        was_training = net.training
        net.eval()
        try:
            with torch.no_grad():
                logits = head(f)                                 # (N, C)
                mean_logits = logits.mean(dim=0)
                if aggregate == "mean_probs":
                    probs = F.softmax(logits / temperature, dim=-1).mean(dim=0)
                    rank = probs
                else:
                    probs = F.softmax(mean_logits / temperature, dim=-1)
                    rank = mean_logits                            # 按 logits 排序，避免下溢并列
        finally:
            if was_training:
                net.train()

        k = min(int(top_k), rank.numel())
        _, idx = torch.topk(rank, k=k)
        p_at = probs[idx]

        results = []
        for i, p in zip(idx.cpu().numpy(), p_at.cpu().numpy()):
            p = float(p)
            if p < threshold:
                continue
            cid = int(i)
            results.append({"class_id": cid, "class_name": cmap.get(cid, f"Class {cid}"),
                            "probability": p})
        if len(results) > top_k:
            results = results[:top_k]

        tag_string = ",".join(r["class_name"] for r in results)
        js = json.dumps({r["class_name"]: r["probability"] for r in results}, ensure_ascii=False)
        return {"ui": {"text": [tag_string, js]},
                "result": (tag_string, js, probs.detach().cpu(), logits.detach().cpu())}


# ======================================================================================
# 2. 画师串 -> 风格张量
# ======================================================================================
class LSNetTagsToFeatures:
    """画师串 -> 风格张量：取 head 折叠 BN 后的类别权重行 W_eff[c] 作为该画师的方向。

    线性分类器的类权重方向 ≠ 该类样本的特征质心，属近似，但可用于闭环校验、
    画风混合的起点、GA 打分。支持 'a, b' 或 'a:0.8, b:1.2' 两种写法。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tag_string": ("STRING", {"default": "", "multiline": True}),
                "model": ("LSNET_MODEL",),
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING", "STRING")
    RETURN_NAMES = ("features", "feature_text", "report")
    FUNCTION = "process"
    CATEGORY = "Computational LSNet"

    @staticmethod
    def _parse(tag_string):
        """支持 'a, b' / '(a:0.8), b:0.2' / 提示词形式的名字（空格、转义括号）。"""
        out = []
        for chunk in re.split(r"[\n,]", str(tag_string)):
            s = _strip_parens_weight(chunk.strip())
            if not s:
                continue
            w = 1.0
            if ":" in s:
                name, _, tail = s.rpartition(":")
                try:
                    w = float(tail.strip())
                    s = name
                except ValueError:
                    pass
            s = s.strip()
            if s:
                out.append((s, w))
        return out

    def process(self, tag_string, model):
        bundle = model
        net = bundle["model"]
        cmap = bundle.get("class_mapping", {})
        name_to_id = {v: k for k, v in cmap.items()}
        fuzzy_id = {}
        for _k, _v in cmap.items():
            fuzzy_id.setdefault(_norm_tag(_v), _k)
        try:
            W = net.head.fuse().weight.detach().float().cpu()
        except Exception:
            bn, lin = net.head[0], net.head[1]
            scale = bn.weight / (bn.running_var + bn.eps) ** 0.5
            W = lin.weight.detach().float().cpu() * scale.detach().float().cpu()[None, :]

        feats, report = [], []
        for name, w in self._parse(tag_string):
            cid = name_to_id.get(name)
            how = "exact"
            if cid is None:                       # 提示词式名字（空格/转义括号）也能认出来
                cid = fuzzy_id.get(_norm_tag(name))
                how = "normalized" if cid is not None else "miss"
            if cid is None:
                report.append({"tag": name, "found": False})
                continue
            feats.append(W[cid] * float(w))
            report.append({"tag": name, "class_id": cid, "class_name": cmap.get(cid, ""),
                           "weight": w, "found": True, "matched": how})

        if not feats:
            f = torch.zeros(W.shape[1])
            return (f, vmath.vector_to_text(f), json.dumps(
                {"error": "没有解析到有效画师标签", "detail": report}, ensure_ascii=False))
        f = torch.stack(feats, dim=0).sum(dim=0).contiguous()
        return (f, vmath.vector_to_text(f),
                json.dumps({"resolved": report, "feature_dim": int(W.shape[1])}, ensure_ascii=False))


# ======================================================================================
# 3. Vector Math —— 向量/标量混合表达式
# ======================================================================================
class VectorMath:
    """向量算术：V0..V9 / F0..F3 为 Any 端口，向量、标量、字符串都能直接连进来。

    别名（与 more_math 习惯一致）：a=V0 b=V1 c=V2 d=V3 w=V4 x=V5 y=V6 z=V7
                                F0..F3 若已连接，则覆盖 w x y z
    变量 V = 所有已连接 Vn 组成的列表，Vcnt = 连接个数
    常用：norm(a)  cos_sim(a,b)  mix(a,b,w)  a*(1-w)+b*w  mean(a,b)  softmax(a)  topk(a,10)
    遗传算法/矩阵运算补充（详见 README §3）：
      randu(seed,[S,D]) randn(seed,[S,D])  带随机种的均匀/正态噪声（不写 seed 就用全局 RNG）
      seedmix(seed,k)                      基准随机种 × 轮次 -> 每轮不同、可复现的随机种
      rownorm(M)   normclip(M,lo,hi,ref)   逐行范数 / 逐行把范数钳制到 [lo*ref,hi*ref]
      ints('1,2,4') reprows(f,n) take(M,i) catrows(A,B)   序号串->下标、复制成 n 行、按行取值、按行拼接

    尺度警告（实测）：head 只在训练尺度（‖f‖≈25~100）附近有效，因为 head.bn.running_mean 的
    范数是 48.13 ≈ 典型特征范数 49。把向量 L2 归一化（‖f‖=1）后再反推，图像信息会被 BN 的
    常数项淹没 —— 实测所有图片都退化成同一个画师（top1 全变 uni_mate，与全零向量相同）。
      * 反推画师串 / 混合风格：用原始尺度，如 mean(V0, V1) 或 mix(V0, V1, 0.3)
      * 只想比较方向时再用 normalize() + cos_sim()
      * 需要统一尺度时：normalize(V0) * norm(V1)（把 V0 缩放到 V1 的尺度）
    多语句：用换行或 ';' 分隔，支持 t = a+b 这样的赋值，结果为最后一条语句的值。
    """

    @classmethod
    def INPUT_TYPES(s):
        opt = {"expression": ("STRING", {"default": "a*(1-w) + b*w", "multiline": True}),
               "precision": ("INT", {"default": 6, "min": 0, "max": 12})}
        for i in range(10):
            opt[f"V{i}"] = (ANY,)
        for i in range(4):
            opt[f"F{i}"] = (ANY,)
        return {"required": {}, "optional": opt}

    RETURN_TYPES = ("TENSOR", "FLOAT", "STRING")
    RETURN_NAMES = ("result", "value", "text")
    FUNCTION = "process"
    CATEGORY = "Computational LSNet"
    OUTPUT_NODE = True

    def process(self, expression, precision, **kwargs):
        variables = {}
        v_list, f_list = [], []
        for i in range(10):
            v = kwargs.get(f"V{i}", None)
            if v is not None:
                variables[f"V{i}"] = v
                v_list.append(v)
        for i in range(4):
            f = kwargs.get(f"F{i}", None)
            if f is not None:
                variables[f"F{i}"] = f
                f_list.append(f)
        variables.update({
            "a": kwargs.get("V0", None), "b": kwargs.get("V1", None),
            "c": kwargs.get("V2", None), "d": kwargs.get("V3", None),
            "w": kwargs.get("V4", None), "x": kwargs.get("V5", None),
            "y": kwargs.get("V6", None), "z": kwargs.get("V7", None),
        })
        for k in ("a", "b", "c", "d"):        # 未连接则按 0 处理，避免整条链因缺参数报错
            if variables[k] is None:
                variables[k] = 0.0
        for i, key in enumerate(("w", "x", "y", "z")):          # F 覆盖同名别名
            if f_list and kwargs.get(f"F{i}", None) is not None:
                variables[key] = kwargs[f"F{i}"]
            elif variables[key] is None:
                variables[key] = 0.0
        for i in range(10):
            variables.setdefault(f"V{i}", 0.0)
        variables["V"] = v_list
        variables["Vcnt"] = float(len(v_list))
        variables["V_count"] = float(len(v_list))
        variables["F"] = f_list
        variables["Fcnt"] = float(len(f_list))

        result = vmath.evaluate(expression, variables)
        text = vmath.describe(result, precision=int(precision))
        if torch.is_tensor(result):
            t = result.detach().float()
            scalar = float(t.reshape(-1)[0]) if t.numel() == 1 else float("nan")
        elif isinstance(result, (int, float, bool)):
            t, scalar = torch.tensor(float(result)), float(result)
        elif isinstance(result, (list, tuple)):
            t, scalar = vmath.to_tensor(result), float("nan")
        else:
            raise ValueError(f"表达式结果类型 {type(result).__name__} 不能作为张量输出")
        return {"ui": {"text": [text]}, "result": (t, scalar, text)}


# ======================================================================================
# 4. Vector -> Text
# ======================================================================================
class VectorToText:
    """把向量完整地变成文本：不再出现 tensor([...]) 的省略号；可选写入 txt 文件。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vector": (ANY,),
                "format": (["plain", "comma", "json", "b64", "lines"], {"default": "plain"}),
                "precision": ("INT", {"default": 9, "min": 1, "max": 12,
                                      "tooltip": "float32 无损往返需要 9 位有效数字；b64 格式与精度无关，逐比特无损"}),
                "display_limit": ("INT", {"default": 0, "min": 0, "max": 100000,
                                          "tooltip": "0 = 界面文本完全显示；>0 只显示前 N 个值"}),
            },
            "optional": {
                "save_to_file": ("BOOLEAN", {"default": True}),
                "filename": ("STRING", {"default": "lsnet_vector.txt"}),
                "subfolder": ("STRING", {"default": "lsnet_vectors"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "TENSOR")
    RETURN_NAMES = ("text", "summary", "saved_path", "numel", "vector_out")
    FUNCTION = "process"
    CATEGORY = "Computational LSNet"
    OUTPUT_NODE = True

    def process(self, vector, format, precision, display_limit, save_to_file=True,
                filename="lsnet_vector.txt", subfolder="lsnet_vectors"):
        t = vector if torch.is_tensor(vector) else _as_tensor(vector)
        text = vmath.vector_to_text(vector, fmt=format, precision=int(precision))
        v = t.detach().float().reshape(-1)
        summary = (f"shape={tuple(t.shape)} numel={v.numel()} dtype={t.dtype} "
                   f"‖x‖={float(torch.linalg.vector_norm(v)):.6g} "
                   f"min={float(v.min()):.6g} max={float(v.max()):.6g} "
                   f"mean={float(v.mean()):.6g} std={float(v.std(unbiased=False)):.6g} "
                   f"zeros={int((v == 0).sum())}")
        preview = text if int(display_limit) <= 0 else text[:int(display_limit)] + \
            f" ... (已截断显示，完整长度 {len(text)} 字符，完整内容见 text 输出/文件)"

        saved = ""
        if save_to_file:
            sub = (subfolder or "lsnet_vectors").strip().strip("/\\")
            root = os.path.join(_output_root(), sub) if sub else _output_root()
            os.makedirs(root, exist_ok=True)
            name = (filename or "").strip() or f"lsnet_vector_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            if not os.path.splitext(name)[1]:
                name += ".txt"
            path = os.path.join(root, name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            saved = path

        return {"ui": {"text": [summary, preview]},
                "result": (text, summary, saved, int(v.numel()), t)}


# ======================================================================================
# 5. Text -> Vector
# ======================================================================================
class TextToVector:
    """从字符串或 txt 文件读回向量；自动识别 b64 / json / tensor([...]) / 空格分隔等格式。

    只要不是 b64，文本里的 "..." 或 nan 会被当作 0（提示会写在 report 里）；
    b64 格式是逐比特无损的，适合“存-读”往返。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["auto", "plain", "comma", "json", "b64", "lines"],
                         {"default": "auto"}),
            },
            "optional": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "file": ("STRING", {"default": "",
                                    "tooltip": "txt 文件名或绝对路径；例如 lsnet_vectors/lsnet_vector.txt"}),
                "dtype": (["float32", "float64", "float16"], {"default": "float32"}),
                "trigger": (ANY, {"tooltip": "把 Vector → Text 的 vector_out 接进来：ComfyUI 不知道"
                                             "文件依赖，这样做才能保证先保存再读取"}),
                "on_missing": (["error", "zeros"], {"default": "error",
                                                    "tooltip": "文件还不存在时：error 直接报错；"
                                                               "zeros 返回一个 1×1 的零向量占位"
                                                               "（用于“第一轮还没有缓存”的循环工作流）"}),
            },
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("vector", "report")
    FUNCTION = "process"
    CATEGORY = "Computational LSNet"

    @classmethod
    def IS_CHANGED(cls, mode, text="", file="", dtype="float32", trigger=None, on_missing="error"):
        """按文件内容判定是否要重新执行：缓存文件每轮被覆盖（mtime 变），本节点才会重读。

        （否则 ComfyUI 会跨队列复用上一次读到的张量 —— 迭代类工作流必须重读。）
        """
        name = (file or "").strip()
        if not name:
            return float("NaN")
        path = _resolve_input_path(name)
        if path is None:
            return f"missing:{name}"
        try:
            st = os.stat(path)
            return f"{path}:{st.st_mtime_ns}:{st.st_size}"
        except Exception:
            return f"{path}:unknown"

    def process(self, mode, text="", file="", dtype="float32", trigger=None, on_missing="error"):
        src = ""
        used = ""
        if (file or "").strip():
            path = _resolve_input_path(file)
            if path is None:
                if on_missing == "zeros":
                    v = torch.zeros(1, 1, dtype=torch.float32)
                    return (v, json.dumps({"source": f"missing:{file}", "detected_format": "zeros",
                                           "numel": 1, "dtype": "float32", "chars": 0,
                                           "note": "文件不存在，按 on_missing=zeros 返回 1×1 零向量"},
                                          ensure_ascii=False))
                raise FileNotFoundError(f"找不到文件: {file}"
                                        f"（已尝试绝对路径、{_output_root()} 及其子目录、input 目录）")
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                src = fh.read()
            used = f"file:{path}"
        else:
            src = text or ""
            used = "text"
        if not src.strip():
            if on_missing == "zeros":
                v = torch.zeros(1, 1, dtype=torch.float32)
                return (v, json.dumps({"source": used, "detected_format": "zeros", "numel": 1,
                                       "dtype": "float32", "chars": 0,
                                       "note": "输入为空，按 on_missing=zeros 返回 1×1 零向量"},
                                      ensure_ascii=False))
            raise ValueError("输入为空：file 与 text 都没有内容")
        fmt = mode
        if fmt == "auto":
            s = src.strip()
            if s.startswith("b64:"):
                fmt = "b64"
            elif s.startswith("["):
                fmt = "json"
            else:
                fmt = "plain"
        if fmt == "json":
            try:
                vals = json.loads(src)
                if isinstance(vals, dict):
                    vals = list(vals.values())
                v = torch.tensor([float(x) for x in vals], dtype=torch.float32)
            except Exception:
                v = vmath.parse_vector_text(src)
        else:
            v = vmath.parse_vector_text(src)
        v = v.to({"float32": torch.float32, "float64": torch.float64,
                  "float16": torch.float16}[dtype])
        report = json.dumps({"source": used, "detected_format": fmt,
                             "numel": int(v.numel()), "dtype": str(v.dtype),
                             "chars": len(src),
                             "note": "非 b64 格式的 '...'/'nan' 已被当作 0"}, ensure_ascii=False)
        return (v, report)


class ArtistTagsToPrompt:
    """反推结果 → 提示词画师串：把 LSNet 的 json（或标签串）变成能直接喂 CLIPTextEncode 的串。

    ⚠ 权重是**启发式**：json 里的概率是分类置信度，与提示词权重没有解析关系（见 README §4）。
    模式：
      rank      按名次线性给权重（默认，最稳）        → (a:0.8),(b:0.5),(c:0.2)
      equal     全部等权（不加括号）                  → a,b,c
      rel       在所选 top_n 的概率上做 min-max → [min,max]
      prob      p/p_max → [min,max]
      as_given  直接沿用 tag_string 里写好的权重（如 "(bbul horn:0.8),arl:0.2"）
    名称处理：下划线→空格、转义 ( ) [ ]、清掉会破坏 "(name:weight)" 语法的 , : | ;，
             并可用 name_prefix / name_suffix 给【每个画师名】加前后缀（不同底模对画师名前缀要求不同，
             如前面加 @ → (@bbul horn:0.8)）。整串级的前后拼接请用 Text Join 节点。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weight_mode": (["rank", "equal", "rel", "prob", "as_given"], {"default": "rank"}),
                "top_n": ("INT", {"default": 5, "min": 1, "max": 60}),
                "max_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "min_weight": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "json_output": ("STRING", {"default": ""}),
                "tag_string": ("STRING", {"default": "", "multiline": True}),
                "name_style": (["spaces", "as_is"], {"default": "spaces",
                                                     "tooltip": "spaces: 下划线转空格（bbul_horn → bbul horn，多数 SDXL 底模习惯）"}),
                "escape_parens": ("BOOLEAN", {"default": True,
                                              "tooltip": "把名字里的 ( ) [ ] 转义成 \\( \\) \\[ \\]，避免破坏 (name:weight) 语法"}),
                "decimals": ("INT", {"default": 2, "min": 0, "max": 4,
                                     "tooltip": "权重小数位，如 2 → 0.80"}),
                "name_prefix": ("STRING", {"default": "", "tooltip": "加在【每个画师名前】（在权重括号内）。"
                                                                     "如填 @ → (@bbul horn:0.8),(@rin31153336:0.4)"}),
                "name_suffix": ("STRING", {"default": "", "tooltip": "加在【每个画师名后】（在权重括号内）。"
                                                                     "如填 style → (bbul horn style:0.8)"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "report")
    FUNCTION = "build"
    CATEGORY = "Computational LSNet"
    OUTPUT_NODE = True

    # ---- 解析输入 -------------------------------------------------------------
    @staticmethod
    def _split_tag(chunk):
        """把 '(name:0.8)' / 'name:0.8' / 'name' 拆成 (name, weight or None)。"""
        s = _strip_parens_weight(chunk.strip().strip(","))
        if not s:
            return None
        w = None
        if ":" in s:
            head, _, tail = s.rpartition(":")
            try:
                w = float(tail.strip())
                s = head
            except ValueError:
                pass
        return (s.strip(), w)

    def _collect(self, json_output, tag_string):
        items = []
        js = (json_output or "").strip()
        if js:
            try:
                d = json.loads(js)
                if isinstance(d, dict):
                    for k, v in d.items():
                        try:
                            p = float(v)
                        except Exception:
                            p = 0.0
                        items.append([str(k), p, None])
            except Exception:
                items = []
        if not items and (tag_string or "").strip():
            for chunk in re.split(r"[\n]", tag_string):
                for piece in chunk.split(","):
                    got = self._split_tag(piece)
                    if got and got[0]:
                        items.append([got[0], 0.0, got[1]])
        # 合并重复名（同名取最大概率、权重取最大）
        merged = {}
        for name, prob, w in items:
            if name in merged:
                merged[name][0] = max(merged[name][0], prob)
                if w is not None:
                    merged[name][1] = max(merged[name][1] or -1e9, w)
            else:
                merged[name] = [prob, w]
        return [[k, v[0], v[1]] for k, v in merged.items()]

    # ---- 名称清洗 -------------------------------------------------------------
    @staticmethod
    def _clean(name, name_style, escape_parens):
        nm = str(name).strip().strip('"\'')
        # 先把已有的转义还原，避免双重转义（如 "\(huhi 1211\)"）
        for a, b in (("\\(", "("), ("\\)", ")"), ("\\[", "["), ("\\]", "]")):
            nm = nm.replace(a, b)
        if name_style == "spaces":
            nm = nm.replace("_", " ")
        nm = re.sub(r"[,;|:]+", " ", nm)
        nm = re.sub(r"\s+", " ", nm).strip()
        if escape_parens:
            nm = nm.replace("(", "\\(").replace(")", "\\)")
            nm = nm.replace("[", "\\[").replace("]", "\\]")
        return nm

    def build(self, weight_mode="rank", top_n=5, max_weight=0.8, min_weight=0.2, json_output="",
              tag_string="", name_style="spaces", escape_parens=True, decimals=2,
              name_prefix="", name_suffix=""):
        items = self._collect(json_output, tag_string)
        if not items:
            raise ValueError("没有可用输入：请把 LSNet Features → Artist Tags 的 json_output 或 tag_string 接进来")
        if weight_mode == "as_given":
            given = [[n, p, w] for n, p, w in items if w is not None]
            if given:
                items = given
            else:                                  # 没写权重就退回 rank
                weight_mode = "rank"
        n = max(1, min(int(top_n), len(items)))
        sel = items[:n]

        probs = [p for _, p, _ in sel]
        p_max, p_min = max(probs), min(probs)
        weights = []
        for i, (name, prob, given_w) in enumerate(sel):
            if weight_mode == "equal":
                w = 1.0
            elif weight_mode == "rank":
                w = max_weight if n == 1 else max_weight - (max_weight - min_weight) * i / (n - 1)
            elif weight_mode == "prob":
                r = (prob / p_max) if p_max > 0 else 1.0
                w = min_weight + (max_weight - min_weight) * max(0.0, min(1.0, r))
            elif weight_mode == "rel":
                r = ((prob - p_min) / (p_max - p_min)) if p_max > p_min else 1.0
                w = min_weight + (max_weight - min_weight) * max(0.0, min(1.0, r))
            elif weight_mode == "as_given":
                w = float(given_w if given_w is not None else max_weight)
            else:
                w = max_weight
            weights.append(round(float(w), max(0, int(decimals)) + 2))

        pre, suf = (name_prefix or ""), (name_suffix or "")
        toks, rep = [], []
        for (name, prob, _), w in zip(sel, weights):
            nm = self._clean(name, name_style, escape_parens)
            if not nm:
                continue
            nm = f"{pre}{nm}{suf}"          # 前后缀加在每个画师名上（括号内）
            tok = nm if (weight_mode == "equal" or abs(w - 1.0) < 1e-9) else \
                f"({nm}:{w:.{max(0, int(decimals))}f})"
            toks.append(tok)
            rep.append({"name": name, "cleaned": nm, "weight": w, "prob": round(prob, 8)})

        prompt = ",".join(toks)
        report = json.dumps({"mode": weight_mode, "count": len(rep),
                             "name_prefix": pre, "name_suffix": suf,
                             "weights": [r["weight"] for r in rep],
                             "tags": rep, "prompt": prompt}, ensure_ascii=False)
        return {"ui": {"text": [prompt]}, "result": (prompt, report)}


class TextJoin:
    """把若干文本/列表拼成一段提示词（不需要 easy-use 的 promptConcat）。

    可接 STRING、STRING 列表（如 WD14 Tagger 的输出）、或任何可转成字符串的值。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "separator": ("STRING", {"default": ","}),
                "skip_empty": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "part1": (ANY,), "part2": (ANY,), "part3": (ANY,), "part4": (ANY,),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "join"
    CATEGORY = "Computational LSNet"
    OUTPUT_NODE = True

    def join(self, separator=",", skip_empty=True, part1=None, part2=None, part3=None, part4=None):
        parts = []
        for p in (part1, part2, part3, part4):
            if p is None:
                continue
            items = list(p) if isinstance(p, (list, tuple)) else [p]
            for it in items:
                t = str(it).strip()
                if skip_empty and not t:
                    continue
                if t:
                    parts.append(t)
        text = separator.join(parts)
        # 去掉重复分隔符与首尾分隔符
        while separator and separator + separator in text:
            text = text.replace(separator + separator, separator)
        if separator:
            text = text.strip().strip(separator)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "LSNetFeaturesToTags": LSNetFeaturesToTags,
    "LSNetTagsToFeatures": LSNetTagsToFeatures,
    "CLSNetVectorMath": VectorMath,
    "CLSNetVectorToText": VectorToText,
    "CLSNetTextToVector": TextToVector,
    "CLSNetArtistPrompt": ArtistTagsToPrompt,
    "CLSNetTextJoin": TextJoin,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LSNetFeaturesToTags": "LSNet Features → Artist Tags",
    "LSNetTagsToFeatures": "LSNet Artist Tags → Features",
    "CLSNetVectorMath": "Vector Math (Computational LSNet)",
    "CLSNetVectorToText": "Vector → Text / txt",
    "CLSNetTextToVector": "Text / txt → Vector",
    "CLSNetArtistPrompt": "Artist Tags → Prompt (画师串→提示词)",
    "CLSNetTextJoin": "Text Join (拼接提示词)",
}
