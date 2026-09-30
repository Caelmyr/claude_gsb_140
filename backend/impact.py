# -*- coding: utf-8 -*-
"""
impact.py — 提交 / 合并前的影响分析与冲突预检
================================================
目标：在「动手」之前给出一份完整的影响清单与落到具体文件的风险提示，
把合并冲突化解在提交/合并之前，而不是等三方合并失败、工作区被写入
冲突标记后再补救。

两个只读分析入口（不修改工作区、不产生提交）：

  * CommitImpact.analyze(branch)
      「这份未提交改动一旦提交，会波及哪些目录/文件/行？」
      并交叉检查其它分支：哪些分支自共同祖先起也在动同一些文件，
      哪些文件在真正的三方合并里已经会产生内容冲突。

  * CommitImpact.preview_merge(source_ref, target_branch)
      「把 source 合并进 target 会发生什么？」——与 versioning.merge
      使用完全相同的 LCA / 三方状态机 / diff3 算法做干跑（dry-run），
      输出快进/无操作/真合并判定、文件级影响清单、冲突位置（base
      行号 + 双方内容片段）与逐文件理由，但不落盘、不提交。

风险级别：high（预计冲突/会丢数据）> warn（双侧异改需复核）>
info（值得知道）> ok（无风险）。每条风险都绑定具体 path 并给出理由。
"""

from collections import Counter, defaultdict

from . import config
from .diff_engine import diff_opcodes, diff_stats, merge3, split_lines
from .util import (decode_text, is_text_mime,
                   looks_binary, merge_label_swap, now, short_hash)

# 片段预览长度限制（防止超大文件/巨长行撑爆接口）
SNIPPET_HEAD_LINES = 4
SNIPPET_MAX_LINE = 200
SNIPPET_MAX_FILE_BYTES = 8 * 1024 * 1024

RISK_ORDER = {"high": 3, "warn": 2, "info": 1, "ok": 0, "": 0}


def classify_base(base_id, ours_id, theirs_id):
    """
    基于提交拓扑的正确三方判定（不受配置中的基线策略影响——预检必须如实
    反映拓扑关系）：
      theirs 是 ours 的祖先 => noop（源已被目标包含）
      ours   是 theirs 的祖先 => fast-forward（目标可快进到源）
      否则 => merge（真三方合并）
    """
    if base_id == theirs_id:
        return "noop"
    if base_id == ours_id:
        return "fast-forward"
    return "merge"


def proper_lca(versions, cid1, cid2):
    """
    按提交图计算最近公共祖先（LCA）：
      1. 取 cid1 的祖先链；沿 cid2 的父边自底向上找第一个出现在祖先链中的
         提交——这是 cid2 视角下「最近」的公共祖先；
      2. 在所有候选（cid2 第一父链/第二父链可能先碰到不同公共点）中选
         到两端最短距离和最小者，ts 仅作确定性兜底。
    相比 versioning.lca 的「ts 最大公共祖先」近似，本算法在含合并提交的
    DAG 中不会把被合并侧的祖先误判成 LCA。
    """
    if not cid1 or not cid2:
        return None
    if cid1 == cid2:
        return cid1
    with versions.meta.lock:
        commits = versions._v()["commits"]

        def distances(root):
            dist = {root: 0}
            frontier = [root]
            while frontier:
                cur = frontier.pop()
                d = dist[cur]
                for p in commits.get(cur, {}).get("parent_ids", []):
                    if p not in commits:
                        continue
                    if p not in dist or d + 1 < dist[p]:
                        dist[p] = d + 1
                        frontier.append(p)
            return dist

        d1, d2 = distances(cid1), distances(cid2)
        common = set(d1) & set(d2)
        if not common:
            return None
        # 「最近」= 到两端最短距离之和最小；距离相等时选更靠下的，ts 兜底
        return min(common,
                   key=lambda c: (max(d1[c], d2[c]), d1[c] + d2[c],
                                  -commits.get(c, {}).get("ts", 0)))


class CommitImpact:
    def __init__(self, nn):
        self.nn = nn
        self.vs = nn.versions

    # ================================================================ 工具
    @staticmethod
    def _ch(e):
        return (e or {}).get("content_hash")

    @staticmethod
    def _top_dir(path):
        parts = [s for s in (path or "").split("/") if s]
        return "/" + parts[0] if len(parts) > 1 else "/（根目录）"

    def _head_commit(self, v, branch):
        head_id = (v["branches"].get(branch) or {}).get("head")
        return v["commits"].get(head_id) if head_id else None

    def _read_entry_text(self, entry):
        """读取快照条目对应文本；二进制/不可读/超限返回 None。"""
        if not entry:
            return None
        if not is_text_mime(entry.get("mime")):
            return None
        if entry.get("size", 0) > SNIPPET_MAX_FILE_BYTES:
            return None
        try:
            data = self.nn.read_blocks(entry.get("block_ids", []))
        except Exception:
            return None
        if looks_binary(data):
            return None
        return decode_text(data) or ""

    @staticmethod
    def _line_stats_text(text_a, text_b):
        """两段文本的行级增删统计。"""
        la, lb = split_lines(text_a), split_lines(text_b)
        if len(la) + len(lb) > config.DIFF_MAX_LINES:
            return {"adds": None, "dels": None, "too_large": True}
        st = diff_stats(diff_opcodes(la, lb))
        return {"adds": st["adds"], "dels": st["dels"],
                "similarity": st["similarity"],
                "old_lines": len(la), "new_lines": len(lb)}

    @staticmethod
    def _line_hunks_text(text_a, text_b, cap=30):
        """行级变更区间（旧/新行号，均从 1 开始；区间为闭区间展示用）。"""
        la, lb = split_lines(text_a), split_lines(text_b)
        if len(la) + len(lb) > config.DIFF_MAX_LINES:
            return []
        out = []
        for tag, i1, i2, j1, j2 in diff_opcodes(la, lb):
            if tag == "equal":
                continue
            out.append({
                "kind": tag,
                "old_range": [i1 + 1, i2] if i2 > i1 else None,
                "new_range": [j1 + 1, j2] if j2 > j1 else None,
            })
            if len(out) >= cap:
                break
        return out

    @staticmethod
    def _clip_snippet(lines):
        if lines is None:
            return None
        clipped = [ln[:SNIPPET_MAX_LINE]
                   + ("…" if len(ln) > SNIPPET_MAX_LINE else "")
                   for ln in lines[:SNIPPET_HEAD_LINES]]
        return {"lines": clipped, "truncated":
                len(lines) > SNIPPET_HEAD_LINES, "total_lines": len(lines)}

    # ================================================================ 三方干跑
    def _three_way_text(self, base_text, ours_text, theirs_text,
                        ours_label, theirs_label):
        """对单个文本文件跑 diff3（与 versioning.merge 同一算法/标签）。"""
        result = merge3(split_lines(base_text), split_lines(ours_text),
                        split_lines(theirs_text),
                        ours_label=ours_label, theirs_label=theirs_label,
                        label_swap=merge_label_swap(config.CONFLICT_LABEL_SWAP),
                        marker_ours=config.CONFLICT_MARKER_OURS,
                        marker_sep=config.CONFLICT_MARKER_SEP,
                        marker_theirs=config.CONFLICT_MARKER_THEIRS)
        snippets = []
        for cf in result.conflicts:
            snippets.append({
                "base_range": [cf["base_start"] + 1, cf["base_end"]]
                if cf["base_end"] > cf["base_start"] else
                [cf["base_start"], cf["base_start"]],
                "base": self._clip_snippet(cf.get("base")),
                "ours": self._clip_snippet(cf.get("ours")),
                "theirs": self._clip_snippet(cf.get("theirs")),
            })
        return {"conflicts": len(result.conflicts), "snippets": snippets,
                "clean": not result.conflicts,
                "merged_text": result.text,
                "merged_lines": list(result.merged_lines)}

    def _simulate_file(self, b, o, t, ours_label="ours",
                       theirs_label="theirs", text_cache=None):
        """
        单个文件的三方合并干跑（对齐 versioning._merge_snapshots 的判定）。
        返回 dict：
          resolution: ours|theirs|auto|delete|none（none=双方一致/共同删除）
          conflict:   None|"content"|"modify/delete"|"binary"|"add/add"
          text:       content 冲突时的文本干跑详情
        """
        def same(x, y):
            if x is None or y is None:
                return x is None and y is None
            return x.get("content_hash") == y.get("content_hash")

        if o is None and t is None:
            return {"resolution": "none", "conflict": None}
        if same(o, t):
            return {"resolution": "ours" if o is not None else "none",
                    "conflict": None}
        if same(b, o):
            return {"resolution": "theirs" if t is not None else "delete",
                    "conflict": None}
        if same(b, t):
            return {"resolution": "ours", "conflict": None}

        # 双侧异改
        if o is None or t is None:
            return {"resolution": "ours" if o is not None else "theirs",
                    "conflict": "modify/delete"}
        if b is None:
            add_kind = "add/add"
        else:
            add_kind = None
        mime = o.get("mime") or t.get("mime") or ""
        too_big = max(o.get("size", 0), t.get("size", 0)) \
            > config.MERGE_MAX_TEXT_BYTES
        if is_text_mime(mime) and not too_big:
            tb = self._read_entry_text(b) if b is not None else ""
            to_ = self._read_entry_text(o)
            tt = self._read_entry_text(t)
            if tb is not None and to_ is not None and tt is not None:
                key = (b.get("content_hash") if b else None,
                       o.get("content_hash"), t.get("content_hash"))
                if text_cache is not None and key in text_cache:
                    tw = text_cache[key]
                else:
                    tw = self._three_way_text(tb, to_, tt,
                                              ours_label, theirs_label)
                    if text_cache is not None:
                        text_cache[key] = tw
                if not tw["conflicts"]:
                    return {"resolution": "auto", "conflict": None,
                            "text": tw}
                return {"resolution": "auto",
                        "conflict": add_kind or "content", "text": tw}
        return {"resolution": "ours",
                "conflict": add_kind or "binary"}

    # ================================================================ 提交预检
    def analyze_commit(self, branch=None):
        """未提交改动的完整影响清单 + 与其它分支的并行修改冲突风险。"""
        v = self.vs._v()
        branch = branch or v.get("head_branch", config.DEFAULT_BRANCH)
        head_id, _br = self.vs.branch_head(branch)
        head = v["commits"].get(head_id) if head_id else None
        head_snap = (head or {}).get("snapshot", {})

        working = self.vs.snapshot_fs()
        changes = self._change_rows(head_snap, working, mode="commit")

        # 波及目录汇总
        dir_stats = self._dir_stats([c["path"] for c in changes])

        # ---- 交叉检查其它分支 ----
        other_branches = [name for name in v["branches"] if name != branch]
        parallel = {name: self._branch_parallel(
            v, branch, head, other_branches_name=name)
            for name in other_branches}
        parallel = {k: x for k, x in parallel.items()
                    if x is not None and x["ahead"] > 0}

        changed_paths = {c["path"] for c in changes}
        text_cache = {}
        risks = []
        for c in changes:
            p = c["path"]
            c["parallel_branches"] = []
            c["risks"] = []
            for name, info in parallel.items():
                if p not in info["paths"]:
                    continue
                br_change = info["paths"][p]
                c["parallel_branches"].append({
                    "branch": name,
                    "head_short": short_hash(info["head"] or "", 8),
                    "kind": br_change["kind"],
                    "commits": br_change["commits"],
                    "authors": br_change["authors"],
                })
                # 以「提交后的工作区」为 ours，分支头为 theirs，真正干跑
                sim = self._simulate_file(
                    info["base_snap"].get(p),
                    c.get("_work_entry"),
                    br_change["their_entry"],
                    ours_label=branch, theirs_label=name,
                    text_cache=text_cache)
                risk = self._commit_file_risk(p, name, c, br_change, sim)
                if risk:
                    c["risks"].append(risk)
                    risks.append(risk)
            c.pop("_work_entry", None)
            c["risk_level"] = max((RISK_ORDER[r["level"]]
                                   for r in c["risks"]), default=0)

        # ---- 全局风险（不绑定具体文件的提示放在 notices） ----
        notices = []
        dirty = bool(changes)
        ahead_notes = self._parallel_notices(parallel, changed_paths)
        notices.extend(ahead_notes)
        if not dirty:
            notices.append({"level": "info",
                            "message": "工作区与 HEAD 一致，没有未提交改动；"
                                       "如需了解合并影响，请使用合并预检。"})

        summary = {
            "mode": "commit",
            "branch": branch,
            "head": head_id,
            "head_short": short_hash(head_id or "", 8),
            "dirty": dirty,
            "files": len(changes),
            "added": sum(1 for c in changes if c["kind"] == "added"),
            "modified": sum(1 for c in changes if c["kind"] == "modified"),
            "deleted": sum(1 for c in changes if c["kind"] == "deleted"),
            "adds": sum(c.get("adds") or 0 for c in changes),
            "dels": sum(c.get("dels") or 0 for c in changes),
            "dirs": dir_stats["total_dirs"],
            "high": sum(1 for r in risks if r["level"] == "high"),
            "warn": sum(1 for r in risks if r["level"] == "warn"),
            "info": sum(1 for r in risks if r["level"] == "info"),
            "generated_at": now(),
        }
        return {"summary": summary, "changes": changes,
                "dirs": dir_stats["dirs"],
                "parallel": [self._parallel_card(name, info)
                             for name, info in sorted(parallel.items())],
                "risks": risks, "notices": notices}

    def _commit_file_risk(self, path, other, c, br_change, sim):
        """根据三方干跑结果，为提交预检中的单个文件生成风险理由。"""
        conflict = sim.get("conflict")
        their_kind = br_change["kind"]
        n_commits = br_change["commits"]
        commits = f"{n_commits} 个提交" if n_commits else "分叉前的提交"
        who = "、".join(sorted(br_change["authors"])) or "其他成员"
        prefix = f"分支 {other} 自分叉后由 {who} 在 {commits} 中"
        tw = sim.get("text")

        if conflict in ("content", "add/add"):
            reason = (f"{prefix}也修改了本文件，且改动行区间与工作区重叠："
                      f"diff3 干跑产生 {tw['conflicts']} 处内容冲突，"
                      f"合并时会写入冲突标记，需人工逐处裁决。")
            return {"path": path, "level": "high", "kind": conflict,
                    "branch": other, "reason": reason,
                    "suggestion": "提交前先与 " + other +
                                  " 协调：错开改动区域，或先合并该分支并在本地解决冲突。",
                    "snippets": tw["snippets"] if tw else []}
        if conflict == "modify/delete":
            if c["kind"] == "deleted":
                reason = (f"{prefix}继续修改了本文件，而工作区把它删除了"
                          "（修改/删除冲突）；合并将保留对方的修改，删除意图会丢失。")
                sug = "确认是否仍需删除；如要删除，先通知对方并在合并时显式记录决议。"
            else:
                reason = (f"{prefix}删除了本文件，而工作区正在修改它"
                          "（修改/删除冲突）；合并将保留工作区版本，对方的删除意图会被覆盖。")
                sug = "确认文件是否应保留；如应删除，先把工作区改动迁移到别处再提交。"
            return {"path": path, "level": "high", "kind": "modify/delete",
                    "branch": other, "reason": reason, "suggestion": sug,
                    "snippets": []}
        if conflict == "binary":
            reason = (f"{prefix}也修改了本文件；该文件为二进制/超大文件，"
                      "无法做行级合并，干跑结果为保留本侧（HEAD）版本，"
                      "对方改动将被静默覆盖。")
            return {"path": path, "level": "high", "kind": "binary",
                    "branch": other, "reason": reason,
                    "suggestion": "与 " + other +
                                  " 约定唯一修改人，或在提交前手工选取正确版本。",
                    "snippets": []}
        # 无冲突：区分自动合并 / 单侧变更
        if sim.get("resolution") == "auto":
            sim_pct = None
            if tw:
                sim_pct = round((tw.get("similarity") or 0) * 100, 1)
            reason = (f"{prefix}也修改了本文件（{self._kind_cn(their_kind)}），"
                      "但双方改动落在不同行，diff3 干跑可自动合并；"
                      "建议提交后尽快合并并复核合并结果。")
            return {"path": path, "level": "warn", "kind": "parallel-clean",
                    "branch": other, "reason": reason,
                    "suggestion": "可安全提交，但请安排尽早合并，避免后续改动交叠。",
                    "snippets": []}
        if c["kind"] == "deleted" and their_kind == "deleted":
            return None
        reason = (f"{prefix}{self._kind_cn(their_kind)}了本文件，"
                  "工作区改动与其不冲突（合并会自动合并双方变更）。")
        return {"path": path, "level": "info", "kind": "parallel-one-side",
                "branch": other, "reason": reason,
                "suggestion": "合并时自动采纳双方变更，无需额外处理。",
                "snippets": []}

    @staticmethod
    def _kind_cn(kind):
        return {"added": "新增", "modified": "修改",
                "deleted": "删除"}.get(kind, kind)

    # ------------------------------------------------------ 并行分支信息
    def _branch_parallel(self, v, branch, head_commit, other_branches_name):
        """
        计算另一分支相对分叉点（LCA）在哪些文件上做了什么。
        返回 {head, base_id, base_entry_of_lca, paths: {path: {kind,
        commits, authors, their_entry}}}。
        """
        theirs = self._head_commit(v, other_branches_name)
        if theirs is None:
            return None
        ours_id = head_commit["id"] if head_commit else None
        base_id = proper_lca(self.vs, ours_id, theirs["id"]) if ours_id else None
        base_kind = classify_base(base_id, ours_id, theirs["id"])
        # 目标已包含对方（noop）或目标无提交时不算并行风险来源
        if base_kind == "noop":
            return None
        base_snap = (v["commits"].get(base_id) or {}).get("snapshot", {}) \
            if base_id else {}
        their_snap = theirs.get("snapshot", {})

        # 统计 base..theirs 之间触及该文件的提交（作者/次数）
        path_commits = defaultdict(list)
        ancestor_ids = self.vs.ancestors(theirs["id"])
        base_ancestors = self.vs.ancestors(base_id) if base_id else set()
        for cid in ancestor_ids - base_ancestors:
            cm = v["commits"].get(cid)
            if not cm:
                continue
            for p, e in cm.get("snapshot", {}).items():
                path_commits[p].append(cm)
            # 删除的文件也计入（其在更早父提交的快照里）
            for pp in self._commit_removed_paths(v, cm):
                path_commits[pp].append(cm)

        paths = {}
        for p in set(base_snap) | set(their_snap):
            be, te = base_snap.get(p), their_snap.get(p)
            if be and te and be.get("content_hash") == te.get("content_hash"):
                continue
            kind = "modified"
            if be is None and te:
                kind = "added"
            elif be and te is None:
                kind = "deleted"
            touched = path_commits.get(p, [])
            paths[p] = {
                "kind": kind,
                "commits": len(touched),
                "authors": sorted({m.get("author", "?") for m in touched}),
                "their_entry": te,
            }
        return {"head": theirs["id"], "base_id": base_id,
                "base_snap": base_snap, "paths": paths,
                "ahead": len(ancestor_ids - base_ancestors)}

    def _commit_removed_paths(self, v, commit):
        """该提交相对第一父提交删除的路径（用于作者归因）。"""
        parents = commit.get("parent_ids") or []
        if not parents:
            return set()
        p0 = v["commits"].get(parents[0])
        if not p0:
            return set()
        return set(p0.get("snapshot", {})) - set(commit.get("snapshot", {}))

    def _parallel_card(self, name, info):
        paths = info["paths"]
        return {
            "branch": name,
            "head_short": short_hash(info["head"] or "", 8),
            "ahead": info["ahead"],
            "touched_files": len(paths),
            "added": sum(1 for x in paths.values() if x["kind"] == "added"),
            "modified": sum(1 for x in paths.values()
                            if x["kind"] == "modified"),
            "deleted": sum(1 for x in paths.values()
                           if x["kind"] == "deleted"),
        }

    def _parallel_notices(self, parallel, changed_paths):
        out = []
        for name, info in parallel.items():
            overlap = set(info["paths"]) & changed_paths
            if not overlap:
                out.append({
                    "level": "info",
                    "message": f"分支 {name} 有 {info['ahead']} 个未合并提交、"
                               f"改动 {len(info['paths'])} 个文件，"
                               "与本次未提交改动没有文件交集，提交不受影响。"})
        return out

    # ================================================================ 合并预检
    def preview_merge(self, source_ref, target_branch=None):
        """三方合并干跑：判定 / 影响清单 / 逐文件冲突理由（只读）。"""
        v = self.vs._v()
        target_branch = target_branch or v.get("head_branch")
        target_head_id, _tbr = self.vs.branch_head(target_branch)
        theirs = self.vs.resolve_ref(source_ref)
        if not theirs:
            raise ValueError(f"源引用不存在: {source_ref}")
        if source_ref == target_branch:
            raise ValueError("源分支与目标分支相同")
        target_head = v["commits"].get(target_head_id) \
            if target_head_id else None

        # ---- 工作区脏：真正合并时会先自动提交，预检用工作区快照作为有效 ours ----
        # 注意：工作区只隶属于当前检出分支（head_branch）。其它目标分支的快照
        # 与活动工作区天然不同，不能把这种差异误判为「工作区脏」。
        working_snap = self.vs.snapshot_fs()
        checked_out = (target_branch == v.get("head_branch"))
        dirty = checked_out and self.vs.is_dirty(target_branch)
        effective_ours = working_snap if dirty else \
            (target_head.get("snapshot", {}) if target_head else {})
        work_changes = self._change_rows(
            target_head.get("snapshot", {}) if target_head else {},
            working_snap, mode="commit") if dirty else []
        work_change_paths = {c["path"] for c in work_changes}

        # 目标分支尚无提交 => 快进到 theirs
        if target_head is None:
            rows = self._change_rows({}, theirs.get("snapshot", {}),
                                     mode="merge")
            return self._merge_response(
                "fast-forward", source_ref, target_branch,
                None, target_head, theirs, rows, [],
                [{"level": "warn",
                  "message": f"目标分支 {target_branch} 尚无提交，"
                             "合并将直接快进到源分支头。"}],
                work_changes, dirty)

        base_id = proper_lca(self.vs, target_head["id"], theirs["id"])
        base_kind = classify_base(base_id, target_head["id"], theirs["id"])
        # 工作区脏时，真实合并会先自动提交工作区快照：
        #   此时 ours 是一个尚未存在的新提交（父=target_head），
        #   与 theirs 的 LCA 仍是 base_id，但 ours 绝不等于 base，
        #   因此 committed 视角下的 noop/fast-forward 都必须升级为真合并。
        if dirty and base_kind in ("noop", "fast-forward"):
            base_kind = "merge"
        if base_kind == "noop":
            return self._merge_response(
                "noop", source_ref, target_branch,
                base_id, target_head, theirs, [], [],
                [{"level": "ok",
                  "message": "源分支的提交已全部包含在目标分支中，"
                             "合并不会产生任何变化。"}],
                work_changes, dirty)

        base_snap = (v["commits"].get(base_id) or {}).get("snapshot", {}) \
            if base_id else {}
        theirs_snap = theirs.get("snapshot", {})

        if base_kind == "fast-forward":
            # 快进：结果直接变成 theirs；工作区脏时被改动文件可能被覆盖
            rows = self._change_rows(effective_ours, theirs_snap,
                                     mode="merge")
            risks, notices = self._fast_forward_risks(
                rows, work_change_paths, source_ref, target_branch)
            return self._merge_response(
                "fast-forward", source_ref, target_branch,
                base_id, target_head, theirs, rows, risks, notices,
                work_changes, dirty)

        # ---- 真正的三方合并：逐文件干跑 ----
        rows, risks, notices = self._three_way_merge_rows(
            base_snap, effective_ours, theirs_snap,
            work_change_paths, source_ref, target_branch)
        return self._merge_response(
            "merge", source_ref, target_branch,
            base_id, target_head, theirs, rows, risks, notices,
            work_changes, dirty)

    def _three_way_merge_rows(self, base_snap, ours_snap, theirs_snap,
                              work_change_paths, source_ref, target_branch):
        paths = sorted(set(base_snap) | set(ours_snap) | set(theirs_snap))
        text_cache = {}
        rows = []
        risks = []
        auto_merge_files = 0
        for p in paths:
            b, o, t = base_snap.get(p), ours_snap.get(p), theirs_snap.get(p)
            sim = self._simulate_file(
                b, o, t, ours_label=target_branch,
                theirs_label=source_ref, text_cache=text_cache)
            resolution, conflict = sim["resolution"], sim["conflict"]
            if resolution == "none":
                continue

            # 合并结果条目（仅用于影响清单的统计/行号）
            merged_text = None
            if resolution == "theirs":
                merged_e, deleted = t, False
            elif resolution == "delete":
                merged_e, deleted = None, True
            elif resolution == "ours":
                merged_e, deleted = o, False
            else:  # auto：内容由 diff3 生成，按 ours->merged 文本统计
                merged_e, deleted = o, False
                merged_text = (sim.get("text") or {}).get("merged_text")

            row = self._merge_row(p, o, merged_e, deleted, resolution,
                                  conflict, sim.get("text"), merged_text)
            # 工作区未提交修改在真实合并时会被先自动提交
            if p in work_change_paths:
                row["auto_commit"] = True
            rows.append(row)

            risk = self._merge_file_risk(
                p, conflict, sim, resolution, source_ref, target_branch,
                p in work_change_paths, o)
            if risk:
                row["risks"] = [risk]
                risks.append(risk)
            elif resolution == "auto" and row["kind"] != "unchanged":
                auto_merge_files += 1

        notices = []
        if auto_merge_files:
            notices.append({"level": "ok",
                            "message": f"{auto_merge_files} 个文本文件双方改动落在不同行，"
                                       "diff3 干跑可自动合并，无需人工干预。"})
        if work_change_paths:
            notices.insert(0, {
                "level": "warn",
                "message": f"工作区有 {len(work_change_paths)} 个未提交文件；"
                           "执行合并时系统会先自动提交它们（auto: 合并前快照），"
                           "预检已按工作区当前内容模拟。"})
        return rows, risks, notices

    def _merge_file_risk(self, path, conflict, sim, resolution,
                         source_ref, target_branch, in_work, ours_entry):
        tw = sim.get("text")
        if conflict in ("content", "add/add"):
            reason = (f"两侧自分叉点后都修改了本文件，且改动行区间重叠："
                      f"diff3 干跑产生 {tw['conflicts'] if tw else 1} 处冲突，"
                      f"真实合并会把 <<<<<<< 标记写入文件并生成「待解决」合并提交。")
            return {"path": path, "level": "high", "kind": conflict,
                    "branch": source_ref, "reason": reason,
                    "suggestion": "先不要直接合并：在差异对比页查看双方版本，"
                                  "约定取舍后再执行合并并逐处解决标记。",
                    "snippets": tw["snippets"] if tw else []}
        if conflict == "modify/delete":
            if ours_entry is not None:
                reason = (f"{source_ref} 删除了本文件，而 {target_branch} 修改了它"
                          "（修改/删除冲突）；合并会保留修改侧，删除不生效。")
                sug = "若确认要删除，先在目标分支删除并提交；否则告知对方保留。"
            else:
                reason = (f"{target_branch} 删除了本文件，而 {source_ref} 修改了它"
                          "（修改/删除冲突）；合并会保留源分支版本，删除不生效。")
                sug = "若确认要删除，合并后手动再删一次；否则无需处理。"
            if in_work:
                reason += " 注意：目标侧的修改来自你未提交的工作区。"
            return {"path": path, "level": "high", "kind": "modify/delete",
                    "branch": source_ref, "reason": reason,
                    "suggestion": sug, "snippets": []}
        if conflict == "binary":
            reason = ("两侧都修改了本文件，且它是二进制/超大文件，"
                      "无法行级合并；干跑结果为保留目标分支版本，"
                      f"{source_ref} 的版本会被覆盖且无文本标记提示。")
            return {"path": path, "level": "high", "kind": "binary",
                    "branch": source_ref, "reason": reason,
                    "suggestion": f"合并前从 {source_ref} 取回该文件人工选定版本，"
                                  "或让双方约定唯一修改人。",
                    "snippets": []}
        return None

    def _fast_forward_risks(self, rows, work_change_paths,
                            source_ref, target_branch):
        """快进合并下，未提交改动可能被直接覆盖/删除。"""
        risks = []
        notices = [{"level": "info",
                    "message": f"目标分支是 {source_ref} 的祖先，合并为快进："
                               "HEAD 直接移动到源分支头，无合并提交。"}]
        for r in rows:
            if r["path"] not in work_change_paths:
                continue
            if r["kind"] == "deleted":
                reason = ("快进会把该文件更新为源分支状态（删除）；"
                          "你在工作区对它的未提交修改将随自动提交保留在历史中，"
                          "但当前文件会从工作区消失。")
                level = "warn"
                sug = "如需保留改动，先在源分支补回该文件或从自动提交中恢复。"
            else:
                reason = ("快进会用源分支版本整体覆盖该文件，"
                          "你在工作区的未提交修改虽会被自动提交保护，"
                          "但不会进入合并结果，事后需手工挑回。")
                level = "high"
                sug = "先提交/暂存你的改动并确认取舍，再执行快进合并。"
            risk = {"path": r["path"], "level": level, "kind": "ff-overwrite",
                    "branch": source_ref, "reason": reason,
                    "suggestion": sug, "snippets": []}
            r["risks"] = [risk]
            risks.append(risk)
        return risks, notices

    # ------------------------------------------------------ 影响清单行
    def _change_rows(self, snap_a, snap_b, mode):
        """
        构造影响清单行（a=基态快照，b=新态快照）。
        mode="commit" 时 b 为工作区，行内补行级增删与行号区间；
        mode="merge" 时只做快照级统计。
        """
        rows = []
        for p in sorted(set(snap_a) | set(snap_b)):
            ea, eb = snap_a.get(p), snap_b.get(p)
            if ea and eb and ea.get("content_hash") == eb.get("content_hash"):
                continue
            if ea and not eb:
                kind = "deleted"
            elif eb and not ea:
                kind = "added"
            else:
                kind = "modified"
            row = {
                "path": p, "kind": kind,
                "dir": self._top_dir(p),
                "old_size": (ea or {}).get("size", 0),
                "new_size": (eb or {}).get("size", 0),
                "mime": (eb or ea or {}).get("mime", ""),
                "old_hash": short_hash((ea or {}).get("content_hash", ""), 10),
                "new_hash": short_hash((eb or {}).get("content_hash", ""), 10),
                "adds": None, "dels": None, "hunks": [],
                "risks": [], "risk_level": 0,
            }
            if mode == "commit":
                # 保留快照条目供三方干跑读取块内容（响应前会弹出该私有字段）
                row["_work_entry"] = dict(eb) if eb else None
                if kind == "modified":
                    ta, tb = self._read_entry_text(ea), \
                        self._read_entry_text(eb)
                    if ta is not None and tb is not None:
                        st = self._line_stats_text(ta, tb)
                        row.update({k: v for k, v in st.items()
                                    if k in ("adds", "dels", "similarity")})
                        row["hunks"] = self._line_hunks_text(ta, tb)
                elif kind == "added":
                    tb = self._read_entry_text(eb)
                    if tb is not None:
                        n = len(split_lines(tb))
                        row["adds"] = n
                        row["dels"] = 0
                        row["new_lines"] = n
                        row["hunks"] = [{"kind": "insert",
                                         "old_range": None,
                                         "new_range": [1, n]}] if n else []
                elif kind == "deleted":
                    ta = self._read_entry_text(ea)
                    if ta is not None:
                        n = len(split_lines(ta))
                        row["adds"] = 0
                        row["dels"] = n
                        row["old_lines"] = n
                        row["hunks"] = [{"kind": "delete",
                                         "old_range": [1, n],
                                         "new_range": None}] if n else []
            rows.append(row)
        return rows

    def _merge_row(self, path, ours_entry, merged_entry, deleted,
                   resolution, conflict, text_info, merged_text=None):
        """合并结果相对目标侧（ours）的影响行。"""
        text_changed = False
        if deleted:
            kind = "deleted"
            ea, eb = ours_entry, None
        elif resolution == "auto" and merged_text is not None:
            ea = ours_entry
            kind = "unchanged"
            eb = merged_entry
        elif merged_entry is not None and (
                ours_entry is None
                or merged_entry.get("content_hash") !=
                ours_entry.get("content_hash")):
            kind = "added" if ours_entry is None else "modified"
            ea, eb = ours_entry, merged_entry
        else:
            kind = "unchanged"
            ea = eb = merged_entry
        resolution_cn = {"ours": "保留目标侧", "theirs": "采纳源分支",
                         "auto": "双方自动合并",
                         "delete": "采纳删除"}.get(resolution, resolution)
        row = {
            "path": path, "kind": kind,
            "dir": self._top_dir(path),
            "old_size": (ea or {}).get("size", 0),
            "new_size": (eb or {}).get("size", 0)
                          if not isinstance(eb, str) else len(eb.encode()),
            "mime": (eb or ea or {}).get("mime", "")
                    if not isinstance(eb, str) else
                    (ea or {}).get("mime", ""),
            "old_hash": short_hash((ea or {}).get("content_hash", ""), 10),
            "new_hash": short_hash(
                (eb or {}).get("content_hash", "")
                if not isinstance(eb, str) else "", 10),
            "resolution": resolution, "resolution_cn": resolution_cn,
            "conflict_kind": conflict,
            "adds": None, "dels": None, "hunks": [],
            "risks": [], "risk_level": RISK_ORDER["high"] if conflict else 0,
        }
        if text_info and resolution == "auto":
            row["text_conflicts"] = text_info["conflicts"]
            ours_text = self._read_entry_text(ours_entry)
            if ours_text is not None and merged_text is not None:
                st = self._line_stats_text(ours_text, merged_text)
                row.update({k: v for k, v in st.items()
                            if k in ("adds", "dels", "similarity")})
                row["hunks"] = self._line_hunks_text(ours_text, merged_text)
            # 自动合并（含写入冲突标记的情形）相对目标侧总会产生新内容
            row["kind"] = "modified" if ours_entry is not None else "added"
        return row

    # ------------------------------------------------------ 目录汇总
    def _dir_stats(self, paths):
        cnt = Counter(self._top_dir(p) for p in paths)
        dirs = [{"dir": d, "files": n} for d, n in cnt.most_common()]
        return {"dirs": dirs, "total_dirs": len(cnt)}

    # ------------------------------------------------------ 响应封装
    def _merge_response(self, kind, source_ref, target_branch, base_id,
                        target_head, theirs, rows, risks, notices,
                        work_changes, dirty, notices_extra=None):
        dirs = self._dir_stats([r["path"] for r in rows
                                if r["kind"] != "unchanged"])
        conflict_rows = [r for r in rows if r.get("conflict_kind")]
        changed_rows = [r for r in rows if r["kind"] != "unchanged"]

        # ahead/behind：target 相对 source
        v = self.vs._v()
        if target_head is not None:
            a_t = self.vs.ancestors(target_head["id"])
            a_s = self.vs.ancestors(theirs["id"])
            ahead = len(a_t - a_s)
            behind = len(a_s - a_t)
        else:
            ahead, behind = 0, len(self.vs.ancestors(theirs["id"]))

        summary = {
            "mode": "merge",
            "kind": kind,
            "source": source_ref,
            "target": target_branch,
            "base": base_id,
            "base_short": short_hash(base_id or "", 8),
            "target_head": target_head["id"] if target_head else None,
            "target_head_short": short_hash(
                target_head["id"] if target_head else "", 8),
            "source_head": theirs["id"],
            "source_head_short": short_hash(theirs["id"], 8),
            "ahead": ahead, "behind": behind,
            "dirty": dirty,
            "auto_commit_files": len(work_changes),
            "files": len(changed_rows),
            "added": sum(1 for r in changed_rows if r["kind"] == "added"),
            "modified": sum(1 for r in changed_rows
                            if r["kind"] == "modified"),
            "deleted": sum(1 for r in changed_rows if r["kind"] == "deleted"),
            "conflicts": len(conflict_rows),
            "content_conflicts": sum(1 for r in conflict_rows
                                     if r["conflict_kind"] in
                                     ("content", "add/add")),
            "high": sum(1 for r in risks if r["level"] == "high"),
            "warn": sum(1 for r in risks if r["level"] == "warn"),
            "info": sum(1 for r in risks if r["level"] == "info"),
            "generated_at": now(),
        }
        all_notices = list(notices or []) + list(notices_extra or [])
        return {"summary": summary, "rows": rows, "dirs": dirs["dirs"],
                "risks": risks, "notices": all_notices,
                "working_changes": work_changes}
