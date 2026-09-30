# -*- coding: utf-8 -*-
"""
versioning.py — 版本控制（提交 / 分支 / 合并 / 检出 / 差异）
================================================================
难点三：版本树冲突合并。

数据模型（meta['versions'] JSON 文档）：
    {
      "head_branch": "main",
      "branches": {name: {"name","head","created_at","created_by","desc"}},
      "commits":  {cid: {
          "id","parent_ids":[..],"message","author","ts",
          "snapshot": {path: {"type","size","content_hash","block_ids","mime","owner","mode"}},
          "tree_hash","stats":{...},"conflicts":[..],"merge_info":{...}
      }}
    }

提交快照直接引用块的 block_ids——块不可变（写入后 genstamp/校验和固定），
因此历史版本天然可读：旧提交引用的块被 GC 保护（引用集 = 活动文件 ∪
全部提交快照）。检出/合并即"把某个快照物化回活动 inode 树"。

合并 = 快照级三方合并（base = LCA）：
  * 单侧变更（含增/删）        -> 直接采纳；
  * 双侧同改但内容一致         -> 采纳其一；
  * 双侧同改且不一致：
      - 文本文件：读取三方内容做 diff3 行级合并（diff_engine.merge3），
        干净则生成合并后内容写入新块；有冲突则写入冲突标记并记录；
      - 修改/删除冲突：保留修改侧并记录冲突；
      - 二进制：保留 ours 并记录冲突。
合并产生双亲提交（merge commit），冲突路径写入 commit.conflicts。
"""

import posixpath
import threading

from . import config
from .diff_engine import diff_opcodes, diff_stats, merge3, split_lines
from .util import (LRU, canonical_json, classify_merge_base, decode_text,
                   gen_id, is_text_mime, looks_binary, merge_label_swap,
                   now, sha256_bytes, sha256_text, short_hash, sort_by_ts)


class VersionError(Exception):
    pass


class VersionStore:
    def __init__(self, nn):
        self.nn = nn
        self.meta = nn.meta
        self.lock = threading.RLock()
        self._diff_cache = LRU(maxsize=128)

    # ---------------------------------------------------------------- 基础
    def _v(self):
        return self.meta.get("versions")

    def ensure_head(self):
        with self.meta.lock:
            v = self._v()
            v.setdefault("head_branch", config.DEFAULT_BRANCH)
            v.setdefault("branches", {})
            v.setdefault("commits", {})
            branches = v["branches"]
            if config.DEFAULT_BRANCH not in branches:
                branches[config.DEFAULT_BRANCH] = {
                    "name": config.DEFAULT_BRANCH, "head": None,
                    "created_at": now(), "created_by": "system",
                    "desc": "默认主分支",
                }
            self.meta.touch("versions")

    # ------------------------------------------------------------ 快照/树
    def snapshot_fs(self):
        """当前活动文件系统 -> 快照 {path: entry}。"""
        snap = {}
        for path, inode in self.nn.fs.all_files():
            snap[path] = {
                "type": "file",
                "size": inode.get("size", 0),
                "content_hash": inode.get("content_hash"),
                "block_ids": list(inode.get("block_ids", [])),
                "mime": inode.get("mime", ""),
                "owner": inode.get("owner", "admin"),
                "mode": inode.get("mode", "rw-r--r--"),
                "inode": inode["id"],
            }
        return snap

    @staticmethod
    def tree_hash(snapshot):
        slim = {p: {"h": e.get("content_hash"), "s": e.get("size")}
                for p, e in sorted(snapshot.items())}
        return sha256_text(canonical_json(slim))

    def branch_head(self, branch=None):
        v = self._v()
        branch = branch or v.get("head_branch", config.DEFAULT_BRANCH)
        br = v["branches"].get(branch)
        if not br:
            raise VersionError(f"分支不存在: {branch}")
        return br.get("head"), br

    def is_dirty(self, branch=None):
        head_id, _br = self.branch_head(branch)
        cur_hash = self.tree_hash(self.snapshot_fs())
        if not head_id:
            return bool(cur_hash != self.tree_hash({}))
        head = self._v()["commits"].get(head_id)
        return (head or {}).get("tree_hash") != cur_hash

    # ---------------------------------------------------------------- 提交
    def commit(self, message, author="admin", branch=None, allow_empty=False):
        message = (message or "").strip()
        if not message:
            raise VersionError("提交信息不能为空")
        with self.meta.lock:
            v = self._v()
            branch = branch or v.get("head_branch")
            head_id, br = self.branch_head(branch)
            snapshot = self.snapshot_fs()
            tree_hash = self.tree_hash(snapshot)
            if head_id:
                head = v["commits"].get(head_id)
                if head and head.get("tree_hash") == tree_hash and not allow_empty:
                    raise VersionError("工作区没有变化，无需提交")
                parents = [head_id]
                base_snap = head.get("snapshot", {})
            else:
                parents = []
                base_snap = {}

            stats = self._snapshot_change_stats(base_snap, snapshot)
            cid = self._commit_id(parents, tree_hash, message, author, snapshot)
            commit = {
                "id": cid,
                "parent_ids": parents,
                "message": message,
                "author": author,
                "ts": now(),
                "snapshot": snapshot,
                "tree_hash": tree_hash,
                "stats": stats,
                "conflicts": [],
            }
            v["commits"][cid] = commit
            br["head"] = cid
            v["head_branch"] = branch
            self.meta.touch("versions")
            self.nn.log_event("INFO", "version", "commit",
                              f"{branch}:{short_hash(cid)}", author,
                              f"提交 '{message}'：{stats['files_changed']} 个文件变更")
            return commit

    def _commit_id(self, parents, tree_hash, message, author, snapshot):
        payload = canonical_json({
            "parents": parents, "tree": tree_hash, "message": message,
            "author": author, "n": len(snapshot), "ts": now(),
        })
        return "c_" + sha256_text(payload)[:16]

    @staticmethod
    def _snapshot_change_stats(base, new):
        added = modified = deleted = 0
        bytes_delta = 0
        for p, e in new.items():
            b = base.get(p)
            if b is None:
                added += 1
                bytes_delta += e.get("size", 0)
            elif b.get("content_hash") != e.get("content_hash"):
                modified += 1
                bytes_delta += e.get("size", 0) - b.get("size", 0)
        for p, e in base.items():
            if p not in new:
                deleted += 1
                bytes_delta -= e.get("size", 0)
        return {"files_changed": added + modified + deleted,
                "added": added, "modified": modified, "deleted": deleted,
                "bytes_delta": bytes_delta}

    # ---------------------------------------------------------------- 分支
    def create_branch(self, name, from_ref=None, author="admin", desc=""):
        name = (name or "").strip()
        if not name or " " in name or "/" in name:
            raise VersionError("分支名不能为空且不含空格/斜杠")
        with self.meta.lock:
            v = self._v()
            if name in v["branches"]:
                raise VersionError(f"分支已存在: {name}")
            if from_ref:
                base_commit = self.resolve_ref(from_ref)
                head = base_commit["id"] if base_commit else None
            else:
                head, _ = self.branch_head()
            v["branches"][name] = {
                "name": name, "head": head, "created_at": now(),
                "created_by": author, "desc": desc or f"从 {from_ref or 'HEAD'} 创建",
            }
            self.meta.touch("versions")
            self.nn.log_event("INFO", "version", "branch_create", name, author,
                              f"创建分支 {name}（起点 {from_ref or 'HEAD'}）")
            return v["branches"][name]

    def delete_branch(self, name, author="admin"):
        with self.meta.lock:
            v = self._v()
            if name == v.get("head_branch"):
                raise VersionError("不能删除当前 HEAD 分支")
            if name == config.DEFAULT_BRANCH:
                raise VersionError("不能删除主分支")
            if name not in v["branches"]:
                raise VersionError(f"分支不存在: {name}")
            del v["branches"][name]
            self.meta.touch("versions")
            self.nn.log_event("WARN", "version", "branch_delete", name, author, "")

    def list_branches(self):
        with self.meta.lock:
            v = self._v()
            head_branch = v.get("head_branch")
            out = []
            for name, br in v["branches"].items():
                commit = v["commits"].get(br.get("head")) if br.get("head") else None
                out.append({
                    "name": name,
                    "head": br.get("head"),
                    "head_short": short_hash(br.get("head") or "", 8),
                    "desc": br.get("desc", ""),
                    "created_at": br.get("created_at"),
                    "created_by": br.get("created_by"),
                    "is_head": name == head_branch,
                    "commit_count": len(self.ancestors(br.get("head")))
                    if br.get("head") else 0,
                    "last_message": (commit or {}).get("message", ""),
                    "last_ts": (commit or {}).get("ts"),
                    "last_author": (commit or {}).get("author", ""),
                    "dirty": self.is_dirty(name) if name == head_branch else False,
                })
            out.sort(key=lambda b: (not b["is_head"], b["name"]))
            return {"head_branch": head_branch, "branches": out}

    def set_head_branch(self, branch):
        with self.meta.lock:
            v = self._v()
            if branch not in v["branches"]:
                raise VersionError(f"分支不存在: {branch}")
            v["head_branch"] = branch
            self.meta.touch("versions")

    # ---------------------------------------------------------------- 遍历
    def resolve_ref(self, ref):
        """ref 可以是分支名 / commit id / 短 id / HEAD。"""
        if not ref:
            return None
        with self.meta.lock:
            v = self._v()
            if ref == "HEAD":
                head_id, _ = self.branch_head()
                return v["commits"].get(head_id) if head_id else None
            br = v["branches"].get(ref)
            if br and br.get("head"):
                return v["commits"].get(br["head"])
            if ref in v["commits"]:
                return v["commits"][ref]
            for cid in v["commits"]:
                if cid.startswith(ref) or cid.endswith(ref):
                    return v["commits"][cid]
            return None

    def ancestors(self, cid, cap=2000):
        """commit 的全部祖先（含自身）id 集合。"""
        if not cid:
            return set()
        with self.meta.lock:
            commits = self._v()["commits"]
            seen = set()
            stack = [cid]
            while stack and len(seen) < cap:
                cur = stack.pop()
                if cur in seen or cur not in commits:
                    continue
                seen.add(cur)
                stack.extend(commits[cur].get("parent_ids", []))
            return seen

    def lca(self, cid1, cid2):
        """两个提交的最近公共祖先（按 ts 最大的公共祖先近似）。"""
        a1 = self.ancestors(cid1)
        a2 = self.ancestors(cid2)
        common = a1 & a2
        if not common:
            return None
        with self.meta.lock:
            commits = self._v()["commits"]
            best = max(common, key=lambda c: commits.get(c, {}).get("ts", 0))
            return best

    def list_commits(self, branch=None, limit=100, offset=0):
        with self.meta.lock:
            v = self._v()
            head_id, _ = self.branch_head(branch)
            ids = self.ancestors(head_id, cap=1000)
            commits = [v["commits"][c] for c in ids if c in v["commits"]]
            commits.sort(key=lambda c: c.get("ts", 0), reverse=True)
            # 分支引用标注
            refs = {}
            for bname, br in v["branches"].items():
                if br.get("head"):
                    refs.setdefault(br["head"], []).append(bname)
            out = []
            for c in commits[offset:offset + min(limit, config.MAX_COMMITS_PER_PAGE)]:
                out.append(self.commit_brief(c, refs.get(c["id"], [])))
            return {"total": len(commits), "commits": out,
                    "branch": branch or v.get("head_branch")}

    def commit_brief(self, c, refs=None):
        refs = refs if refs is not None else self._refs_of(c["id"])
        return {
            "id": c["id"],
            "short": short_hash(c["id"].replace("c_", ""), 8),
            "parent_ids": c.get("parent_ids", []),
            "parents_short": [short_hash(p.replace("c_", ""), 8)
                              for p in c.get("parent_ids", [])],
            "message": c.get("message", ""),
            "author": c.get("author", ""),
            "ts": c.get("ts"),
            "stats": c.get("stats", {}),
            "refs": refs,
            "is_merge": len(c.get("parent_ids", [])) > 1,
            "conflicts": c.get("conflicts", []),
            "tree_hash": short_hash(c.get("tree_hash", ""), 10),
        }

    def _refs_of(self, cid):
        v = self._v()
        return [b for b, br in v["branches"].items() if br.get("head") == cid]

    def get_commit(self, ref):
        c = self.resolve_ref(ref)
        if not c:
            raise VersionError(f"引用不存在: {ref}")
        return c

    # ---------------------------------------------------------------- 差异
    def diff_refs(self, ref_a, ref_b):
        """
        两个引用（提交/分支）之间的文件级差异。
        文本文件计算行级 adds/dels（带缓存）；二进制只报大小变化。
        """
        ca = self.get_commit(ref_a) if ref_a else None
        cb = self.get_commit(ref_b)
        snap_a = (ca or {}).get("snapshot", {})
        snap_b = cb.get("snapshot", {})
        result = self._diff_snapshots(snap_a, snap_b)
        result["a"] = (ca or {}).get("id")
        result["b"] = cb.get("id")
        result["a_label"] = ref_a or "(空)"
        result["b_label"] = ref_b
        return result

    def diff_working(self, branch=None):
        """工作区（活动文件系统）与 HEAD 提交的差异。"""
        head_id, _ = self.branch_head(branch)
        head = self._v()["commits"].get(head_id) if head_id else None
        snap_a = (head or {}).get("snapshot", {})
        snap_b = self.snapshot_fs()
        result = self._diff_snapshots(snap_a, snap_b)
        result["a"] = head_id
        result["b"] = "WORKING"
        result["a_label"] = (head_id or "(空)")[:12]
        result["b_label"] = "工作区"
        result["dirty"] = bool(result["changes"])
        return result

    def commit_preview(self, branch=None):
        """提交前影响分析：变更文件/行、目录，以及其它分支对同一文件的改动。"""
        branch = branch or self._v().get("head_branch", config.DEFAULT_BRANCH)
        diff = self.diff_working(branch)
        risks = []
        changed_paths = {c["path"] for c in diff["changes"]}
        current_snapshot = self.snapshot_fs()
        risks.extend(self._parallel_branch_risks(branch, changed_paths,
                                                 exclude_head=True,
                                                 current_snapshot=current_snapshot))
        for change in diff["changes"]:
            entry = current_snapshot.get(change["path"])
            risk = self._conflict_marker_risk(change["path"], entry)
            if risk:
                risks.append(risk)
        risks.sort(key=lambda r: (-self._risk_weight(r["severity"]), r["path"]))
        return {
            "mode": "commit",
            "branch": branch,
            "head_branch": branch,
            "source_branch": None,
            "dirty": diff["dirty"],
            "changes": diff["changes"],
            "stats": self._impact_stats(diff["changes"], risks),
            "directories": self._directory_impact(diff["changes"]),
            "risks": risks,
            "parallel_branches": self._parallel_branch_summary(
                branch, changed_paths, exclude_head=True,
                current_snapshot=current_snapshot),
            "summary": self._impact_summary("commit", diff["changes"], risks),
        }

    def merge_preview(self, source_ref, target_branch=None):
        """只读预演合并，返回完整影响清单和具体冲突风险，不写工作区/版本树。"""
        v = self._v()
        target_branch = target_branch or v.get("head_branch")
        head_id, _tbr = self.branch_head(target_branch)
        theirs = self.resolve_ref(source_ref)
        if not theirs:
            raise VersionError(f"源引用不存在: {source_ref}")
        if source_ref == target_branch:
            raise VersionError("源分支与目标分支相同")
        ours = v["commits"].get(head_id) if head_id else None
        active_branch = v.get("head_branch")
        if target_branch == active_branch:
            working = self.diff_working(target_branch)
        else:
            working = {"changes": [], "stats": {"files": 0, "adds": 0, "dels": 0},
                       "dirty": False}

        if not ours:
            incoming = self._diff_snapshots({}, theirs.get("snapshot", {}))
            kind = "fast-forward"
            base_id = None
            conflicts = []
            merged_snap = theirs.get("snapshot", {})
            risks = self._dirty_overwrite_risks(working["changes"],
                                                set(theirs.get("snapshot", {})),
                                                kind)
        else:
            base_id = self.lca(ours["id"], theirs["id"])
            base_kind = classify_merge_base(base_id, ours["id"],
                                            theirs["id"],
                                            config.MERGE_BASE_POLICY)
            base_snap = (v["commits"].get(base_id) or {}).get("snapshot", {}) \
                if base_id else {}
            if base_kind == "noop":
                incoming = self._diff_snapshots(ours.get("snapshot", {}),
                                                theirs.get("snapshot", {}))
                kind = "noop"
                conflicts = []
                merged_snap = ours.get("snapshot", {})
                risks = []
                if working["changes"]:
                    dirty_paths = ", ".join(
                        c["path"] for c in working["changes"][:8])
                    if len(working["changes"]) > 8:
                        dirty_paths += f" 等 {len(working['changes'])} 个"
                    risks.append({
                        "severity": "medium", "path": dirty_paths,
                        "reason": f"源分支 {source_ref} 已包含在 {target_branch} 中；"
                                  f"未提交文件 {dirty_paths} 不会被本次合并带入，"
                                  "建议先提交或暂存。",
                        "suggestion": "先提交这些文件，或取消合并继续当前修改。",
                    })
            elif base_kind == "fast-forward":
                incoming = self._diff_snapshots(ours.get("snapshot", {}),
                                                theirs.get("snapshot", {}))
                kind = "fast-forward"
                conflicts = []
                merged_snap = theirs.get("snapshot", {})
                risks = self._dirty_overwrite_risks(working["changes"],
                                                    set(merged_snap), kind)
            else:
                kind = "merge"
                risks = []
                incoming = self._diff_snapshots(base_snap,
                                                theirs.get("snapshot", {}))
                plan, conflicts = self._merge_snapshots(
                    base_snap, ours.get("snapshot", {}), theirs.get("snapshot", {}),
                    ours_label=target_branch, theirs_label=source_ref)
                merged_snap = self._planned_snapshot(
                    ours.get("snapshot", {}), plan, preview=True)
                if working["changes"]:
                    # 真实 merge 会先把工作区提交到目标分支。预检时先模拟这个
                    # “自动提交后的 ours”，再以它为 base 合并入站结果，避免把
                    # 同一份本地修改重复计算成一次冲突。
                    working_snap = self.snapshot_fs()
                    auto_plan, _auto_local_conflicts = self._merge_snapshots(
                        ours.get("snapshot", {}), ours.get("snapshot", {}),
                        working_snap,
                        ours_label="HEAD", theirs_label="工作区")
                    committed_snap = self._planned_snapshot(
                        ours.get("snapshot", {}), auto_plan, preview=True)
                    incoming_plan, incoming_conflicts = self._merge_snapshots(
                        base_snap, committed_snap, theirs.get("snapshot", {}),
                        ours_label="自动提交", theirs_label=source_ref)
                    merged_snap = self._planned_snapshot(
                        committed_snap, incoming_plan, preview=True)
                    conflicts = self._prefix_conflicts(
                        incoming_conflicts, "未提交修改与入站改动冲突：")
                    dirty_paths = ", ".join(
                        c["path"] for c in working["changes"][:8])
                    if len(working["changes"]) > 8:
                        dirty_paths += f" 等 {len(working['changes'])} 个"
                    risks.append({
                        "severity": "medium", "path": dirty_paths,
                        "reason": f"执行合并会先自动提交 {len(working['changes'])} 个未提交文件："
                                  f"{dirty_paths}。操作范围从一次分支合并扩大为“自动提交 + 合并”。",
                        "suggestion": "先显式提交并复跑预检，确认这些文件的修改符合预期。",
                    })
                risks = self._merge_conflict_risks(conflicts)

        merge_changes = self._diff_snapshots(ours.get("snapshot", {}) if ours else {},
                                             merged_snap)
        all_paths = {c["path"] for c in merge_changes["changes"]}
        risks.extend(self._parallel_branch_risks(
            target_branch, all_paths,
            exclude_heads={head_id, theirs.get("id")} if head_id else {theirs.get("id")},
            current_snapshot=merged_snap))
        risks.sort(key=lambda r: (-self._risk_weight(r["severity"]), r["path"]))
        return {
            "mode": "merge",
            "kind": kind,
            "branch": target_branch,
            "head_branch": target_branch,
            "source_branch": source_ref,
            "target_branch": target_branch,
            "base": base_id,
            "base_short": short_hash(base_id or "", 8),
            "ours": head_id,
            "theirs": theirs.get("id"),
            "dirty": working["dirty"],
            "working_changes": working["changes"],
            "incoming_changes": incoming["changes"],
            "changes": merge_changes["changes"],
            "stats": self._impact_stats(merge_changes["changes"], risks,
                                        working_changes=working["changes"],
                                        incoming_changes=incoming["changes"]),
            "directories": self._directory_impact(merge_changes["changes"]),
            "risks": risks,
            "conflicts": conflicts,
            "parallel_branches": self._parallel_branch_summary(
                target_branch, all_paths,
                exclude_heads={head_id, theirs.get("id")} if head_id else {theirs.get("id")},
                current_snapshot=merged_snap),
            "summary": self._impact_summary("merge", merge_changes["changes"], risks),
        }

    def _diff_snapshots(self, snap_a, snap_b):
        changes = []
        total_adds = total_dels = 0
        paths = sorted(set(snap_a) | set(snap_b))
        for p in paths:
            ea, eb = snap_a.get(p), snap_b.get(p)
            if ea and eb and ea.get("content_hash") == eb.get("content_hash"):
                continue
            if ea and not eb:
                kind = "deleted"
            elif eb and not ea:
                kind = "added"
            else:
                kind = "modified"
            change = {
                "path": p, "kind": kind,
                "old_size": (ea or {}).get("size", 0),
                "new_size": (eb or {}).get("size", 0),
                "mime": (eb or ea or {}).get("mime", ""),
                "old_hash": short_hash((ea or {}).get("content_hash", ""), 10),
                "new_hash": short_hash((eb or {}).get("content_hash", ""), 10),
            }
            line_stats = self._line_diff_stats(ea, eb)
            if line_stats:
                change.update(line_stats)
                total_adds += line_stats.get("adds", 0)
                total_dels += line_stats.get("dels", 0)
            changes.append(change)
        return {
            "changes": changes,
            "stats": {"files": len(changes), "adds": total_adds,
                      "dels": total_dels,
                      "added": sum(1 for c in changes if c["kind"] == "added"),
                      "modified": sum(1 for c in changes if c["kind"] == "modified"),
                      "deleted": sum(1 for c in changes if c["kind"] == "deleted")},
        }

    def _line_diff_stats(self, ea, eb):
        """文本文件行级增删统计与受影响行段（内容按块读取，缓存按内容哈希对）。"""
        if not ea and not eb:
            return None
        mime = (ea or eb or {}).get("mime", "")
        if not is_text_mime(mime):
            return None
        entries = [e for e in (ea, eb) if e]
        if max(e.get("size", 0) for e in entries) > config.MERGE_MAX_TEXT_BYTES:
            return {"binary": True}
        key = ((ea or {}).get("content_hash"), (eb or {}).get("content_hash"),
               (ea or {}).get("size", 0), (eb or {}).get("size", 0))
        cached = self._diff_cache.get(key)
        if cached is not None:
            return cached
        try:
            data_a = ea.get("__preview_data__")
            if data_a is None:
                data_a = self.nn.read_blocks(ea.get("block_ids", [])) if ea else b""
            data_b = eb.get("__preview_data__")
            if data_b is None:
                data_b = self.nn.read_blocks(eb.get("block_ids", [])) if eb else b""
            if looks_binary(data_a) or looks_binary(data_b):
                res = {"binary": True}
            else:
                la, lb = split_lines(decode_text(data_a) or ""), \
                         split_lines(decode_text(data_b) or "")
                if len(la) + len(lb) > config.DIFF_MAX_LINES:
                    res = {"binary": False, "too_large": True}
                else:
                    ops = diff_opcodes(la, lb)
                    st = diff_stats(ops)
                    hunks = []
                    for tag, i1, i2, j1, j2 in ops:
                        if tag == "equal":
                            continue
                        hunks.append({
                            "kind": tag,
                            "old_start": i1 + 1,
                            "old_end": i2,
                            "new_start": j1 + 1,
                            "new_end": j2,
                            "old_count": i2 - i1,
                            "new_count": j2 - j1,
                        })
                    res = {"adds": st["adds"], "dels": st["dels"],
                           "similarity": st["similarity"], "hunks": hunks}
        except Exception:
            res = None
        if res is not None:
            self._diff_cache.put(key, res)
        return res

    # -------------------------------------------------------- 影响分析辅助
    @staticmethod
    def _risk_weight(level):
        return {"high": 3, "medium": 2, "low": 1}.get(level, 0)

    @staticmethod
    def _directory_impact(changes):
        dirs = {}
        for ch in changes:
            path = ch.get("path", "")
            parent = posixpath.dirname(path) or "/"
            touched = set()
            cur = parent
            while True:
                touched.add(cur)
                if cur == "/":
                    break
                cur = posixpath.dirname(cur) or "/"
            for d in touched:
                item = dirs.setdefault(d, {
                    "path": d, "files": 0, "added": 0, "modified": 0,
                    "deleted": 0, "adds": 0, "dels": 0})
                item["files"] += 1
                kind = ch.get("kind")
                if kind in ("added", "modified", "deleted"):
                    item[kind] += 1
                item["adds"] += ch.get("adds", 0) or 0
                item["dels"] += ch.get("dels", 0) or 0
        return sorted(dirs.values(), key=lambda x: (-x["files"], x["path"]))

    @staticmethod
    def _impact_stats(changes, risks, working_changes=None, incoming_changes=None):
        def _add_stats(items):
            return {
                "files": len(items or []),
                "added": sum(1 for c in items or [] if c.get("kind") == "added"),
                "modified": sum(1 for c in items or [] if c.get("kind") == "modified"),
                "deleted": sum(1 for c in items or [] if c.get("kind") == "deleted"),
                "adds": sum(c.get("adds", 0) or 0 for c in items or []),
                "dels": sum(c.get("dels", 0) or 0 for c in items or []),
            }
        stats = _add_stats(changes)
        stats["risk_total"] = len(risks)
        stats["risk_high"] = sum(1 for r in risks if r.get("severity") == "high")
        stats["risk_medium"] = sum(1 for r in risks if r.get("severity") == "medium")
        stats["risk_low"] = sum(1 for r in risks if r.get("severity") == "low")
        if working_changes is not None:
            stats["working"] = _add_stats(working_changes)
        if incoming_changes is not None:
            stats["incoming"] = _add_stats(incoming_changes)
        return stats

    def _impact_summary(self, mode, changes, risks):
        high = sum(1 for r in risks if r.get("severity") == "high")
        medium = sum(1 for r in risks if r.get("severity") == "medium")
        action = "提交" if mode == "commit" else "合并"
        if high:
            return f"{action}将影响 {len(changes)} 个文件；发现 {high} 个高风险点，需先处理。"
        if medium:
            return f"{action}将影响 {len(changes)} 个文件；发现 {medium} 个需确认项。"
        return f"{action}将影响 {len(changes)} 个文件，未发现具体文件冲突。"

    def _planned_snapshot(self, base_snap, plan, preview=False):
        out = {}
        for p, (action, entry, data) in plan.items():
            if action == "delete":
                continue
            if action == "reuse":
                if entry:
                    out[p] = dict(entry)
            elif action == "write":
                new_entry = dict(entry or {})
                if preview and data is not None:
                    new_entry["__preview_data__"] = data
                    new_entry["size"] = len(data)
                    new_entry["content_hash"] = sha256_bytes(data)
                out[p] = new_entry
        return out

    @staticmethod
    def _prefix_conflicts(conflicts, prefix):
        out = []
        for c in conflicts:
            x = dict(c)
            x["detail"] = prefix + x.get("detail", "")
            out.append(x)
        return out

    @staticmethod
    def _line_label(start, end):
        if not start and not end:
            return "-"
        if start == end:
            return str(start)
        return f"{start}-{end}"

    def _merge_conflict_risks(self, conflicts):
        risks = []
        for c in conflicts:
            hunks = c.get("hunks") or []
            if hunks:
                h = hunks[0]
                parts = [f"base 第 {self._line_label(h.get('base_start'), h.get('base_end'))} 行"]
                if h.get("ours_start"):
                    parts.append(f"目标侧第 {self._line_label(h.get('ours_start'), h.get('ours_end'))} 行")
                if h.get("theirs_start"):
                    parts.append(f"源侧第 {self._line_label(h.get('theirs_start'), h.get('theirs_end'))} 行")
                lines = "，".join(parts)
            else:
                lines = "文件级"
            severity = "high"
            reason = f"{lines}：{c.get('detail', '两侧修改无法自动合并')}"
            if c.get("kind") == "binary":
                reason = "二进制或超大文件无法做行级三方合并，当前策略会直接保留目标分支版本。"
            elif c.get("kind") == "modify/delete":
                reason = "一侧删除文件，另一侧修改文件；自动结果会保留被修改版本，删除意图可能丢失。"
            risks.append({
                "severity": severity,
                "path": c.get("path"),
                "kind": c.get("kind"),
                "reason": reason,
                "suggestion": "打开差异页逐段确认；统一两边意图后再执行合并。",
                "hunks": hunks,
            })
        return risks

    def _dirty_overwrite_risks(self, working_changes, target_paths, kind):
        if not working_changes:
            return []
        touched = [c for c in working_changes if c.get("path") in target_paths]
        if not touched:
            names = "、".join(c["path"] for c in working_changes[:5])
            more = f" 等 {len(working_changes)} 个" if len(working_changes) > 5 else ""
            return [{
                "severity": "medium", "path": names,
                "reason": f"当前有 {len(working_changes)} 个未提交文件（{names}{more}）；"
                          f"本次 {kind} 不直接覆盖这些文件，但工作区仍混杂在合并动作中。",
                "suggestion": "先提交工作区，避免合并后难以区分改动来源。",
            }]
        names = "、".join(c["path"] for c in touched[:5])
        more = f" 等 {len(touched)} 个" if len(touched) > 5 else ""
        return [{
            "severity": "high", "path": ", ".join(c["path"] for c in touched),
            "reason": f"快进合并会直接物化源分支快照，未提交文件 {names}{more} 将被覆盖。",
            "suggestion": "先提交或撤回这些本地改动，再执行快进合并。",
        }]

    def _conflict_marker_risk(self, path, entry):
        if not entry or not is_text_mime(entry.get("mime", "")):
            return None
        if entry.get("size", 0) > config.MERGE_MAX_TEXT_BYTES:
            return None
        try:
            data = self.nn.read_blocks(entry.get("block_ids", []))
            text = decode_text(data) or ""
        except Exception:
            return None
        markers = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.startswith("<<<<<<< ") or line.startswith(">>>>>>> "):
                markers.append(n)
                if len(markers) >= 10:
                    break
        if not markers:
            return None
        shown = ", ".join(str(n) for n in markers[:5])
        return {
            "severity": "high", "path": path,
            "reason": f"文件仍含未解决的冲突标记（第 {shown} 行附近），提交会把冲突状态固化到历史。",
            "suggestion": "先删除冲突标记并合并两侧内容，再提交。",
        }

    def _branch_changed_paths(self, head_id, base_id):
        commits = self._v()["commits"]
        head = commits.get(head_id) or {}
        base = commits.get(base_id) or {}
        diff = self._diff_snapshots(base.get("snapshot", {}),
                                    head.get("snapshot", {}))
        return {c["path"]: c for c in diff["changes"]}

    def _parallel_branch_risks(self, current_branch, changed_paths,
                               exclude_head=False, exclude_heads=None,
                               current_snapshot=None):
        v = self._v()
        commits = v["commits"]
        current_head, _ = self.branch_head(current_branch)
        skip_ancestor_ids = self.ancestors(current_head) if current_head else set()
        exclude_heads = set(exclude_heads or ())
        if exclude_head and current_head:
            exclude_heads.add(current_head)
        for excluded in exclude_heads:
            skip_ancestor_ids |= self.ancestors(excluded)
        risks = []
        for name, br in v["branches"].items():
            if name == current_branch:
                continue
            other_head = br.get("head")
            if not other_head or other_head in exclude_heads:
                continue
            if other_head in skip_ancestor_ids:
                continue
            base_id = self.lca(current_head, other_head) if current_head else None
            base_snap = (commits.get(base_id) or {}).get("snapshot", {}) \
                if base_id else {}
            other_changes = self._branch_changed_paths(other_head, base_id)
            overlap = sorted(changed_paths & set(other_changes))
            actual_conflicts = {}
            if overlap:
                simulated_plan, simulated_conflicts = self._merge_snapshots(
                    base_snap, current_snapshot,
                    commits.get(other_head, {}).get("snapshot", {}),
                    ours_label=current_branch, theirs_label=name)
                actual_conflicts = {c["path"]: c for c in simulated_conflicts}
            for path in overlap:
                current_entry = self._current_path_entry(path, current_branch,
                                                          changed_paths,
                                                          current_snapshot)
                other_change = other_changes[path]
                other_entry = commits.get(other_head, {}).get("snapshot", {}).get(path)
                severity, reason, suggestion = self._parallel_file_risk(
                    path, current_entry, other_entry, other_change, name, base_id,
                    actual_conflicts.get(path))
                risks.append({
                    "severity": severity, "path": path,
                    "branches": [current_branch, name],
                    "reason": reason, "suggestion": suggestion,
                })
        return risks

    def _current_path_entry(self, path, branch, changed_paths, current_snapshot=None):
        if current_snapshot is not None:
            return current_snapshot.get(path)
        change = changed_paths.get(path) if isinstance(changed_paths, dict) else None
        if change:
            snap = self.snapshot_fs()
            return snap.get(path)
        head_id, _ = self.branch_head(branch)
        head = self._v()["commits"].get(head_id) if head_id else None
        return (head or {}).get("snapshot", {}).get(path)

    def _parallel_file_risk(self, path, ours_entry, theirs_entry,
                            their_change, other_branch, base_id,
                            actual_conflict=None):
        if ours_entry is None or theirs_entry is None:
            return ("high",
                    f"与分支 {other_branch} 出现修改/删除分叉："
                    f"{'当前侧删除' if ours_entry is None else f'{other_branch} 删除'}，"
                    "另一侧仍保留修改。",
                    "先与该分支确认应删除还是保留，再提交/合并。")
        ours_h = ours_entry.get("content_hash")
        theirs_h = theirs_entry.get("content_hash")
        if ours_h == theirs_h:
            return ("low",
                    f"分支 {other_branch} 也改了同一文件，但两边内容哈希一致；"
                    "合并时通常可自动收敛。",
                    "合并后复核一次即可。")
        mime = ours_entry.get("mime") or theirs_entry.get("mime") or ""
        if not is_text_mime(mime):
            return ("high",
                    f"分支 {other_branch} 同时修改了该二进制/非文本文件，"
                    "系统无法做行级自动合并。",
                    "让一侧基于另一侧最新版本重新生成文件，或人工选择版本。")
        base_entry = (self._v()["commits"].get(base_id) or {}) \
            .get("snapshot", {}).get(path) if base_id else None
        if base_entry is None:
            return ("high",
                    f"分支 {other_branch} 与当前分支都新增了同一路径，但内容不同（add/add）。",
                    "双方共同确认文件内容，或把其中一侧迁移到不同路径。")
        if base_entry.get("content_hash") == ours_h:
            return ("low",
                    f"只有分支 {other_branch} 相对共同祖先改动该文件；当前工作区改动可能来自未同步基线。",
                    "先合并该分支最新内容后复跑预检。")
        if base_entry.get("content_hash") == theirs_h:
            return ("low", "当前侧改动相对共同祖先独占，另一分支尚未修改。", "正常提交即可。")
        if actual_conflict:
            hunks = actual_conflict.get("hunks") or []
            line_part = "文件级冲突"
            if hunks:
                h = hunks[0]
                line_part = (f"共同版本第 {self._line_label(h.get('base_start'), h.get('base_end'))} "
                             "行附近冲突")
            return ("high",
                    f"分支 {other_branch} 也修改了该文件；三方预演已确认 {line_part}，"
                    f"类型为 {actual_conflict.get('kind', 'content')}。",
                    "先合并该分支或协调同一文件，提前解决后再提交。")
        return ("low",
                f"分支 {other_branch} 也修改了该文件，但三方预演可自动合并；"
                "当前改动落在不同文本区域。",
                "合并后回归该文件相关功能即可。")

    def _parallel_branch_summary(self, current_branch, changed_paths,
                                 exclude_head=False, exclude_heads=None,
                                 current_snapshot=None):
        risks = self._parallel_branch_risks(
            current_branch, changed_paths, exclude_head=exclude_head,
            exclude_heads=exclude_heads, current_snapshot=current_snapshot)
        out = {}
        for r in risks:
            for b in r.get("branches", []):
                if b == current_branch:
                    continue
                item = out.setdefault(b, {"branch": b, "high": 0,
                                          "medium": 0, "low": 0,
                                          "paths": set()})
                item[r["severity"]] += 1
                item["paths"].add(r["path"])
        return [{
            "branch": b, "high": x["high"], "medium": x["medium"],
            "low": x["low"], "paths": sorted(x["paths"]),
        } for b, x in sorted(out.items())]

    def file_at(self, ref, path):
        """读取某引用下某文件的内容（历史版本读取）。"""
        c = self.get_commit(ref)
        entry = c.get("snapshot", {}).get(path)
        if not entry:
            return {"exists": False, "path": path, "ref": c["id"]}
        data = self.nn.read_blocks(entry.get("block_ids", []))
        text = decode_text(data)
        return {
            "exists": True, "path": path, "ref": c["id"],
            "size": entry.get("size", 0),
            "content_hash": entry.get("content_hash"),
            "mime": entry.get("mime"),
            "is_text": text is not None and not looks_binary(data),
            "content": text if text is not None and not looks_binary(data) else None,
        }

    def file_history(self, path, branch=None, limit=50):
        """某文件在分支历史中的版本序列。"""
        with self.meta.lock:
            head_id, _ = self.branch_head(branch)
            ids = self.ancestors(head_id, cap=1000)
            commits = sort_by_ts([self._v()["commits"][c] for c in ids
                                  if c in self._v()["commits"]],
                                 config.HISTORY_ORDER)
            out = []
            last_hash = None
            for c in commits:
                e = c.get("snapshot", {}).get(path)
                h = (e or {}).get("content_hash")
                if e is None:
                    if last_hash is not None:
                        out.append({"commit": self.commit_brief(c),
                                    "state": "deleted", "hash": None})
                        last_hash = None
                    continue
                if h != last_hash:
                    out.append({"commit": self.commit_brief(c),
                                "state": "changed", "hash": h,
                                "size": e.get("size", 0)})
                    last_hash = h
                if len(out) >= limit:
                    break
            return out

    # ---------------------------------------------------------------- 合并
    def merge(self, source_ref, target_branch=None, author="admin"):
        """
        把 source_ref（分支/提交）合并进 target_branch（默认 HEAD 分支）。
        返回 {"kind": "noop"|"fast-forward"|"merge", ...}
        """
        with self.meta.lock:
            v = self._v()
            target_branch = target_branch or v.get("head_branch")
            head_id, tbr = self.branch_head(target_branch)
            theirs = self.resolve_ref(source_ref)
            if not theirs:
                raise VersionError(f"源引用不存在: {source_ref}")
            if source_ref == target_branch:
                raise VersionError("源分支与目标分支相同")
            ours = v["commits"].get(head_id) if head_id else None

            if not ours:
                # 目标分支还没有提交：直接快进
                tbr["head"] = theirs["id"]
                self._materialize(theirs.get("snapshot", {}), author)
                self.meta.touch("versions")
                return {"kind": "fast-forward", "commit": self.commit_brief(theirs),
                        "conflicts": [], "stats": theirs.get("stats", {})}

            base_id = self.lca(ours["id"], theirs["id"])
            base_kind = classify_merge_base(base_id, ours["id"],
                                            theirs["id"],
                                            config.MERGE_BASE_POLICY)
            if base_kind == "noop":
                return {"kind": "noop", "message": "已经是最新（源分支被目标包含）"}
            if base_kind == "fast-forward":
                # 快进合并：物化 theirs 快照
                tbr["head"] = theirs["id"]
                self._materialize(theirs.get("snapshot", {}), author)
                self.meta.touch("versions")
                self.nn.log_event("INFO", "version", "merge_ff",
                                  f"{source_ref} -> {target_branch}", author,
                                  "快进合并")
                return {"kind": "fast-forward",
                        "commit": self.commit_brief(theirs),
                        "conflicts": [], "stats": theirs.get("stats", {})}

            # 工作区脏 => 先自动提交，保证可回退
            if self.is_dirty(target_branch):
                self.commit(f"auto: 合并 {source_ref} 前的工作区快照",
                            author, target_branch)
                head_id, tbr = self.branch_head(target_branch)
                ours = v["commits"][head_id]

            base_snap = (v["commits"].get(base_id) or {}).get("snapshot", {})
            ours_snap = ours.get("snapshot", {})
            theirs_snap = theirs.get("snapshot", {})

            plan, conflicts = self._merge_snapshots(
                base_snap, ours_snap, theirs_snap,
                ours_label=target_branch, theirs_label=source_ref)

            # 物化合并结果
            self._materialize({p: e for p, (action, e, _data) in plan.items()
                               if action == "reuse"}, author)
            merged_snap = {}
            for p, (action, entry, data) in plan.items():
                if action == "delete":
                    continue
                if action == "reuse":
                    merged_snap[p] = entry
                elif action == "write":
                    info = self.nn.write_file_internal(
                        p, data, author, mime=entry.get("mime"))
                    merged_snap[p] = {
                        "type": "file", "size": info["size"],
                        "content_hash": info["content_hash"],
                        "block_ids": info["block_ids"],
                        "mime": info["mime"],
                        "owner": entry.get("owner", author),
                        "mode": entry.get("mode", "rw-r--r--"),
                        "inode": info["inode_id"],
                    }
            stats = self._snapshot_change_stats(ours_snap, merged_snap)
            cid = self._commit_id([ours["id"], theirs["id"]],
                                  self.tree_hash(merged_snap),
                                  f"merge {source_ref}", author, merged_snap)
            commit = {
                "id": cid,
                "parent_ids": [ours["id"], theirs["id"]],
                "message": f"Merge '{source_ref}' into '{target_branch}'",
                "author": author, "ts": now(),
                "snapshot": merged_snap,
                "tree_hash": self.tree_hash(merged_snap),
                "stats": stats,
                "conflicts": conflicts,
                "merge_info": {
                    "source": source_ref, "target": target_branch,
                    "base": base_id,
                    "source_head": theirs["id"], "ours_head": ours["id"],
                    "clean": not conflicts,
                },
            }
            v["commits"][cid] = commit
            tbr["head"] = cid
            self.meta.touch("versions")
            self.nn.log_event(
                "WARN" if conflicts else "INFO", "version", "merge",
                f"{source_ref} -> {target_branch}", author,
                f"合并完成：{stats['files_changed']} 文件变更，"
                f"{len(conflicts)} 处冲突")
            return {"kind": "merge", "commit": self.commit_brief(commit),
                    "conflicts": conflicts, "stats": stats,
                    "base": short_hash(base_id, 8)}

    def _merge_snapshots(self, base, ours, theirs, ours_label, theirs_label):
        """
        快照级三方合并。返回 (plan, conflicts)：
          plan[path] = (action, entry, data)
            action: reuse（直接引用旧块）| write（写新内容）| delete
          conflicts: [{"path","kind","detail"}]
        """
        plan = {}
        conflicts = []
        paths = sorted(set(base) | set(ours) | set(theirs))

        def same(x, y):
            if x is None or y is None:
                return x is None and y is None
            return x.get("content_hash") == y.get("content_hash")

        for p in paths:
            b, o, t = base.get(p), ours.get(p), theirs.get(p)
            if o is None and t is None:
                plan[p] = ("delete", None, None)
                continue
            if same(o, t):
                if o is not None:
                    plan[p] = ("reuse", o, None)
                continue
            if same(b, o):            # 我方未动 => 采纳对方（含删除）
                if t is not None:
                    plan[p] = ("reuse", t, None)
                else:
                    plan[p] = ("delete", None, None)
                continue
            if same(b, t):            # 对方未动 => 保留我方
                plan[p] = ("reuse", o, None)
                continue
            # 双侧异改
            if o is None or t is None:
                keep = o if o is not None else t
                side = "ours" if o is not None else "theirs"
                conflicts.append({
                    "path": p, "kind": "modify/delete",
                    "detail": f"一侧删除、另一侧修改；保留 {side} 版本"})
                plan[p] = ("reuse", keep, None)
                continue
            mime = o.get("mime") or t.get("mime") or ""
            too_big = max(o.get("size", 0), t.get("size", 0)) > config.MERGE_MAX_TEXT_BYTES
            if is_text_mime(mime) and not too_big:
                merged = self._merge_text_file(p, b, o, t,
                                               ours_label, theirs_label)
                if merged is not None:
                    text, file_conflicts, conflict_hunks = merged
                    plan[p] = ("write", dict(o), text.encode("utf-8"))
                    if file_conflicts:
                        conflicts.append({
                            "path": p, "kind": "content",
                            "detail": f"{file_conflicts} 处文本冲突，已写入冲突标记",
                            "hunks": conflict_hunks})
                    continue
            conflicts.append({"path": p, "kind": "binary",
                              "detail": "二进制/超大文件双侧修改，保留 ours"})
            plan[p] = ("reuse", o, None)
        return plan, conflicts

    def _merge_text_file(self, path, b, o, t, ours_label, theirs_label):
        """读取三方内容做 diff3 行级合并；失败返回 None（按二进制冲突处理）。"""
        try:
            base_data = (b or {}).get("__preview_data__")
            if base_data is None:
                base_data = self.nn.read_blocks((b or {}).get("block_ids", [])) if b else b""
            ours_data = o.get("__preview_data__")
            if ours_data is None:
                ours_data = self.nn.read_blocks(o.get("block_ids", []))
            theirs_data = t.get("__preview_data__")
            if theirs_data is None:
                theirs_data = self.nn.read_blocks(t.get("block_ids", []))
        except Exception:
            return None
        if looks_binary(ours_data) or looks_binary(theirs_data):
            return None
        base_text = decode_text(base_data) or ""
        ours_text = decode_text(ours_data) or ""
        theirs_text = decode_text(theirs_data) or ""
        result = merge3(split_lines(base_text), split_lines(ours_text),
                        split_lines(theirs_text),
                        ours_label=ours_label, theirs_label=theirs_label,
                        label_swap=merge_label_swap(
                            config.CONFLICT_LABEL_SWAP),
                        marker_ours=config.CONFLICT_MARKER_OURS,
                        marker_sep=config.CONFLICT_MARKER_SEP,
                        marker_theirs=config.CONFLICT_MARKER_THEIRS)
        hunks = []

        def side_ranges(lines_by_base):
            if not lines_by_base:
                return None, None
            starts = [x[0] for x in lines_by_base if x[0] is not None]
            ends = [x[1] for x in lines_by_base if x[1] is not None]
            if starts and ends:
                return min(starts) + 1, max(ends)
            return None, None

        for c in result.conflicts:
            ours_start, ours_end = side_ranges(c.get("ours_by_base"))
            theirs_start, theirs_end = side_ranges(c.get("theirs_by_base"))
            hunks.append({
                "base_start": c.get("base_start", 0) + 1,
                "base_end": c.get("base_end", 0),
                "ours_start": ours_start,
                "ours_end": ours_end,
                "theirs_start": theirs_start,
                "theirs_end": theirs_end,
            })
        return result.text, len(result.conflicts), hunks

    # ---------------------------------------------------------------- 检出
    def checkout(self, branch, author="admin", auto_commit=True):
        """
        切换 HEAD 分支并物化其头提交快照到活动文件系统。
        工作区有未提交修改时先自动提交（可追溯、防丢数据）。
        """
        with self.meta.lock:
            v = self._v()
            if branch not in v["branches"]:
                raise VersionError(f"分支不存在: {branch}")
            cur = v.get("head_branch")
            auto = None
            if cur == branch and not self.is_dirty(branch):
                return {"branch": branch, "auto_commit": None,
                        "message": "已在该分支且工作区干净"}
            if self.is_dirty(cur):
                if not auto_commit:
                    raise VersionError("工作区有未提交修改")
                auto = self.commit(f"auto: 切换到 {branch} 前的工作区快照",
                                   author, cur)
            v["head_branch"] = branch
            head_id = v["branches"][branch].get("head")
            snap = (v["commits"].get(head_id) or {}).get("snapshot", {}) \
                if head_id else {}
            self._materialize(snap, author)
            self.meta.touch("versions")
            self.nn.log_event("INFO", "version", "checkout", branch, author,
                              f"检出分支 {branch}（物化 {len(snap)} 个文件）")
            return {"branch": branch,
                    "auto_commit": self.commit_brief(auto) if auto else None,
                    "head": short_hash(head_id or "", 8),
                    "files": len(snap)}

    def _materialize(self, snapshot, author):
        """把快照物化为活动 inode 树：多余文件硬删除，缺失文件重建。"""
        fs = self.nn.fs
        with self.meta.lock:
            existing = {p: inode for p, inode in fs.all_files()}
            # 1. 删除不在快照中的文件
            for p, inode in existing.items():
                if p not in snapshot:
                    parent = fs.get_inode(inode.get("parent"))
                    if parent and inode["id"] in parent.get("children", []):
                        parent["children"].remove(inode["id"])
                    fs._remove_subtree(inode["id"])
            # 2. 创建/对齐快照中的文件
            for p in sorted(snapshot):
                entry = snapshot[p]
                if entry.get("type") != "file":
                    continue
                dir_path = "/".join(p.split("/")[:-1]) or "/"
                name = p.split("/")[-1]
                fs.mkdirs(dir_path, author)
                cur = existing.get(p)
                if cur and cur.get("content_hash") == entry.get("content_hash"):
                    continue
                fs.create_file(dir_path, name, entry.get("size", 0),
                               entry.get("content_hash"),
                               entry.get("block_ids", []),
                               entry.get("mime", ""),
                               entry.get("owner", author))
            # 3. 清理空目录（不属于任何快照路径前缀）
            prefixes = set()
            for p in snapshot:
                parts = [s for s in p.split("/") if s]
                for i in range(1, len(parts)):
                    prefixes.add("/" + "/".join(parts[:i]))
            self._prune_empty_dirs(fs, prefixes)
            self.meta.touch("fs")

    @staticmethod
    def _prune_empty_dirs(fs, keep_prefixes):
        inodes = fs._inodes()
        changed = True
        guard = 0
        while changed and guard < 50:
            changed = False
            guard += 1
            for iid, node in list(inodes.items()):
                if node["type"] != "dir":
                    continue
                if iid in (fs.root_id, fs.trash_id):
                    continue
                path = fs.path_of(iid)
                if path in keep_prefixes:
                    continue
                if not node.get("children"):
                    parent = inodes.get(node.get("parent"))
                    if parent and iid in parent.get("children", []):
                        parent["children"].remove(iid)
                    inodes.pop(iid, None)
                    changed = True

    # ---------------------------------------------------------------- 还原
    def restore_file(self, ref, path, author="admin"):
        """把历史版本中的单个文件恢复到活动文件系统。"""
        with self.meta.lock:
            c = self.get_commit(ref)
            entry = c.get("snapshot", {}).get(path)
            if not entry:
                raise VersionError(f"{ref} 中不存在文件: {path}")
            dir_path = "/".join(path.split("/")[:-1]) or "/"
            name = path.split("/")[-1]
            fs = self.nn.fs
            fs.mkdirs(dir_path, author)
            inode = fs.create_file(dir_path, name, entry.get("size", 0),
                                   entry.get("content_hash"),
                                   entry.get("block_ids", []),
                                   entry.get("mime", ""), author)
            self.nn.log_event("INFO", "version", "restore_file", path, author,
                              f"从 {short_hash(c['id'], 8)} 恢复文件")
            return {"path": path, "inode": inode["id"],
                    "from_commit": c["id"]}

    # ---------------------------------------------------------------- 图
    def graph(self, branch=None, limit=60):
        """时间线/图谱数据：提交列表 + 简单泳道分配（前端渲染）。"""
        with self.meta.lock:
            data = self.list_commits(branch, limit=limit)
            commits = data["commits"]
            lanes = {}
            next_lane = 0
            for c in commits:
                # 简单泳道：沿用第一个父的泳道，其余父分配新泳道
                parents = c.get("parent_ids", [])
                lane = None
                for p in parents:
                    if p in lanes:
                        lane = lanes[p]
                        break
                if lane is None:
                    lane = next_lane
                    next_lane += 1
                lanes[c["id"]] = lane
                c["lane"] = lane % 6
                for p in parents:
                    if p not in lanes:
                        lanes[p] = next_lane
                        next_lane += 1
            data["lane_count"] = min(max(lanes.values(), default=0) + 1, 6)
            return data

    def repo_stats(self):
        with self.meta.lock:
            v = self._v()
            commits = v["commits"]
            merges = sum(1 for c in commits.values()
                         if len(c.get("parent_ids", [])) > 1)
            conflicted = sum(1 for c in commits.values() if c.get("conflicts"))
            return {
                "commits": len(commits),
                "branches": len(v["branches"]),
                "head_branch": v.get("head_branch"),
                "merge_commits": merges,
                "conflicted_commits": conflicted,
                "first_ts": min((c.get("ts", now()) for c in commits.values()),
                                default=None),
            }

    def all_referenced_blocks(self):
        """全部提交快照引用到的块 id 集合（GC 保护集的一部分）。"""
        with self.meta.lock:
            refs = set()
            for c in self._v()["commits"].values():
                for e in c.get("snapshot", {}).values():
                    refs.update(e.get("block_ids", []))
            return refs
