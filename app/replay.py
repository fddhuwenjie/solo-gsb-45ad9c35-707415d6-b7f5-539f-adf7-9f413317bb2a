"""草稿特征库上线前的历史判定回放评估。

回放报告在**预约**（:meth:`app.db.Database.reserve_replay_report`）时冻结判定
水位 ``watermark = MAX(judgments.id)``，随后只重放该水位内**使用服务端特征库**
（``library_source == 'stored'``，记录保存了服务端版本号）的判定：

- 使用内联特征库（``library_source == 'inline'`` 或未保存版本号）的判定与草稿
  无关，按 ``inline_library`` 原因跳过、计数，不参与分类；
- 旧版迁移产生的无请求记录按 ``legacy_record`` 跳过。

每条参与回放的判定**沿用原观测与四项阈值**（容差、最大钟漂、漏记/误记上限），
改用草稿版本的不可变快照执行与 ``/judge`` 完全相同的
:func:`app.align.judge` 判定算法，得到新裁决后与原响应比较：

- ``unchanged``：前后都无候选，或原匹配且最佳航标不变；
- ``match_gained``：原无合格候选，新裁决匹配到航标；
- ``match_lost``：原匹配航标，新裁决无合格候选；
- ``winner_changed``：前后都匹配，但最佳航标编号改变。

回放只写 ``replay_reports`` / ``replay_report_items``，**不写入
``judgments``**；每条明细都保存旧、新最佳候选证据（候选详情或无候选原因）。

同键并发：第一个请求预约执行，其余请求短轮询等待同一份报告（键被其他版本
占用在预约事务内返回 409），因此并发不会生成重复报告。
"""
from __future__ import annotations

import time
from collections import Counter

from .align import AidPattern, judge
from .db import (
    Database,
    ReplayKeyConflict,
    VersionNotFound,
    VersionStateError,
)


class ReplayWaitTimeout(Exception):
    """同一幂等键的回放仍在执行，等待超时（HTTP 503，建议稍后重试）。"""

    def __init__(self, report_id: int):
        super().__init__(f"replay report {report_id} is still running; retry shortly")
        self.report_id = report_id


def _classify(old_matched: bool, new_matched: bool, old_aid: str | None,
              new_aid: str | None) -> str:
    if old_matched and new_matched:
        return "unchanged" if old_aid == new_aid else "winner_changed"
    if not old_matched and new_matched:
        return "match_gained"
    if old_matched and not new_matched:
        return "match_lost"
    # 前后都无合格候选：最佳航标未变（均无），计为 unchanged
    return "unchanged"


def _execute(db: Database, report_id: int, version_id: int, watermark: int,
             grid_max_steps: int, seed_top_k: int) -> dict:
    """取草稿快照与水位内判定，逐条重算分类并原子保存报告。"""
    aids = db.get_version_aids(version_id)
    patterns = [AidPattern(a["aid_id"], a["period"], tuple(a["flashes"])) for a in aids]
    judgments = db.fetch_judgments_for_replay(watermark)

    items: list[dict] = []
    skipped: Counter = Counter()
    for row in judgments:
        request = row["request"]
        response = row["response"]
        if (
            not isinstance(request, dict)
            or "observations" not in request
            or not isinstance(response, dict)
            or "parameters" not in response
        ):
            # 旧版迁移或早期无结构记录：没有可沿用的观测/阈值，无法回放
            skipped["legacy_record"] += 1
            continue
        if response.get("library_source") != "stored" or response.get("library_version") is None:
            # 内联库判定：与服务端草稿无关，按原因跳过
            skipped["inline_library"] += 1
            continue

        parameters = response.get("parameters") or {}
        observations = request.get("observations") or []
        best, _candidates, reason = judge(
            patterns,
            observations,
            tolerance=parameters["tolerance"],
            max_drift=parameters["max_drift"],
            max_missed=parameters["max_missed"],
            max_false=parameters["max_false"],
            grid_max_steps=grid_max_steps,
            seed_top_k=seed_top_k,
        )

        old_status = response.get("status") or ""
        old_matched = old_status == "matched"
        old_best = response.get("best")
        new_best = best.to_detail() if best is not None else None
        new_status = "matched" if best is not None else "no_candidate"
        change_type = _classify(
            old_matched,
            best is not None,
            (old_best or {}).get("aid_id") if old_best is not None else None,
            best.aid_id if best is not None else None,
        )
        items.append(
            {
                "judgment_id": row["id"],
                "change_type": change_type,
                "old_status": old_status,
                "new_status": new_status,
                "old_best_aid_id": (old_best or {}).get("aid_id")
                if old_best is not None else None,
                "new_best_aid_id": best.aid_id if best is not None else None,
                "old_best": old_best,
                "new_best": new_best,
                "old_reason": response.get("reason"),
                "new_reason": reason,
            }
        )

    return db.complete_replay_report(report_id, items, dict(skipped))


def start_replay_report(
    db: Database,
    version_id: int,
    idempotency_key: str,
    *,
    grid_max_steps: int = 201,
    seed_top_k: int = 5,
    wait_timeout: float = 30.0,
    poll_interval: float = 0.05,
) -> tuple[dict, bool]:
    """幂等地创建/取回回放报告。

    返回 ``(报告摘要, 是否本次新建)``。预约后在写事务外完成全部重算；同键的
    并发请求短轮询等待第一份报告（相同键重试返回原报告，状态码由调用方按
    ``created`` 决定 201/200）。等待 ``wait_timeout`` 秒仍在执行抛
    :class:`ReplayWaitTimeout`（503）。计算失败把报告标记 failed 并向上抛出，
    同键随后可重新预约重试。
    """
    report, state = db.reserve_replay_report(version_id, idempotency_key)
    while state != "created":
        if state == "completed":
            return report, False
        # 同版本同键的另一个请求正在回放：等待其完成，不另建报告；
        # 对方失败则用同键重新预约（failed 报告允许同版本重试）。
        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            current = db.get_replay_report(version_id, report["report_id"])
            if current is not None and current["status"] == "completed":
                return current, False
            if current is None or current["status"] == "failed":
                # 对方执行失败（或失败报告已被另一重试重置）：跳出并重新预约
                break
        else:
            raise ReplayWaitTimeout(report["report_id"])
        report, state = db.reserve_replay_report(version_id, idempotency_key)

    try:
        report = _execute(
            db,
            report["report_id"],
            version_id,
            report["watermark"],
            grid_max_steps=grid_max_steps,
            seed_top_k=seed_top_k,
        )
    except Exception as exc:
        db.fail_replay_report(report["report_id"], f"{type(exc).__name__}: {exc}")
        raise
    return report, True


__all__ = [
    "ReplayKeyConflict",
    "ReplayWaitTimeout",
    "VersionNotFound",
    "VersionStateError",
    "start_replay_report",
]
