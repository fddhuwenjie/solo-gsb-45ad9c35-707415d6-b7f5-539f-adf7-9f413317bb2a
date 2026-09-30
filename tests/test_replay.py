"""草稿特征库历史判定回放（replay-reports）测试。"""
import json
import os
import sys
import threading

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import create_app

PSI = 1_700_000_000.0

BASE_AIDS = [
    {"aid_id": "A", "period": 10.0, "flashes": [0.0, 2.0]},
    {"aid_id": "B", "period": 10.0, "flashes": [0.0]},
    {"aid_id": "C", "period": 6.0, "flashes": [0.0]},
]
# 草稿：删除 B，新增周期 5 的单闪光航标 Cp
DRAFT_AIDS = [
    {"aid_id": "A", "period": 10.0, "flashes": [0.0, 2.0]},
    {"aid_id": "C", "period": 6.0, "flashes": [0.0]},
    {"aid_id": "Cp", "period": 5.0, "flashes": [0.0]},
]


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_path=str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def synth(period, flashes, psi=PSI, n=8, drop=(), extra=()):
    obs = []
    for i in range(n):
        for j, f in enumerate(flashes):
            if (i, j) in drop:
                continue
            obs.append(psi + i * period + f)
    obs.extend(psi + e for e in extra)
    return sorted(obs)


def judge(client, obs, *, mm, mf, library=None):
    body = {
        "observations": obs,
        "tolerance": 0.05,
        "max_drift": 0.0,
        "max_missed": mm,
        "max_false": mf,
    }
    if library is not None:
        body["library"] = library
    r = client.post("/judge", json=body)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture()
def setup(client):
    """导入基础库 → 建草稿 → 产生 5 条服务端判定（覆盖四类）+ 1 条内联判定。"""
    client.post("/library/import", json={"aids": BASE_AIDS, "mode": "merge"})
    v1 = client.get("/library").json()["library_version"]
    draft = client.post("/library/versions", json={
        "aids": DRAFT_AIDS, "base_version": v1,
    }).json()

    j1 = judge(client, synth(10.0, [0.0]), mm=10, mf=2)          # 原 B → 草稿 A
    j2 = judge(client, synth(5.0, [0.0]), mm=0, mf=0)            # 原无候选 → 草稿 Cp
    j3 = judge(client, synth(10.0, [0.0]), mm=0, mf=0)           # 原 B → 草稿无候选
    j4 = judge(client, synth(10.0, [0.0], drop={(3, 0), (5, 0)}),
               mm=0, mf=0)                                       # 前后均无候选
    j5 = judge(client, synth(6.0, [0.0]), mm=2, mf=0)            # 前后均 C
    j_inline = judge(                                          # 内联库 → 跳过
        client, synth(10.0, [0.0]), mm=3, mf=3,
        library=[{"aid_id": "X", "period": 10.0, "flashes": [0.0]}],
    )
    assert j1["best"]["aid_id"] == "B"
    assert j2["status"] == "no_candidate"
    assert j3["best"]["aid_id"] == "B"
    assert j4["status"] == "no_candidate"
    assert j5["best"]["aid_id"] == "C"
    assert j_inline["library_source"] == "inline"
    return {
        "v1": v1,
        "draft_id": draft["version_id"],
        "ids": {
            "winner_changed": j1["judgment_id"],
            "match_gained": j2["judgment_id"],
            "match_lost": j3["judgment_id"],
            "none_none": j4["judgment_id"],
            "unchanged": j5["judgment_id"],
            "inline": j_inline["judgment_id"],
        },
    }


def _by_type(page):
    return {item["judgment_id"]: item for item in page["changes"]}


def test_replay_report_classification_and_evidence(client, setup):
    r = client.post(
        f"/library/versions/{setup['draft_id']}/replay-reports",
        json={"idempotency_key": "key-1"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["version_id"] == setup["draft_id"]
    assert body["idempotency_key"] == "key-1"
    # 冻结水位 = 当前最大 judgment_id（含内联判定）
    assert body["watermark"] == max(setup["ids"].values())
    assert body["replayed_count"] == 5
    assert body["skipped_count"] == 1
    assert body["considered"] == 6
    assert body["skipped_by_reason"] == {"inline_library": 1}
    assert body["summary"] == {
        "unchanged": 2,
        "match_gained": 1,
        "match_lost": 1,
        "winner_changed": 1,
    }
    assert body["error"] is None
    assert body["completed_at"] is not None

    # POST 响应自带第一页变更明细（默认排除 unchanged）
    assert body["count"] == 3
    by_jid = _by_type(body)
    assert set(by_jid) == {
        setup["ids"]["winner_changed"],
        setup["ids"]["match_gained"],
        setup["ids"]["match_lost"],
    }

    wc = by_jid[setup["ids"]["winner_changed"]]
    assert wc["change_type"] == "winner_changed"
    assert wc["old_status"] == "matched" and wc["new_status"] == "matched"
    assert wc["old_best_aid_id"] == "B" and wc["new_best_aid_id"] == "A"
    # 旧、新最佳候选证据均完整保存
    assert wc["old_best"]["aid_id"] == "B" and len(wc["old_best"]["events"]) == 8
    assert wc["new_best"]["aid_id"] == "A" and len(wc["new_best"]["events"]) == 8

    mg = by_jid[setup["ids"]["match_gained"]]
    assert mg["change_type"] == "match_gained"
    assert mg["old_status"] == "no_candidate" and mg["new_status"] == "matched"
    assert mg["old_best_aid_id"] is None and mg["old_best"] is None
    assert mg["new_best_aid_id"] == "Cp" and mg["new_best"]["matched"] == 8
    assert mg["old_reason"]  # 原无候选原因保留

    ml = by_jid[setup["ids"]["match_lost"]]
    assert ml["change_type"] == "match_lost"
    assert ml["old_best_aid_id"] == "B" and ml["new_best_aid_id"] is None
    assert ml["new_best"] is None and ml["new_reason"]

    # unchanged 记录可经 change_type 过滤取出
    r = client.get(
        f"/library/versions/{setup['draft_id']}/replay-reports/"
        f"{body['report_id']}/changes?change_type=unchanged"
    )
    assert r.status_code == 200
    unchanged = _by_type(r.json())
    assert set(unchanged) == {setup["ids"]["unchanged"], setup["ids"]["none_none"]}
    u = unchanged[setup["ids"]["unchanged"]]
    assert u["old_best_aid_id"] == u["new_best_aid_id"] == "C"
    nn = unchanged[setup["ids"]["none_none"]]
    assert nn["old_status"] == nn["new_status"] == "no_candidate"
    assert nn["old_best"] is None and nn["new_best"] is None


def test_replay_idempotent_same_key_returns_original(client, setup):
    url = f"/library/versions/{setup['draft_id']}/replay-reports"
    r1 = client.post(url, json={"idempotency_key": "dup-key"})
    assert r1.status_code == 201
    first = r1.json()

    r2 = client.post(url, json={"idempotency_key": "dup-key"})
    assert r2.status_code == 200
    again = r2.json()
    assert again["report_id"] == first["report_id"]
    assert again["watermark"] == first["watermark"]
    assert again["summary"] == first["summary"]
    assert again["completed_at"] == first["completed_at"]

    # 数据库中该键只有一份报告、一份明细
    import sqlite3
    db_path = client.app.state.db._path
    conn = sqlite3.connect(db_path)
    n_reports = conn.execute(
        "SELECT COUNT(*) FROM replay_reports WHERE idempotency_key = 'dup-key'"
    ).fetchone()[0]
    n_items = conn.execute(
        "SELECT COUNT(*) FROM replay_report_items WHERE report_id = ?",
        (first["report_id"],),
    ).fetchone()[0]
    conn.close()
    assert n_reports == 1
    assert n_items == 5


def test_idempotency_key_occupied_by_other_version_conflicts(client, setup):
    # 第二个草稿（同一基线）也想用同一个键 → 409
    other = client.post("/library/versions", json={
        "aids": BASE_AIDS, "base_version": setup["v1"],
    }).json()
    r1 = client.post(
        f"/library/versions/{setup['draft_id']}/replay-reports",
        json={"idempotency_key": "shared-key"},
    )
    assert r1.status_code == 201

    r2 = client.post(
        f"/library/versions/{other['version_id']}/replay-reports",
        json={"idempotency_key": "shared-key"},
    )
    assert r2.status_code == 409
    detail = r2.json()["detail"]
    assert detail["error"] == "replay_key_conflict"
    assert detail["existing_version_id"] == setup["draft_id"]
    assert detail["existing_report_id"] == r1.json()["report_id"]

    # 换键即可正常创建
    r3 = client.post(
        f"/library/versions/{other['version_id']}/replay-reports",
        json={"idempotency_key": "other-key"},
    )
    assert r3.status_code == 201


def test_watermark_freezed_and_no_judgments_written(client, setup):
    before = client.get("/judgments?limit=500").json()["count"]
    r = client.post(
        f"/library/versions/{setup['draft_id']}/replay-reports",
        json={"idempotency_key": "wm-key"},
    )
    report = r.json()
    assert report["watermark"] == before  # judgment_id 从 1 连续递增时

    # 回放不写入 judgments
    assert client.get("/judgments?limit=500").json()["count"] == before

    # 水位之后产生的新判定不会混入既有报告
    judge(client, synth(10.0, [0.0, 2.0]), mm=3, mf=3)
    page = client.get(
        f"/library/versions/{setup['draft_id']}/replay-reports/"
        f"{report['report_id']}/changes?limit=200"
    ).json()
    assert all(item["judgment_id"] <= report["watermark"] for item in page["changes"])
    assert client.get("/judgments?limit=500").json()["count"] == before + 1

    # 摘要接口与明细一致
    summary = client.get(
        f"/library/versions/{setup['draft_id']}/replay-reports/{report['report_id']}"
    ).json()
    assert summary["summary"] == report["summary"]
    assert "changes" not in summary


def test_changes_cursor_pagination(client, setup):
    report = client.post(
        f"/library/versions/{setup['draft_id']}/replay-reports",
        json={"idempotency_key": "page-key"},
    ).json()
    rid = report["report_id"]
    base = f"/library/versions/{setup['draft_id']}/replay-reports/{rid}/changes"

    # 取全部 unchanged 明细（2 条），分页大小 1
    seen = []
    cursor = 0
    pages = 0
    while True:
        page = client.get(
            f"{base}?change_type=unchanged&after_judgment_id={cursor}&limit=1"
        ).json()
        pages += 1
        seen.extend(item["judgment_id"] for item in page["changes"])
        if page["next_cursor"] is None:
            break
        cursor = page["next_cursor"]
    assert pages == 2
    assert seen == sorted(seen)
    assert set(seen) == {setup["ids"]["unchanged"], setup["ids"]["none_none"]}

    # 游标越过最后一条 → 空页、next_cursor 为 null
    tail = client.get(
        f"{base}?change_type=unchanged&after_judgment_id={max(seen)}&limit=1"
    ).json()
    assert tail["changes"] == [] and tail["next_cursor"] is None


def test_replay_version_errors(client, setup):
    url = f"/library/versions/{setup['draft_id']}/replay-reports"

    # 版本不存在 → 404
    r = client.post("/library/versions/999/replay-reports", json={"idempotency_key": "k"})
    assert r.status_code == 404

    # 空键 / 纯空白键 → 422
    assert client.post(url, json={"idempotency_key": ""}).status_code == 422
    assert client.post(url, json={"idempotency_key": "   "}).status_code == 422
    # 首尾空白被规范化后占用同一键
    assert client.post(url, json={"idempotency_key": "trim-key"}).status_code == 201
    assert client.post(url, json={"idempotency_key": "  trim-key  "}).status_code == 200

    # 非草稿（激活后）→ 400
    client.post(f"/library/versions/{setup['draft_id']}/activate")
    r = client.post(url, json={"idempotency_key": "after-activate"})
    assert r.status_code == 400

    # 报告不属于该版本 / 不存在 → 404
    other = client.post("/library/versions", json={
        "aids": BASE_AIDS, "base_version": setup["draft_id"],
    }).json()
    assert client.get(
        f"/library/versions/{other['version_id']}/replay-reports/1"
    ).status_code == 404
    assert client.get(
        f"/library/versions/{setup['draft_id']}/replay-reports/9999"
    ).status_code == 404
    assert client.get(
        f"/library/versions/{other['version_id']}/replay-reports/1/changes"
    ).status_code == 404

    # 非法 change_type → 422
    assert client.get(
        f"/library/versions/{setup['draft_id']}/replay-reports/1/changes"
        "?change_type=bogus"
    ).status_code == 422


def test_concurrent_same_key_single_report(client, setup):
    url = f"/library/versions/{setup['draft_id']}/replay-reports"
    n = 10
    barrier = threading.Barrier(n)
    results = []
    lock = threading.Lock()

    def submit():
        barrier.wait()
        r = client.post(url, json={"idempotency_key": "race-key"})
        with lock:
            results.append(r)

    threads = [threading.Thread(target=submit) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r.status_code in (200, 201) for r in results)
    report_ids = {r.json()["report_id"] for r in results}
    assert len(report_ids) == 1
    # 并发不会生成重复报告与重复明细
    import sqlite3
    conn = sqlite3.connect(client.app.state.db._path)
    assert conn.execute(
        "SELECT COUNT(*) FROM replay_reports WHERE idempotency_key = 'race-key'"
    ).fetchone()[0] == 1
    rid = next(iter(report_ids))
    assert conn.execute(
        "SELECT COUNT(*) FROM replay_report_items WHERE report_id = ?", (rid,)
    ).fetchone()[0] == 5
    conn.close()


def test_legacy_judgment_record_skipped_by_reason(client, setup, tmp_path):
    """旧版迁移产生的无请求判定记录按 legacy_record 原因跳过。"""
    import sqlite3
    import time as _time

    conn = sqlite3.connect(client.app.state.db._path)
    conn.execute(
        "INSERT INTO judgments (created_at, status, best_aid_id, matched, missed, "
        "false_count, drift, library_version, request, response) "
        "VALUES (?, 'matched', 'OLD', 1, 0, 0, 0.0, NULL, '{}', '{}')",
        (_time.time(),),
    )
    conn.commit()
    conn.close()

    body = client.post(
        f"/library/versions/{setup['draft_id']}/replay-reports",
        json={"idempotency_key": "legacy-key"},
    ).json()
    assert body["watermark"] >= 1
    assert body["skipped_by_reason"] == {"inline_library": 1, "legacy_record": 1}
    assert body["skipped_count"] == 2
    assert body["replayed_count"] == 5


def test_still_running_returns_503_with_retry_after(client, setup, monkeypatch):
    """同键回放仍在执行且等待超时：503 + Retry-After，且不生成第二份报告。"""
    import app.replay as replay_mod
    from app import main as main_module

    entered = threading.Event()
    release = threading.Event()

    # 阻塞首次回放的重算阶段：预约（running）之后、完成之前
    real = replay_mod._execute

    def slow_execute(db, report_id, *args, **kwargs):
        entered.set()
        release.wait(timeout=10)
        return real(db, report_id, *args, **kwargs)

    monkeypatch.setattr(replay_mod, "_execute", slow_execute)
    # 路由等待上限缩短到 0.3s，避免测试长时间忙等
    real_start = main_module.start_replay_report
    monkeypatch.setattr(
        main_module,
        "start_replay_report",
        lambda *a, **kw: real_start(*a, wait_timeout=0.3, poll_interval=0.02, **kw),
    )

    result = {}

    def first():
        result["status"] = client.post(
            f"/library/versions/{setup['draft_id']}/replay-reports",
            json={"idempotency_key": "slow-key"},
        ).status_code

    t = threading.Thread(target=first)
    t.start()
    assert entered.wait(timeout=5)

    # 另一请求同键：等待超时 → 503
    r = client.post(
        f"/library/versions/{setup['draft_id']}/replay-reports",
        json={"idempotency_key": "slow-key"},
    )
    assert r.status_code == 503
    assert r.headers.get("retry-after") == "1"
    assert r.json()["detail"]["error"] == "replay_still_running"

    release.set()
    t.join()
    assert result["status"] == 201

    # 超时等待没有创建第二份报告；首个请求完成后同键重试得到原报告（200）
    r = client.post(
        f"/library/versions/{setup['draft_id']}/replay-reports",
        json={"idempotency_key": "slow-key"},
    )
    assert r.status_code == 200 and r.json()["status"] == "completed"
    import sqlite3
    conn = sqlite3.connect(client.app.state.db._path)
    assert conn.execute(
        "SELECT COUNT(*) FROM replay_reports WHERE idempotency_key = 'slow-key'"
    ).fetchone()[0] == 1
    conn.close()


def test_failed_report_retryable_with_same_key(client, setup, monkeypatch):
    """首个请求计算失败：报告标记 failed；同键同版本重试重新回放成功。"""
    import pytest

    import app.replay as replay_mod

    calls = {"n": 0}
    real = replay_mod._execute

    def flaky(db, report_id, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom during replay")
        return real(db, report_id, *args, **kwargs)

    monkeypatch.setattr(replay_mod, "_execute", flaky)

    # 首个请求计算异常：异常向上抛出，报告落为 failed
    with pytest.raises(RuntimeError, match="boom during replay"):
        replay_mod.start_replay_report(
            client.app.state.db, setup["draft_id"], "flaky-key"
        )
    failed = client.get(
        f"/library/versions/{setup['draft_id']}/replay-reports/1"
    ).json()
    assert failed["status"] == "failed" and "boom during replay" in failed["error"]

    # 同键同版本重试（HTTP 201）：原地重置（report_id 不变）并完成
    r = client.post(
        f"/library/versions/{setup['draft_id']}/replay-reports",
        json={"idempotency_key": "flaky-key"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["report_id"] == 1 and body["status"] == "completed"
    assert body["summary"]["winner_changed"] == 1
    assert body["error"] is None

    # 失败明细未残留、键仍唯一
    import sqlite3
    conn = sqlite3.connect(client.app.state.db._path)
    assert conn.execute(
        "SELECT COUNT(*) FROM replay_reports WHERE idempotency_key = 'flaky-key'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM replay_report_items WHERE report_id = 1"
    ).fetchone()[0] == 5
    conn.close()
