#!/usr/bin/env python3
"""草稿特征库上线前评估：创建草稿并对历史判定发起回放报告。

脚本演示完整流程（仅依赖标准库，默认对接 http://localhost:8000）：

1. 读取草稿特征库 JSON（结构同 /library/import 的 aids，支持直接传含
   ``aids`` 键的文件，如 samples/library.json）；
2. 以当前激活版本为 base_version 创建草稿（不激活）；
3. 用客户端生成的幂等键调用
   ``POST /library/versions/{id}/replay-reports`` 发起回放——服务端冻结当前
   最大 judgment_id，只重放水位内服务端特征库判定，内联库判定按原因跳过；
4. 打印报告摘要，并按 judgment_id 游标分页拉取全部变更明细。

用法：
    python3 samples/evaluate_draft.py samples/library.json
    python3 samples/evaluate_draft.py draft.json --base-url http://localhost:8000 \
        --idempotency-key draft-2026-09-30

重复执行（同一 --idempotency-key）安全：服务端返回原报告，不会重复计算。
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid


def request(method, url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        try:
            detail = json.loads(detail)
        except json.JSONDecodeError:
            pass
        return exc.code, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draft", help="草稿特征库 JSON 文件（aid 列表或含 aids 键）")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--idempotency-key",
        default=None,
        help="回放幂等键（缺省随机生成；重试时请显式传入相同的键）",
    )
    parser.add_argument("--page-size", type=int, default=50)
    args = parser.parse_args()

    with open(args.draft, encoding="utf-8") as fh:
        payload = json.load(fh)
    aids = payload["aids"] if isinstance(payload, dict) and "aids" in payload else payload

    status, active = request("GET", f"{args.base_url}/library")
    if status != 200:
        print(f"查询当前激活版本失败: {status} {active}", file=sys.stderr)
        return 1
    base_version = active["library_version"]
    if base_version is None:
        print("服务端尚无特征库版本，请先 POST /library/import", file=sys.stderr)
        return 1

    status, draft = request(
        "POST",
        f"{args.base_url}/library/versions",
        {"base_version": base_version, "aids": aids},
    )
    if status != 201:
        print(f"创建草稿失败: {status} {json.dumps(draft, ensure_ascii=False)}",
              file=sys.stderr)
        return 1
    version_id = draft["version_id"]
    print(f"已创建草稿版本 {version_id}（base_version={base_version}）")
    print("相对当前版本的增删改：" + json.dumps(draft["diff"], ensure_ascii=False))

    key = args.idempotency_key or f"eval-{uuid.uuid4()}"
    status, report = request(
        "POST",
        f"{args.base_url}/library/versions/{version_id}/replay-reports",
        {"idempotency_key": key},
    )
    if status not in (200, 201):
        print(f"创建回放报告失败: {status} "
              f"{json.dumps(report, ensure_ascii=False)}", file=sys.stderr)
        return 1
    print(f"回放报告 {report['report_id']}（HTTP {status}，"
          f"{'新创建' if status == 201 else '幂等返回原报告'}，键 {key}）")
    print(f"冻结水位 watermark={report['watermark']}")
    print(f"参与评估 {report['considered']} 条：重放 {report['replayed_count']} 条，"
          f"跳过 {report['skipped_count']} 条 "
          f"{json.dumps(report['skipped_by_reason'], ensure_ascii=False)}")
    print("变更摘要：" + json.dumps(report["summary"], ensure_ascii=False))

    # 按原 judgment_id 游标分页拉取全部发生变化的明细
    cursor = 0
    changed = []
    while True:
        status, page = request(
            "GET",
            (f"{args.base_url}/library/versions/{version_id}/replay-reports/"
             f"{report['report_id']}/changes"
             f"?after_judgment_id={cursor}&limit={args.page_size}"),
        )
        if status != 200:
            print(f"拉取变更明细失败: {status} {page}", file=sys.stderr)
            return 1
        changed.extend(page["changes"])
        if page["next_cursor"] is None:
            break
        cursor = page["next_cursor"]

    for item in changed:
        old = item["old_best_aid_id"] or "—"
        new = item["new_best_aid_id"] or "—"
        print(f"  judgment {item['judgment_id']:>4}  "
              f"{item['change_type']:<15} {old} → {new}")
    print(f"共 {len(changed)} 条判定发生变化；评估通过后再激活草稿："
          f"POST /library/versions/{version_id}/activate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
