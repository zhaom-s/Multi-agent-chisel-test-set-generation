# LangGraph_agents/verifier.py

import os
import re
import shutil
import requests
import json


def check_file_via_http(api_url, file_path: str):
    """
    带有超时保护的单文件检测逻辑
    """
    params = {"path": file_path}
    try:
        resp = requests.get(api_url, params=params, stream=True, timeout=(5, 60))

        if resp.status_code != 200:
            return False, f"HTTP FAIL | status={resp.status_code}"

        final_json = None
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line: continue
            line = raw_line.strip()
            if "data:complete:true" in line:
                brace_idx = line.find("{")
                if brace_idx != -1:
                    json_str = line[brace_idx:]
                    try:
                        final_json = json.loads(json_str)
                        break
                    except: continue

        if final_json is None:
            return False, "未捕获 data:complete:true"

        files = final_json.get("files", [])
        if not files: return False, "complete 中 files 为空"

        info = files[0]
        syntax = info.get("syntax")
        mlir = str(info.get("mlir")).lower()
        top_modules = info.get("topModules", [])

        if syntax is True and mlir in ("true", "none") and len(top_modules) > 0:
            return True, info
        else:
            return False, info

    except requests.exceptions.Timeout:
        error_msg = "TIMEOUT | Scala compiler hung"
        print(f"  ⏰ [Timeout] {os.path.basename(file_path)} 耗时过长，已跳过")
        return False, {"error": "Timeout", "message": error_msg}
    except Exception as e:
        error_msg = f"RUNTIME_ERROR | {str(e)}"
        print(f"  💥 [Error] {os.path.basename(file_path)} 验证中断: {e}")
        return False, {"error": "Exception", "message": error_msg}


def _classify_error(detail) -> str:
    """
    从 API 返回的 detail 中推断错误类型，供 Planner 感知使用。
    返回: "syntax" | "mlir" | "empty" | "timeout" | "unknown"
    """
    if isinstance(detail, str):
        if "TIMEOUT" in detail: return "timeout"
        if "HTTP FAIL" in detail: return "unknown"
        if "files 为空" in detail or "complete" in detail: return "empty"
        return "unknown"

    if isinstance(detail, dict):
        if detail.get("error") == "Timeout": return "timeout"
        if detail.get("error") == "Exception": return "unknown"
        syntax = detail.get("syntax")
        mlir = str(detail.get("mlir", "")).lower()
        top_modules = detail.get("topModules", [])
        if syntax is not True:
            return "syntax"
        if mlir not in ("true", "none"):
            return "mlir"
        if not top_modules:
            return "mlir"

    return "unknown"


def run_verifier(scala_t_dir, scala_f_dir, api_url, iter_dir, attempt_count: dict = None,
                 pass_subdir: str = "verified_pass",
                 fail_subdir: str = "verified_fail",
                 log_name: str = "verify_report.log"):
    """
    验证所有 Scala 文件，返回：
      - failed_ids      : 失败的规则 ID 列表（兼容旧接口）
      - error_context   : {rule_id: {error_type, message, file_path, code_type, attempt}}
                          供 Planner Agent 感知失败根因使用
    pass_subdir / fail_subdir / log_name 可自定义，供二次验证使用。
    """
    pass_dir = os.path.join(iter_dir, pass_subdir)
    fail_dir = os.path.join(iter_dir, fail_subdir)
    log_file = os.path.join(iter_dir, log_name)
    os.makedirs(pass_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    test_files = []
    for d in [scala_t_dir, scala_f_dir]:
        if os.path.exists(d):
            test_files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".scala")]

    failed_ids = set()
    error_context = {}

    print(f"🧐 [Verifier] 开始验证 {len(test_files)} 个 Scala 文件...")

    with open(log_file, "w", encoding="utf-8") as log:
        log.write("=== Iteration Verification Report ===\n\n")

        for file_path in test_files:
            file_name = os.path.basename(file_path)
            try:
                rule_id = int(re.search(r"rule_(\d+)", file_name).group(1))
            except Exception:
                rule_id = 0

            code_type = "T" if "_T.scala" in file_name else "F"

            ok, detail = check_file_via_http(api_url, file_path)

            status_str = "✅ PASS" if ok else "❌ FAIL"
            target_path = os.path.join(pass_dir if ok else fail_dir, file_name)
            shutil.copy(file_path, target_path)

            if not ok:
                failed_ids.add(rule_id)
                error_type = _classify_error(detail)
                attempt = (attempt_count or {}).get(rule_id, 0)
                # 提取真正有用的 errorMsg，而非整个 JSON
                if isinstance(detail, dict):
                    error_msg = detail.get("errorMsg") or detail.get("message") or json.dumps(detail)
                else:
                    error_msg = str(detail)
                # 对同一 rule_id，以 MLIR 失败优先（更严重）
                existing = error_context.get(rule_id, {})
                if existing.get("error_type") != "mlir":
                    error_context[rule_id] = {
                        "error_type": error_type,
                        "message":    error_msg,
                        "file_path":  file_path,
                        "code_type":  code_type,
                        "attempt":    attempt + 1,
                    }

            log_line = f"{status_str} | {file_name}"
            print(f"  {log_line}")
            log.write(log_line + "\n")
            if not ok:
                log.write(f"    Reason: {json.dumps(detail)}\n")

    return sorted(list(failed_ids)), error_context