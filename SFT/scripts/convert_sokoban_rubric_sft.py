import json
from pathlib import Path
import re
# ── 固定的 user 消息内容 ──────────────────────────────────────────────────────
USER_CONTENT = "[Initial Observation]:\n{images}Analyze the current Sokoban board state and generate evaluation rubrics."
NEW_SYSTEM_CONTENT = "\n# Role\nYou are a Rubric Designer for Sokoban. Analyze the board image and generate evaluation rubrics.\n\n# Context\n## [System Rules] You are a Sokoban solver.\nGoal: Push all boxes onto targets.\nSymbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target\nRules: Push boxes only, avoid walls. Actions: Left, Down, Right, Up. Up to 3 actions per turn separated by comma.\nResponse format: observation, think, answer, prediction.\nIn observation and prediction, state each object position relative to player using exactly one vertical term (above, below, same row) and one horizontal term (left, right, same column).\nIn answer, output 1 to 3 valid actions separated by comma.\n\n## [Trajectory History] None (Initial Turn)\n## [Current Visual State] [Provided via attached Image]\n\nSTEP 1: Identify player, boxes, targets, and 2-cell obstacles. Classify moves as Illegal or Fatal with ALP ids. Compute freshness score where below 0.60 is fresh and above 0.85 is stale. Flag loops. Find best action and 2-step optimal sequence using distance delta. Trigger Pre-Target Alignment weight 15 if box is one push from target, or Push Side Positioning weight 6 if repositioning is needed.\nSTEP 2: Generate rubrics in fixed order. Rule 1 Spatial Grounding weight 14. Rule 2 Action Legality weight 12 with veto. Rule 3 Logic Freshness weight 5. Rule 4 Strategic Progress weight 5 or 8. Rule 5 Push Side Positioning weight 6 if triggered. Rule 6 Pre-Target Alignment weight 15 if triggered.\n\n# OUTPUT FORMAT\nReturn ONLY the JSON in a json block.\n\n{\n  \"turn_id\": \"turn_000001\",\n  \"turn_analysis\": {\n    \"player_pos\": [r, c],\n    \"box_positions\": [[r, c]],\n    \"target_positions\": [[r, c]],\n    \"legal_actions\": [\"Dir\"],\n    \"between_check\": \"YES/NO\",\n    \"push_side_pos\": [r, c],\n    \"best_action\": \"Dir\",\n    \"optimal_seq\": [\"Dir\", \"Dir\"],\n    \"freshness_score\": 0.0,\n    \"summary\": \"one sentence summary\"\n  },\n  \"rubrics\": [\n    {\n      \"rule_id\": \"SPATIAL_GROUNDING\",\n      \"description\": \"Part a 4pts: observation states correct position for all objects, score 0 if wrong or missing. Part b 6pts: if BETWEEN is YES then think must contain detour keywords, score 0 if absent. Part c 4pts: if obstacle within 2 cells then think must name it, score 0 if unmentioned.\",\n      \"weight\": 14\n    },\n    {\n      \"rule_id\": \"ACTION_LEGALITY\",\n      \"description\": \"List each forbidden direction with ALP id and reason, score 0 if answer uses it. Fail collapses all other scores to 0.\",\n      \"weight\": 12\n    },\n    {\n      \"rule_id\": \"LOGIC_FRESHNESS\",\n      \"description\": \"PASS if freshness below 0.60, PARTIAL if 0.60 to 0.85, FAIL if above 0.85. FAIL if 3 or more turns share over 85 percent overlap. PASS on loop only if think proposes new plan.\",\n      \"weight\": 5\n    },\n    {\n      \"rule_id\": \"STRATEGIC_PROGRESS\",\n      \"description\": \"PASS if delta positive. PASS if delta zero and reposition justified. PASS if delta negative only when detour justified. First action must match optimal sequence. If plan depth needed then think must name at least 2 steps.\",\n      \"weight\": 5\n    }\n  ]\n}\n"

def _try_parse_json(text: str):
    """
    Try multiple strategies to extract a JSON object from a string.
    Strategy 1: ```json ... ``` fenced block
    Strategy 2: First { ... } span
    Strategy 3: Direct json.loads
    """
    if not text:
        return None

    # Strategy 1 – fenced block
    fence_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2 – first balanced brace span
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start: i + 1])
                    except json.JSONDecodeError:
                        break

    # Strategy 3 – raw parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def convert_item(raw: dict, line_num: int) -> tuple[dict | None, list[str]]:
    """
    将单条原始数据转换为 LLaMA-Factory SFT 标准格式

    Returns:
        (converted_item, errors)
        converted_item: 转换成功则返回dict，否则None
        errors: 错误信息列表，空列表表示无错误
    """
    errors = []

    # ── 字段存在性检查 ────────────────────────────────────────────────────────
    required_fields = ["rubric_prompt", "rubric", "images"]
    for field in required_fields:
        if field not in raw:
            errors.append(f"缺少必要字段: '{field}'")

    if errors:
        return None, errors

    # ── 字段类型检查 ──────────────────────────────────────────────────────────
    if not raw["rubric_prompt"]:
        errors.append("'rubric_prompt' 为空")

    if not isinstance(raw["rubric"], (str, dict)) or not raw["rubric"]:
        errors.append("'rubric' 为空或类型不受支持")

    if not isinstance(raw["images"], list) or len(raw["images"]) == 0:
        errors.append("'images' 为空或非列表")

    if errors:
        return None, errors

    if isinstance(raw["rubric"], dict):
        parsed = raw["rubric"]
    else:
        parsed = _try_parse_json(raw["rubric"])
    response = json.dumps(parsed, ensure_ascii=False) if parsed is not None else raw["rubric"]
    # ── 构建 messages ─────────────────────────────────────────────────────────
    messages = [
        {
            "role": "system",
            "content": NEW_SYSTEM_CONTENT
        },
        {
            "role": "user",
            "content": USER_CONTENT.format(images="<image>\n" * len(raw["images"]))
        },
        {
            "role": "assistant",
            "content": response
        }
    ]

    # ── 组装输出条目 ──────────────────────────────────────────────────────────
    converted = {
        "messages": messages,
        "images": raw["images"]          # 绝对路径原样保留
    }

    # # 保留其他非核心字段（如 turn_id / metadata 等）
    # skip_keys = {"rubric_prompt", "rubric", "images"}
    # for key, value in raw.items():
    #     if key not in skip_keys:
    #         converted[key] = value

    return converted, []


def validate_converted_item(item: dict) -> list[str]:
    """验证转换后条目的完整性"""
    errors = []

    messages = item.get("messages", [])
    roles = [m.get("role") for m in messages]

    # 检查角色顺序：system -> user -> assistant
    expected_roles = ["system", "user", "assistant"]
    if roles != expected_roles:
        errors.append(f"消息角色顺序异常: 期望 {expected_roles}，实际 {roles}")

    # 检查每条消息 content 非空
    for idx, msg in enumerate(messages):
        if not msg.get("content", "").strip():
            errors.append(f"第 {idx} 条消息 content 为空 (role={msg.get('role')})")

    # 检查 user 消息包含 <image> 占位符
    user_msgs = [m for m in messages if m.get("role") == "user"]
    for msg in user_msgs:
        if "<image>" not in msg.get("content", ""):
            errors.append("user 消息中缺少 <image> 占位符")

    # 检查图像占位符数量与 images 数量匹配
    n_placeholders = sum(
        m.get("content", "").count("<image>") for m in messages
    )
    n_images = len(item.get("images", []))
    if n_placeholders != n_images:
        errors.append(
            f"<image> 占位符数量 ({n_placeholders}) 与图像数量 ({n_images}) 不匹配"
        )

    # 检查 assistant 内容包含合法 JSON（rubric 应为 json 块）
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    for msg in assistant_msgs:
        content = msg.get("content", "")
        if "```json" not in content and not content.strip().startswith("{"):
            errors.append("assistant 内容中未检测到 JSON 格式的 rubric")

    return errors


def convert_jsonl(
    input_path: str,
    output_path: str,
    skip_invalid: bool = False,
    verbose: bool = True,
):
    """
    将原始 JSONL 转换为 LLaMA-Factory rubric SFT 标准格式

    Args:
        input_path  : 输入 JSONL 文件路径
        output_path : 输出 JSONL 文件路径
        skip_invalid: True=跳过验证失败条目；False=警告但仍写入
        verbose     : 是否打印处理进度
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total"   : 0,
        "success" : 0,
        "skipped" : 0,
        "warnings": 0,
    }

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            # ── 解析原始行 ────────────────────────────────────────────────
            try:
                raw_item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Line {line_num}] ❌ JSON 解析失败: {e}")
                stats["skipped"] += 1
                continue

            # ── 转换 ──────────────────────────────────────────────────────
            converted, convert_errors = convert_item(raw_item, line_num)

            if convert_errors:
                stats["warnings"] += 1
                if verbose:
                    print(f"[Line {line_num}] ⚠️  转换错误:")
                    for err in convert_errors:
                        print(f"  • {err}")
                stats["skipped"] += 1
                if verbose:
                    print(f"  → 已跳过")
                continue

            # ── 验证 ──────────────────────────────────────────────────────
            validate_errors = validate_converted_item(converted)
            if validate_errors:
                stats["warnings"] += 1
                if verbose:
                    print(f"[Line {line_num}] ⚠️  验证警告:")
                    for err in validate_errors:
                        print(f"  • {err}")
                if skip_invalid:
                    stats["skipped"] += 1
                    if verbose:
                        print(f"  → 已跳过")
                    continue

            # ── 写入 ──────────────────────────────────────────────────────
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            stats["success"] += 1

            if verbose and stats["success"] % 500 == 0:
                print(f"  ✅ 已处理 {stats['success']} 条...")

    # ── 统计摘要 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("📊 转换统计:")
    print(f"  总计输入 : {stats['total']}  条")
    print(f"  成功转换 : {stats['success']} 条")
    print(f"  跳过数量 : {stats['skipped']} 条")
    print(f"  警告数量 : {stats['warnings']} 条")
    print(f"  输出文件 : {output_path}")
    print("=" * 50)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="将 Sokoban rubric JSONL 转换为 LLaMA-Factory messages 格式"
    )
    parser.add_argument("--input",        "-i", required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output",       "-o", required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--skip_invalid", action="store_true",  help="跳过验证失败的条目")
    parser.add_argument("--quiet",  "-q", action="store_true",  help="减少输出信息")

    args = parser.parse_args()


    convert_jsonl(
        input_path   = args.input,
        output_path  = args.output,
        skip_invalid = args.skip_invalid,
        verbose      = not args.quiet,
    )
