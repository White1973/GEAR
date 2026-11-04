import json
import re
import os
import glob
def find_jsonl_files_glob(root_dir):
    """
    使用 glob 递归查找指定路径下所有 jsonl 文件
    """
    pattern = os.path.join(root_dir, "**", "*.jsonl")

    jsonl_files = glob.glob(pattern, recursive=True)
    return jsonl_files

def extract_im_blocks(text):
    """
    提取所有 <|im_start|>role\n...<|im_end|> 之间的内容
    返回列表，每个元素为 (role, content) 元组
    """
    pattern = r'<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches


# 匹配新的输出格式：observation -> reasoning -> prediction -> answer
# 标签之间可能有任意文本（如 "..." 等省略号），用 .*? 连接
_ANSWER_PATTERN = re.compile(
    r'(<observation>.*?</observation>.*?<reasoning>.*?</reasoning>.*?<prediction>.*?</prediction>.*?<answer>.*?</answer>)',
    re.DOTALL,
)
# 兼容旧格式：observation -> reasoning -> answer -> prediction
_ANSWER_PATTERN_LEGACY = re.compile(
    r'(<observation>.*?</observation>.*?<reasoning>.*?</reasoning>.*?<answer>.*?</answer>.*?<prediction>.*?</prediction>)',
    re.DOTALL,
)


def _extract_tag_content(text, tag):
    """提取 <tag>...</tag> 中的内容，返回首个匹配或 None。"""
    m = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return m.group(1) if m else None


def extract_last_assistant_content(output_text):
    """
    从 output 中提取最后一个 assistant 回复的结构化内容。
    output 格式：<|im_start|>user\n...<|im_end|><|im_start|>assistant\n...<|im_end|>

    模型输出中标签之间可能混入噪音文本（如 "..."），因此分别提取四个标签
    的纯净内容后按标准顺序重新拼接，确保 SFT 数据干净。

    返回清理后的结构化文本，若无匹配返回 None。
    """
    assistant_blocks = re.findall(
        r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', output_text, flags=re.DOTALL
    )
    if not assistant_blocks:
        return None

    last_block = assistant_blocks[-1]

    # 提取四个标签的内容
    obs = _extract_tag_content(last_block, "observation")
    reas = _extract_tag_content(last_block, "reasoning")
    pred = _extract_tag_content(last_block, "prediction")
    ans = _extract_tag_content(last_block, "answer")

    # 新格式：observation + reasoning + prediction + answer
    if obs is not None and reas is not None and pred is not None and ans is not None:
        return (
            f"<observation>{obs}</observation>\n"
            f"<reasoning>{reas}</reasoning>\n"
            f"<prediction>{pred}</prediction>\n"
            f"<answer>{ans}</answer>"
        )

    # 旧格式：observation + reasoning + answer（无 prediction 或 prediction 可选）
    if obs is not None and reas is not None and ans is not None:
        if pred is not None:
            return (
                f"<observation>{obs}</observation>\n"
                f"<reasoning>{reas}</reasoning>\n"
                f"<prediction>{pred}</prediction>\n"
                f"<answer>{ans}</answer>"
            )
        return (
            f"<observation>{obs}</observation>\n"
            f"<reasoning>{reas}</reasoning>\n"
            f"<answer>{ans}</answer>"
        )

    return None


def process_jsonl_file(input_file, output_file, min_dino=0.90, min_dreamsim=0.70):
    """
    读取jsonl文件，筛选高质量数据（dino_score >= 0.90, dreamsim_score >= 0.70），
    提取input中的system/user prompt和output中最后一个assistant的结构化回复，
    组装为SFT训练格式。
    """
    processed_count = 0
    skipped_format = 0
    skipped_image = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'a', encoding='utf-8') as outfile:

            for line_num, line in enumerate(infile):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    if data.get("dino_score", 0) < min_dino or data.get("dreamsim_score", 0) < min_dreamsim:
                        continue

                    input_text = data.get("input", "")
                    output_text = data.get("output", "")
                    step = data.get("step", "")

                    # 解析 input 中的 system 和 user 块
                    blocks = extract_im_blocks(input_text)
                    if len(blocks) < 2:
                        print(f"警告: 第 {line_num} 行 input 块不足2个，已跳过。")
                        continue

                    system_text = blocks[0][1].strip()
                    user_content = blocks[1][1].strip()
                    human_text = "<image>" + user_content

                    # 从 output 提取最后一个 assistant 的结构化回复
                    extracted_content = extract_last_assistant_content(output_text)

                    image_path = os.path.abspath(os.path.join(
                        os.path.dirname(input_file),
                        f"image_{step}", f"images_{line_num}", "0.png"
                    ))

                    if not os.path.exists(image_path):
                        skipped_image += 1
                        continue

                    if extracted_content:
                        result = {
                            "messages": [
                                {"role": "system", "content": system_text},
                                {"role": "user", "content": human_text},
                                {"role": "assistant", "content": extracted_content}
                            ],
                            "images": [image_path]
                        }
                        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                        processed_count += 1
                    else:
                        skipped_format += 1

                except json.JSONDecodeError:
                    print(f"错误: 第 {line_num} 行 JSON 格式无效，已跳过。")
                except Exception as e:
                    print(f"错误: 处理第 {line_num} 行时发生异常: {e}")
                    return None

        print(f"处理完成！共处理 {processed_count} 条数据。"
              f" 跳过(格式不匹配): {skipped_format}, 跳过(图片缺失): {skipped_image}")
        print(f"结果已保存至: {output_file}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")

# 运行函数
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract SFT data from rollout jsonl files.")
    parser.add_argument("--root_dir", type=str, default="svg_ppo_qwen25vl3b_s3/rollout_data",
                        help="Directory containing rollout jsonl files")
    parser.add_argument("--output_file", type=str, default="processed.jsonl",
                        help="Output jsonl file path")
    parser.add_argument("--min_step", type=int, default=100, help="Minimum step number to include")
    parser.add_argument("--max_step", type=int, default=1300, help="Maximum step number to include")
    parser.add_argument("--min_dino", type=float, default=0.90, help="Minimum dino_score threshold")
    parser.add_argument("--min_dreamsim", type=float, default=0.70, help="Minimum dreamsim_score threshold")
    args = parser.parse_args()

    # Remove old output file to avoid appending to stale data
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    jsonl_files = find_jsonl_files_glob(args.root_dir)

    total_files = 0
    for file_path in jsonl_files:
        count = int(file_path.split("/")[-1].split(".")[0])
        if args.min_step <= count <= args.max_step:
            print(f"processing .... {file_path} step={count}")
            process_jsonl_file(
                file_path,
                args.output_file,
                min_dino=args.min_dino,
                min_dreamsim=args.min_dreamsim,
            )
            total_files += 1

    print(f"Total files processed: {total_files}")
