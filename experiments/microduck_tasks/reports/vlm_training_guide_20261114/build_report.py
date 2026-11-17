"""Compile the Chinese guide from frozen snapshots and chapter source files."""

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PDF_NAME = "Microduck_VLM_从预训练到微调完整指南.pdf"


def tex(value):
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(c, c) for c in str(value))


def references():
    sources = []
    for name in ["sources_qwen.json", "sources_methods.json"]:
        sources.extend(json.loads((ROOT / name).read_text()))
    source_map = {}
    for entry in sources:
        if entry["key"] in source_map and source_map[entry["key"]]["url"] != entry["url"]:
            raise ValueError(f"Conflicting source key {entry['key']}")
        source_map[entry["key"]] = entry
    chapter_text = "\n".join(f.read_text() for f in ROOT.glob("*.tex") if f.name != "references.tex")
    cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", chapter_text) for key in group.split(",")}
    if missing := cited - source_map.keys():
        raise ValueError(f"Missing bibliography sources: {missing}")
    lines = [
        r"\chapter*{来源与证据}",
        r"\addcontentsline{toc}{chapter}{来源与证据}",
        "公开模型事实优先引用官方模型卡、配置、代码和论文。模型系列公开报告不能替代特定检查点完整数据配方。"
        "本地参数、样本数量、失败处理与运行状态取自本项目源码和固定快照；建议方案与教学示例已在正文标明。"
        "网页访问日期为2026年11月14日。",
        r"\begingroup\renewcommand{\bibname}{公开一手资料}",
        r"\begin{thebibliography}{99}",
    ]
    for key, entry in source_map.items():
        lines.extend([
            rf"\bibitem{{{key}}} {tex(entry['title'])}.",
            rf"\url{{{entry['url']}}}",
        ])
    lines.extend([
        r"\end{thebibliography}\endgroup",
        r"\section*{本地证据}",
        r"\addcontentsline{toc}{section}{本地证据}",
        "项目根目录：",
        r"\begin{Verbatim}",
        "/home/luckysir/Strands_Robots",
        r"\end{Verbatim}",
        "以下证据以项目根目录为起点：",
        r"\begin{itemize}",
    ])
    for path, detail in [
        ("experiments/microduck_tasks/outputs/training_collection_audit_20261114.json", "三组采集计划、失败案例与原始资产审计"),
        ("experiments/microduck_tasks/outputs/training_dataset_final_20261114/manifest.json", "正式数据划分、样本数量、标签分布与源记录"),
        ("experiments/microduck_tasks/outputs/training_audit_20261114/final_dataset.json", "数据泄漏、图片一致性与导出结构审计"),
        ("experiments/microduck_tasks/outputs/qlora_20261114/training_config.json", "模型、依赖、超参数、数据与权重指纹"),
        ("experiments/microduck_tasks/outputs/qlora_20261114/metrics.jsonl", "真实更新损失、学习率、显存与定期验证"),
        ("experiments/microduck_tasks/outputs/qlora_20261114/startup_verification.json", "正式首个检查点参数与状态核验"),
        ("experiments/microduck_tasks/outputs/resource_verification_20261114/resource_verification.json", "资源试跑和恢复验证；与正式任务成绩分开"),
    ]:
        lines.append(rf"\item {tex(detail)}：\path{{{path}}}")
    lines.extend([
        r"\end{itemize}",
        "报告目录中的 \\texttt{sources\\_local\\_data.json} 列出数据与执行代码的具体行号及SHA256；"
        "\\texttt{sources\\_methods.json} 列出训练方法来源；"
        "\\texttt{snapshot\\_*.json} 与 \\texttt{snapshot\\_metrics.jsonl} 是生成本书时使用的固定状态。"
        "原始训练日志在继续增长，报告不会自动刷新。",
    ])
    (ROOT / "references.tex").write_text("\n".join(lines) + "\n")
    (ROOT / "sources_public.json").write_text(json.dumps(list(source_map.values()), ensure_ascii=False, indent=2))


def main():
    references()
    subprocess.run(["python3", str(ROOT / "build_assets.py")], cwd=ROOT, check=True)
    for pass_number in range(1, 4):
        with (ROOT / f"build_pass_{pass_number}.log").open("w") as log:
            subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
                 "-jobname=microduck_vlm_guide", "guide.tex"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
    source = ROOT / "microduck_vlm_guide.pdf"
    shutil.copy2(source, ROOT / PDF_NAME)
    manifest = {
        "pdf": PDF_NAME,
        "pdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "snapshot": json.loads((ROOT / "snapshot_time.json").read_text()),
        "chapters": [
            {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(ROOT.glob("*.tex"))
        ],
        "training_process_modified": False,
        "source_of_charts": "snapshot_metrics.jsonl and snapshot_baseline_evaluation.json",
    }
    (ROOT / "report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(ROOT / PDF_NAME)


if __name__ == "__main__":
    main()
