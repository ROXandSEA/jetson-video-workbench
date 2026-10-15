"""Relative paths, labels and the production pipeline entry point."""
from pathlib import Path

def resolve(base, raw):
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()

def load_names(base, cfg, mode):
    if "names_file" in cfg:
        import yaml
        loaded = yaml.safe_load(resolve(base, cfg["names_file"]).read_text(encoding="utf-8"))
        names = loaded["names"]
    else:
        names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise RuntimeError("类别标签应是连续编号的非空字符串列表。")
    if mode == "coco" and len(names) != 80:
        raise RuntimeError("COCO 模式需要 80 类标签。")
    return names



def process_video(request, publish, cancel):
    from jetson_runtime import process_video as run
    return run(request, publish, cancel)
