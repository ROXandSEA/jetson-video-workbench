"""Run native dialogs in a separate process so tkinter stays on its main thread."""
import sys
import tkinter as tk
from tkinter import filedialog


def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial = {"initialdir": sys.argv[2]} if len(sys.argv) > 2 else {}
    try:
        if sys.argv[1] == "video":
            path = filedialog.askopenfilename(
                title="选择输入视频",
                filetypes=[("视频文件", "*.mp4 *.avi *.mkv *.mov *.m4v *.webm"), ("所有文件", "*")],
                **initial,
            )
        else:
            path = filedialog.askdirectory(title="选择结果导出目录", mustexist=True, **initial)
        sys.stdout.buffer.write((str(path) if path else "").encode("utf-8"))
        sys.stdout.buffer.flush()
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
