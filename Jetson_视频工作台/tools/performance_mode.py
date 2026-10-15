#!/usr/bin/env python3
"""Opt-in desktop action; administrator credentials stay in the system dialog."""
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox

def main():
    root=tk.Tk();root.title('Jetson 高性能模式');root.geometry('480x210')
    label=tk.Label(root,text='将 CPU、GPU 和内存频率锁至当前功耗模式的上限。\n当前开机有效，重启恢复系统默认。\n管理员密码由系统授权窗口输入。',font=('sans',12),wraplength=440)
    label.pack(padx=20,pady=20)
    def enable():
        button.config(state='disabled');label.config(text='等待系统管理员授权…')
        def work():
            try:
                result=subprocess.run(['pkexec','/usr/bin/jetson_clocks'],capture_output=True,text=True)
                message='已启用当前功耗模式下的最高频率。\n现在可以双击“启动网页”开始使用。' if result.returncode==0 else '未启用：'+(result.stderr.strip() or '授权已取消')
            except OSError as exc:message=str(exc)
            root.after(0,lambda: (label.config(text=message),button.config(state='normal')))
        threading.Thread(target=work,daemon=True).start()
    button=tk.Button(root,text='启用高性能模式',command=enable,font=('sans',12));button.pack(pady=8)
    root.mainloop()
if __name__=='__main__':main()
