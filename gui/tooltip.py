"""
Lớp tooltip cho giao diện Tkinter
"""

import tkinter as tk

class ToolTip:
    """Tool Tip widget cho Tkinter.
    Hiển thị mô tả text khi đưa chuột vào widget.
    """
    def __init__(self, widget, text=None, delay=500, wraplength=250):
        """Khởi tạo Tooltip

        Tham số:
        widget -- widget để gắn tooltip
        text -- text hiển thị trong tooltip
        delay -- thời gian trì hoãn hiển thị (ms)
        wraplength -- độ dài dòng tối đa
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)

    def enter(self, event=None):
        """Bắt đầu quá trình hiển thị tooltip."""
        self.schedule()

    def leave(self, event=None):
        """Hủy bỏ hiển thị tooltip."""
        self.unschedule()
        self.hidetip()

    def schedule(self):
        """Lên lịch hiển thị tooltip sau delay ms."""
        self.unschedule()
        self.id = self.widget.after(self.delay, self.showtip)

    def unschedule(self):
        """Hủy bỏ lịch hiển thị tooltip."""
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self):
        """Hiển thị tooltip."""
        if self.tipwindow or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 25
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        try:
            # For macOS
            tw.tk.call("::tk::unsupported::MacWindowStyle",
                       "style", tw._w,
                       "help", "noActivates")
        except tk.TclError:
            pass
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                       background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                       wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        """Ẩn tooltip."""
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy() 