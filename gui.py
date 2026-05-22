# gui.py
"""
University Student Information Chatbot — Desktop GUI (Tkinter)

Professional chat interface for student support queries.
Delegates all NLP to ChatbotEngine (app.py); this module handles presentation only.
"""

import os
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from datetime import datetime

# ---------------------------------------------------------------------------
# Backend import — GUI stays usable if model files are missing
# ---------------------------------------------------------------------------
try:
    from app import ChatbotEngine
    _BACKEND_OK = True
    _BACKEND_ERROR = ""
except Exception as _e:
    _BACKEND_OK = False
    _BACKEND_ERROR = str(_e)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
APP_TITLE = "Student Information Chatbot"
WIN_WIDTH = 980
WIN_HEIGHT = 720
MIN_WIDTH = 680
MIN_HEIGHT = 520

# Calm university palette
C_BG = "#F0F4F3"
C_CHAT_BG = "#FFFFFF"
C_USER_BG = "#D6EAF8"
C_USER_FG = "#1A5276"
C_BOT_BG = "#ECEFF1"
C_BOT_FG = "#263238"
C_SYS_BG = "#E8F5E9"
C_SYS_FG = "#2E7D32"
C_DEBUG_FG = "#78909C"
C_INPUT_BG = "#FFFFFF"
C_INPUT_DISABLED = "#ECEFF1"
C_SEND_BG = "#1B5E20"
C_SEND_DISABLED = "#A5D6A7"
C_SEND_FG = "#FFFFFF"
C_STATUS_BG = "#1B365D"
C_STATUS_FG = "#CFD8DC"
C_STATUS_ACCENT = "#81C784"
C_HEADER_BG = "#1B365D"
C_HEADER_FG = "#FFFFFF"
C_HEADER_SUB = "#B0BEC5"
C_TYPING_FG = "#607D8B"
C_BTN_BG = "#FFFFFF"
C_BTN_FG = "#1B365D"
C_BTN_BORDER = "#B0BEC5"
C_TOOLBAR_BG = "#E8EEF2"
C_BORDER = "#CFD8DC"

FONT_HEADER = ("Segoe UI", 15, "bold")
FONT_SUBHEADER = ("Segoe UI", 10)
FONT_CHAT = ("Segoe UI", 11)
FONT_CHAT_TS = ("Segoe UI", 8)
FONT_INPUT = ("Segoe UI", 11)
FONT_SEND = ("Segoe UI", 10, "bold")
FONT_STATUS = ("Segoe UI", 9)
FONT_DEBUG = ("Segoe UI", 9, "italic")
FONT_BTN = ("Segoe UI", 9)
FONT_LABEL = ("Segoe UI", 10, "bold")

SESSION_LOG = os.path.join("logs", "chat_session.txt")

QUICK_ACTIONS = [
    ("Registration", "How do I register?"),
    ("Fees", "How much are the fees?"),
    ("Exams", "When is the exam schedule?"),
    ("Courses", "What courses are available?"),
    ("Locations", "Where is the library?"),
    ("Scholarships", "How do I apply for scholarship?"),
    ("Student Services", "Where is the health center?"),
    ("Contacts", "How can I contact the registrar?"),
]


def _ts() -> str:
    return datetime.now().strftime("%H:%M")


def _full_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _friendly_error(exc: Exception) -> str:
    """Map exceptions to user-safe messages (no raw tracebacks in chat)."""
    msg = str(exc).lower()
    if "no such file" in msg or "model" in msg or "vectorizer" in msg:
        return (
            "The assistant could not load its knowledge files. "
            "Please contact your administrator or try again later."
        )
    return "Something went wrong while processing your message. Please try again."


# ---------------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------------
class ChatbotGUI:
    """Tkinter desktop chat UI for the student information assistant."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.debug_mode = tk.BooleanVar(value=False)
        self._typing_visible = False
        self._typing_anim_id = None
        self._typing_dots = 0

        self._last_intent = "—"
        self._last_confidence = 0.0
        self._model_loaded = False

        self._configure_root()
        self._build_ui()
        self._init_engine()
        self._show_welcome(log=True)
        self._update_send_state()

    # ------------------------------------------------------------------
    # WINDOW
    # ------------------------------------------------------------------
    def _configure_root(self):
        self.root.title(APP_TITLE)
        self.root.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")
        self.root.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.root.configure(bg=C_BG)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        try:
            self.root.iconbitmap("assets/icon.ico")
        except Exception:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        self._build_header()
        self._build_chat_area()
        self._build_quick_actions()
        self._build_toolbar()
        self._build_input_area()
        self._build_status_bar()

    def _build_header(self):
        header = tk.Frame(self.root, bg=C_HEADER_BG, pady=14)
        header.grid(row=0, column=0, sticky="ew")

        left = tk.Frame(header, bg=C_HEADER_BG)
        left.pack(side=tk.LEFT, padx=18)

        tk.Label(
            left,
            text="Student Information Assistant",
            font=FONT_HEADER,
            bg=C_HEADER_BG,
            fg=C_HEADER_FG,
        ).pack(anchor="w")

        tk.Label(
            left,
            text="University student support — registration, fees, exams & more",
            font=FONT_SUBHEADER,
            bg=C_HEADER_BG,
            fg=C_HEADER_SUB,
        ).pack(anchor="w", pady=(2, 0))

    def _build_chat_area(self):
        frame = tk.Frame(self.root, bg=C_BG)
        frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=(12, 8))
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        scrollbar = tk.Scrollbar(frame, cursor="arrow")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.chat_display = tk.Text(
            frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
            bg=C_CHAT_BG,
            relief=tk.FLAT,
            bd=0,
            highlightbackground=C_BORDER,
            highlightthickness=1,
            font=FONT_CHAT,
            padx=18,
            pady=16,
            spacing1=4,
            spacing3=4,
            yscrollcommand=scrollbar.set,
            cursor="arrow",
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.chat_display.yview)

        self._configure_chat_tags()

    def _configure_chat_tags(self):
        t = self.chat_display
        t.tag_config(
            "user_bubble",
            background=C_USER_BG,
            foreground=C_USER_FG,
            font=FONT_CHAT,
            lmargin1=120,
            lmargin2=120,
            rmargin=12,
            spacing1=6,
            spacing3=10,
        )
        t.tag_config(
            "bot_bubble",
            background=C_BOT_BG,
            foreground=C_BOT_FG,
            font=FONT_CHAT,
            lmargin1=12,
            lmargin2=12,
            rmargin=120,
            spacing1=6,
            spacing3=10,
        )
        t.tag_config(
            "sys_bubble",
            background=C_SYS_BG,
            foreground=C_SYS_FG,
            font=("Segoe UI", 10),
            lmargin1=60,
            lmargin2=60,
            rmargin=60,
            justify=tk.CENTER,
            spacing1=8,
            spacing3=8,
        )
        t.tag_config("timestamp", foreground="#90A4AE", font=FONT_CHAT_TS)
        t.tag_config(
            "label_user",
            foreground=C_USER_FG,
            font=FONT_LABEL,
            lmargin1=120,
        )
        t.tag_config(
            "label_bot",
            foreground="#37474F",
            font=FONT_LABEL,
            lmargin1=12,
        )
        t.tag_config(
            "debug_info",
            foreground=C_DEBUG_FG,
            font=FONT_DEBUG,
            lmargin1=12,
            lmargin2=12,
            spacing3=8,
        )
        t.tag_config(
            "typing",
            foreground=C_TYPING_FG,
            font=("Segoe UI", 10, "italic"),
            lmargin1=12,
        )

    def _build_quick_actions(self):
        wrap = tk.Frame(self.root, bg=C_BG)
        wrap.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 6))

        tk.Label(
            wrap,
            text="Quick questions:",
            font=("Segoe UI", 9),
            bg=C_BG,
            fg="#546E7A",
        ).pack(anchor="w", pady=(0, 4))

        grid = tk.Frame(wrap, bg=C_BG)
        grid.pack(fill=tk.X)

        for i, (label, query) in enumerate(QUICK_ACTIONS):
            row, col = divmod(i, 4)
            btn = tk.Button(
                grid,
                text=label,
                font=FONT_BTN,
                bg=C_BTN_BG,
                fg=C_BTN_FG,
                activebackground="#E3F2FD",
                activeforeground=C_BTN_FG,
                relief=tk.GROOVE,
                bd=1,
                highlightbackground=C_BTN_BORDER,
                cursor="hand2",
                padx=8,
                pady=5,
                command=lambda q=query: self._send_quick_action(q),
            )
            btn.grid(row=row, column=col, padx=(0, 6), pady=3, sticky="ew")
            grid.grid_columnconfigure(col, weight=1)

    def _build_toolbar(self):
        bar = tk.Frame(self.root, bg=C_TOOLBAR_BG, pady=6)
        bar.grid(row=3, column=0, sticky="ew", padx=16)

        tk.Button(
            bar,
            text="Clear Chat",
            font=FONT_BTN,
            bg="#FFFFFF",
            fg="#C62828",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._on_clear_chat,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            bar,
            text="Export Chat",
            font=FONT_BTN,
            bg="#FFFFFF",
            fg=C_BTN_FG,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._on_export_chat,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            bar,
            text="Reset Context",
            font=FONT_BTN,
            bg="#FFFFFF",
            fg=C_BTN_FG,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._on_reset_context,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Checkbutton(
            bar,
            text="Debug overlay",
            variable=self.debug_mode,
            bg=C_TOOLBAR_BG,
            fg=C_BTN_FG,
            activebackground=C_TOOLBAR_BG,
            selectcolor="#FFFFFF",
            font=FONT_BTN,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=8)

        tk.Label(
            bar,
            text="Ctrl+L — clear chat",
            font=("Segoe UI", 8),
            bg=C_TOOLBAR_BG,
            fg="#90A4AE",
        ).pack(side=tk.RIGHT, padx=4)

    def _build_input_area(self):
        frame = tk.Frame(self.root, bg=C_BG)
        frame.grid(row=4, column=0, sticky="ew", padx=16, pady=(4, 10))
        frame.grid_columnconfigure(0, weight=1)

        self.input_field = tk.Text(
            frame,
            height=2,
            font=FONT_INPUT,
            bg=C_INPUT_BG,
            fg="#263238",
            insertbackground="#263238",
            relief=tk.FLAT,
            highlightbackground=C_BORDER,
            highlightthickness=1,
            wrap=tk.WORD,
        )
        self.input_field.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input_field.bind("<Return>", self._on_enter_pressed)
        self.input_field.bind("<Shift-Return>", self._on_shift_enter)
        self.input_field.bind("<KeyRelease>", self._on_input_changed)
        self.root.bind("<Control-l>", self._on_clear_chat)
        self.root.bind("<Control-L>", self._on_clear_chat)

        self.send_btn = tk.Button(
            frame,
            text="Send  ⏎",
            font=FONT_SEND,
            bg=C_SEND_BG,
            fg=C_SEND_FG,
            activebackground="#2E7D32",
            activeforeground=C_SEND_FG,
            disabledforeground="#E8F5E9",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2",
            command=self._on_send,
        )
        self.send_btn.grid(row=0, column=1, sticky="ns")

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=C_STATUS_BG, pady=5)
        bar.grid(row=5, column=0, sticky="ew")

        self.status_model = tk.StringVar(value="Model: …")
        self.status_intent = tk.StringVar(value="Intent: —")
        self.status_conf = tk.StringVar(value="Confidence: —")
        self.status_flags = tk.StringVar(value="")

        for var, anchor in (
            (self.status_model, "w"),
            (self.status_intent, "w"),
            (self.status_conf, "w"),
            (self.status_flags, "e"),
        ):
            tk.Label(
                bar,
                textvariable=var,
                bg=C_STATUS_BG,
                fg=C_STATUS_FG,
                font=FONT_STATUS,
                anchor=anchor,
            ).pack(side=tk.LEFT if anchor == "w" else tk.RIGHT, padx=12)

    # ------------------------------------------------------------------
    # EVENT HANDLERS
    # ------------------------------------------------------------------
    def _on_enter_pressed(self, event):
        if self.send_btn.cget("state") != tk.DISABLED:
            self._on_send()
        return "break"

    def _on_shift_enter(self, event):
        return None

    def _on_input_changed(self, event=None):
        self._update_send_state()

    def _update_send_state(self):
        text = self.input_field.get("1.0", tk.END).strip()
        if text and self.input_field.cget("state") == tk.NORMAL:
            self.send_btn.configure(state=tk.NORMAL, bg=C_SEND_BG, cursor="hand2")
        else:
            self.send_btn.configure(state=tk.DISABLED, bg=C_SEND_DISABLED, cursor="arrow")

    def _on_clear_chat(self, event=None):
        if self._typing_visible:
            self._hide_typing()
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self._log_session("SYSTEM", "User cleared chat")
        self._show_welcome(log=False)

    def _on_reset_context(self):
        if self.engine is not None and hasattr(self.engine, "memory"):
            try:
                self.engine.memory.clear()
                self._append_system("Conversation context has been reset.")
                self._log_session("SYSTEM", "Context reset by user")
            except Exception as exc:
                print(f"[GUI] Context reset failed: {exc}")
                self._append_system("Could not reset context. Please try again.")
        else:
            self._append_system("Context reset is unavailable while the assistant is offline.")

    def _on_export_chat(self, event=None):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Export Chat Log",
        )
        if not filepath:
            return
        content = self.chat_display.get("1.0", tk.END)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Export successful", f"Chat saved to:\n{filepath}")
        except Exception:
            messagebox.showerror(
                "Export failed",
                "We could not save the chat file. Check the folder permissions and try again.",
            )

    def _send_quick_action(self, query: str):
        if self.input_field.cget("state") != tk.NORMAL:
            return
        self.input_field.delete("1.0", tk.END)
        self.input_field.insert(tk.END, query)
        self._update_send_state()
        self._on_send()

    # ------------------------------------------------------------------
    # ENGINE
    # ------------------------------------------------------------------
    def _init_engine(self):
        if not _BACKEND_OK:
            self.engine = None
            self._model_loaded = False
            self._refresh_status_bar(ood=False, fallback=True)
            self._append_system(
                "The assistant backend could not start. "
                "Please ensure the application files are installed correctly."
            )
            print(f"[GUI] Backend import failed: {_BACKEND_ERROR}")
            return

        try:
            self.engine = ChatbotEngine()
            self._model_loaded = bool(self.engine.is_loaded)
            self._refresh_status_bar()
            if not self._model_loaded:
                self._append_system(
                    "Running in limited mode — trained models were not found. "
                    "Basic answers may still be available."
                )
        except Exception as exc:
            self.engine = None
            self._model_loaded = False
            self._refresh_status_bar(ood=False, fallback=True)
            self._append_system(_friendly_error(exc))
            print(f"[GUI] Engine init failed: {exc}")

    # ------------------------------------------------------------------
    # WELCOME
    # ------------------------------------------------------------------
    def _show_welcome(self, log=True):
        intro = (
            "Welcome to the Student Information Assistant.\n\n"
            "Ask about registration, fees, exams, courses, campus locations, "
            "scholarships, student services, or contacts.\n\n"
            "Tip: use the quick-question buttons below, or type your question and press Enter."
        )
        self._append_system(intro)
        self._append_bot(
            "Hello! I'm here to help with university student information. "
            "What would you like to know?",
            intent="greeting",
            confidence=1.0,
            is_fallback=False,
            is_ood=False,
        )
        if log:
            self._log_session("SYSTEM", "Session started")

    # ------------------------------------------------------------------
    # SEND / RECEIVE
    # ------------------------------------------------------------------
    def _on_send(self, event=None):
        raw = self.input_field.get("1.0", tk.END).strip()
        if not raw:
            return

        self.input_field.delete("1.0", tk.END)
        self._update_send_state()
        self.input_field.focus_set()

        self._append_user(raw)
        self._log_session("USER", raw)
        self._set_input_state(False)
        self._show_typing()

        threading.Thread(
            target=self._get_bot_response,
            args=(raw,),
            daemon=True,
        ).start()

    def _get_bot_response(self, user_text: str):
        intent, confidence, is_fallback, is_ood = "fallback", 0.0, True, False
        try:
            if self.engine is None:
                response = (
                    "The assistant is temporarily unavailable. "
                    "Please restart the application or try again later."
                )
            else:
                response, intent, confidence, is_fallback = self.engine.get_reply(user_text)
                # OOD path in engine uses fallback intent with zero confidence
                is_ood = bool(
                    is_fallback and intent == "fallback" and confidence == 0.0
                )
        except Exception as exc:
            response = _friendly_error(exc)
            intent, confidence, is_fallback, is_ood = "error", 0.0, True, False
            print(f"[GUI ERROR] {exc}")

        self.root.after(
            0,
            self._display_bot_response,
            response,
            intent,
            confidence,
            is_fallback,
            is_ood,
        )

    def _display_bot_response(
        self, response, intent, confidence, is_fallback, is_ood
    ):
        self._hide_typing()
        self._last_intent = intent
        self._last_confidence = confidence

        self._append_bot(
            response,
            intent=intent,
            confidence=confidence,
            is_fallback=is_fallback,
            is_ood=is_ood,
        )
        self._log_session("BOT", response)
        self._refresh_status_bar(
            intent=intent,
            confidence=confidence,
            is_fallback=is_fallback,
            is_ood=is_ood,
        )
        self._set_input_state(True)
        self._update_send_state()
        self.input_field.focus_set()

    # ------------------------------------------------------------------
    # MESSAGE RENDERING
    # ------------------------------------------------------------------
    def _append_user(self, text: str):
        self._write("\n", "timestamp")
        self._write(f"You  ·  {_ts()}\n", "label_user")
        self._write(f"{text}\n", "user_bubble")
        self._scroll_to_end()

    def _append_bot(
        self,
        text: str,
        intent="",
        confidence=0.0,
        is_fallback=False,
        is_ood=False,
    ):
        self._write("\n", "timestamp")
        self._write(f"Assistant  ·  {_ts()}\n", "label_bot")
        self._write(f"{text}\n", "bot_bubble")

        if self.debug_mode.get() and intent:
            flags = []
            if is_ood:
                flags.append("OOD")
            if is_fallback:
                flags.append("FALLBACK")
            flag_str = f"  [{' | '.join(flags)}]" if flags else ""
            self._write(
                f"  ↳ intent: {intent}  ·  confidence: {confidence:.2f}{flag_str}\n",
                "debug_info",
            )
        self._scroll_to_end()

    def _append_system(self, text: str):
        self._write("\n", "timestamp")
        self._write(f"{text}\n", "sys_bubble")
        self._scroll_to_end()

    def _write(self, text: str, tag: str):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, text, tag)
        self.chat_display.configure(state=tk.DISABLED)

    def _scroll_to_end(self):
        self.chat_display.see(tk.END)

    # ------------------------------------------------------------------
    # TYPING INDICATOR
    # ------------------------------------------------------------------
    def _show_typing(self):
        self._typing_visible = True
        self._typing_dots = 0
        self._render_typing_line()
        self._scroll_to_end()

    def _render_typing_line(self):
        if not self._typing_visible:
            return
        self._hide_typing(silent=True)
        dots = "." * ((self._typing_dots % 3) + 1)
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(
            tk.END,
            f"\nAssistant is typing{dots}\n",
            "typing",
        )
        self._typing_marker = "typing_active"
        self.chat_display.mark_set(self._typing_marker, "end-2l linestart")
        self.chat_display.mark_gravity(self._typing_marker, tk.LEFT)
        self.chat_display.configure(state=tk.DISABLED)
        self._scroll_to_end()
        self._typing_dots += 1
        self._typing_anim_id = self.root.after(450, self._animate_typing)

    def _animate_typing(self):
        if self._typing_visible:
            self._render_typing_line()

    def _hide_typing(self, silent=False):
        if self._typing_anim_id:
            self.root.after_cancel(self._typing_anim_id)
            self._typing_anim_id = None
        if not self._typing_visible:
            return
        self._typing_visible = False
        self.chat_display.configure(state=tk.NORMAL)
        try:
            if hasattr(self, "_typing_marker"):
                self.chat_display.delete(self._typing_marker, tk.END)
        except tk.TclError:
            content = self.chat_display.get("1.0", tk.END)
            for line in ("\nAssistant is typing.\n", "\nAssistant is typing..\n", "\nAssistant is typing...\n"):
                if line in content:
                    idx = content.rfind(line)
                    if idx >= 0:
                        self.chat_display.delete(f"1.0+{idx}c", tk.END)
                    break
        self.chat_display.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # INPUT STATE
    # ------------------------------------------------------------------
    def _set_input_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.input_field.configure(
            state=state,
            bg=C_INPUT_BG if enabled else C_INPUT_DISABLED,
        )
        if not enabled:
            self.send_btn.configure(state=tk.DISABLED, bg=C_SEND_DISABLED)
        else:
            self._update_send_state()

    # ------------------------------------------------------------------
    # STATUS BAR
    # ------------------------------------------------------------------
    def _refresh_status_bar(
        self,
        intent=None,
        confidence=None,
        is_fallback=False,
        is_ood=False,
    ):
        if self.engine is None:
            self.status_model.set("Model: unavailable")
        elif self._model_loaded:
            self.status_model.set("Model: loaded")
        else:
            self.status_model.set("Model: fallback mode")

        if intent is not None:
            self._last_intent = intent
            self._last_confidence = confidence if confidence is not None else 0.0

        self.status_intent.set(f"Intent: {self._last_intent}")
        if self._last_confidence is not None and self._last_intent not in ("—", "error"):
            self.status_conf.set(f"Confidence: {self._last_confidence:.2f}")
        else:
            self.status_conf.set("Confidence: —")

        flags = []
        if is_ood:
            flags.append("OOD")
        if is_fallback:
            flags.append("Fallback")
        if intent == "error":
            flags.append("Error")
        self.status_flags.set("  ·  ".join(flags) if flags else "Ready")

    # ------------------------------------------------------------------
    # SESSION LOG
    # ------------------------------------------------------------------
    def _log_session(self, role: str, text: str):
        try:
            os.makedirs("logs", exist_ok=True)
            with open(SESSION_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{_full_ts()}] {role}: {text}\n")
        except Exception:
            pass

    def _on_close(self):
        self._hide_typing()
        self._log_session("SYSTEM", "Session ended")
        self.root.destroy()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    ChatbotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
