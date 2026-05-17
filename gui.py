# gui.py
"""
University Student Information Chatbot - Phase 9: GUI Application

A professional desktop chat interface built with Tkinter.

Features:
  - Styled user / bot / system message bubbles with timestamps
  - Typing indicator (non-blocking via threading)
  - Status bar showing model state, confidence threshold, and last event
  - Optional debug mode (intent + confidence overlay)
  - Chat session saved to logs/chat_session.txt
  - Full NLP pipeline: OOD detection, NER, ML classification, context memory
  - Graceful error handling — never crashes
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, scrolledtext
from datetime import datetime

# ---------------------------------------------------------------------------
# Backend import — isolate so GUI can display an error if model is missing
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
APP_TITLE   = "Student Information Chatbot"
WIN_WIDTH   = 900
WIN_HEIGHT  = 650
MIN_WIDTH   = 700
MIN_HEIGHT  = 500

# Colour palette
C_BG        = "#F5F6FA"        # window background
C_CHAT_BG   = "#FFFFFF"        # chat canvas
C_USER_BG   = "#D4EDDA"        # user bubble  (soft green)
C_USER_FG   = "#155724"
C_BOT_BG    = "#D1ECF1"        # bot bubble   (soft blue)
C_BOT_FG    = "#0C5460"
C_SYS_BG    = "#FFF3CD"        # system msg   (soft yellow)
C_SYS_FG    = "#856404"
C_DEBUG_FG  = "#6C757D"        # debug overlay (grey)
C_INPUT_BG  = "#FFFFFF"
C_SEND_BG   = "#007BFF"
C_SEND_FG   = "#FFFFFF"
C_STATUS_BG = "#343A40"
C_STATUS_FG = "#ADB5BD"
C_HEADER_BG = "#1A237E"        # deep university blue
C_HEADER_FG = "#FFFFFF"
C_TYPING_FG = "#6C757D"

# Font definitions (resolved after root window exists)
FONT_HEADER  = ("Segoe UI", 14, "bold")
FONT_CHAT    = ("Segoe UI", 11)
FONT_CHAT_TS = ("Segoe UI", 8)
FONT_INPUT   = ("Segoe UI", 11)
FONT_SEND    = ("Segoe UI", 10, "bold")
FONT_STATUS  = ("Segoe UI", 9)
FONT_DEBUG   = ("Segoe UI", 9, "italic")

SESSION_LOG = os.path.join("logs", "chat_session.txt")


# ---------------------------------------------------------------------------
# HELPER — timestamp string
# ---------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now().strftime("%H:%M")


def _full_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# MAIN APPLICATION CLASS
# ---------------------------------------------------------------------------
class ChatbotGUI:
    """
    Main GUI application class.

    Responsibilities:
      - Build and manage the Tkinter window.
      - Render user/bot/system messages with distinct styling.
      - Spawn bot response in a background thread to keep UI responsive.
      - Delegate all NLP work to ChatbotEngine (from app.py).
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.debug_mode = tk.BooleanVar(value=False)
        self._typing_id = None          # after() handle for typing indicator
        self._typing_visible = False

        self._configure_root()
        self._build_ui()
        self._init_engine()
        self._show_welcome()

    # ------------------------------------------------------------------
    # WINDOW SETUP
    # ------------------------------------------------------------------
    def _configure_root(self):
        self.root.title(APP_TITLE)
        self.root.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")
        self.root.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.root.configure(bg=C_BG)
        # Icon (silently ignore if not found)
        try:
            self.root.iconbitmap("assets/icon.ico")
        except Exception:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------
    def _build_ui(self):
        self._build_header()
        self._build_chat_area()
        self._build_input_area()
        self._build_status_bar()

    def _build_header(self):
        header = tk.Frame(self.root, bg=C_HEADER_BG, pady=10)
        header.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            header,
            text="  STUDENT INFORMATION CHATBOT",
            font=FONT_HEADER,
            bg=C_HEADER_BG,
            fg=C_HEADER_FG,
        ).pack(side=tk.LEFT, padx=14)

        # Debug toggle checkbox on the right
        debug_cb = tk.Checkbutton(
            header,
            text="Debug Mode",
            variable=self.debug_mode,
            bg=C_HEADER_BG,
            fg=C_HEADER_FG,
            selectcolor=C_HEADER_BG,
            activebackground=C_HEADER_BG,
            activeforeground=C_HEADER_FG,
            font=("Segoe UI", 9),
            cursor="hand2",
        )
        debug_cb.pack(side=tk.RIGHT, padx=14)

    def _build_chat_area(self):
        """Scrollable Text widget used as the chat display."""
        frame = tk.Frame(self.root, bg=C_BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 0))

        # Scrollbar
        scrollbar = tk.Scrollbar(frame, cursor="arrow")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_display = tk.Text(
            frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
            bg=C_CHAT_BG,
            relief=tk.FLAT,
            bd=0,
            font=FONT_CHAT,
            padx=12,
            pady=10,
            spacing1=4,
            spacing3=4,
            yscrollcommand=scrollbar.set,
            cursor="arrow",
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.chat_display.yview)

        # Define text tags for message styling
        self.chat_display.tag_config(
            "user_bubble",
            background=C_USER_BG,
            foreground=C_USER_FG,
            font=FONT_CHAT,
            relief="flat",
            lmargin1=200, lmargin2=200,   # push right
            rmargin=10,
            spacing1=6, spacing3=6,
        )
        self.chat_display.tag_config(
            "bot_bubble",
            background=C_BOT_BG,
            foreground=C_BOT_FG,
            font=FONT_CHAT,
            lmargin1=10, lmargin2=10,
            rmargin=200,
            spacing1=6, spacing3=6,
        )
        self.chat_display.tag_config(
            "sys_bubble",
            background=C_SYS_BG,
            foreground=C_SYS_FG,
            font=("Segoe UI", 10, "italic"),
            lmargin1=60, lmargin2=60,
            rmargin=60,
            justify=tk.CENTER,
            spacing1=4, spacing3=4,
        )
        self.chat_display.tag_config(
            "timestamp",
            foreground="#999999",
            font=FONT_CHAT_TS,
        )
        self.chat_display.tag_config(
            "label_user",
            foreground=C_USER_FG,
            font=("Segoe UI", 9, "bold"),
            lmargin1=200,
        )
        self.chat_display.tag_config(
            "label_bot",
            foreground=C_BOT_FG,
            font=("Segoe UI", 9, "bold"),
            lmargin1=10,
        )
        self.chat_display.tag_config(
            "debug_info",
            foreground=C_DEBUG_FG,
            font=FONT_DEBUG,
            lmargin1=10, lmargin2=10,
            spacing1=0, spacing3=4,
        )
        self.chat_display.tag_config(
            "typing",
            foreground=C_TYPING_FG,
            font=("Segoe UI", 10, "italic"),
            lmargin1=10,
        )

    def _build_input_area(self):
        """Input row: text entry + Send button."""
        frame = tk.Frame(self.root, bg=C_BG, pady=8)
        frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(4, 4))

        self.input_var = tk.StringVar()

        self.input_field = tk.Entry(
            frame,
            textvariable=self.input_var,
            font=FONT_INPUT,
            bg=C_INPUT_BG,
            fg="#212529",
            insertbackground="#212529",
            relief=tk.GROOVE,
            bd=2,
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 8))
        self.input_field.bind("<Return>", self._on_send)
        self.input_field.bind("<KP_Enter>", self._on_send)  # numpad Enter
        self.input_field.focus_set()

        self.send_btn = tk.Button(
            frame,
            text="Send",
            command=self._on_send,
            font=FONT_SEND,
            bg=C_SEND_BG,
            fg=C_SEND_FG,
            activebackground="#0056B3",
            activeforeground=C_SEND_FG,
            relief=tk.FLAT,
            padx=18,
            pady=6,
            cursor="hand2",
            bd=0,
        )
        self.send_btn.pack(side=tk.RIGHT)

    def _build_status_bar(self):
        """Persistent status bar at the very bottom."""
        self.status_var = tk.StringVar(value="Initializing...")
        bar = tk.Frame(self.root, bg=C_STATUS_BG, pady=3)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(
            bar,
            textvariable=self.status_var,
            bg=C_STATUS_BG,
            fg=C_STATUS_FG,
            font=FONT_STATUS,
            anchor="w",
        ).pack(side=tk.LEFT, padx=10)

    # ------------------------------------------------------------------
    # ENGINE INITIALIZATION
    # ------------------------------------------------------------------
    def _init_engine(self):
        """Load ChatbotEngine; display error if backend failed to import."""
        if not _BACKEND_OK:
            self._set_status(f"ERROR: Backend failed to load — {_BACKEND_ERROR[:80]}")
            self._append_system(f"Backend import error: {_BACKEND_ERROR}")
            self.engine = None
            return

        try:
            self.engine = ChatbotEngine()
            threshold = self.engine.model is not None
            if self.engine.is_loaded:
                self._set_status(
                    f"System Ready  |  Confidence Threshold: 0.65  |  Model: Loaded"
                )
            else:
                self._set_status("WARNING: Model not found — running in fallback-only mode")
        except Exception as exc:
            self.engine = None
            self._set_status(f"Engine error: {exc}")
            self._append_system(f"Could not start chatbot engine: {exc}")

    # ------------------------------------------------------------------
    # WELCOME MESSAGE
    # ------------------------------------------------------------------
    def _show_welcome(self):
        lines = [
            "=" * 52,
            "   STUDENT INFORMATION CHATBOT",
            "=" * 52,
            "",
            "Supported Topics:",
            "  - Registration & Enrollment",
            "  - Courses & Programs",
            "  - Tuition & Fees",
            "  - Exam Schedules",
            "  - Academic Calendars",
            "  - Campus Locations",
            "  - University Contacts",
            "  - Scholarships",
            "  - Student Services",
            "",
            "=" * 52,
            "",
        ]
        self._append_system("\n".join(lines))
        self._append_bot(
            "Welcome! I'm your university student information assistant. "
            "How can I help you today?",
            intent="greeting", confidence=1.0, is_fallback=False, is_ood=False,
        )
        self._log_session("SYSTEM", "Session started")

    # ------------------------------------------------------------------
    # SEND / RECEIVE FLOW
    # ------------------------------------------------------------------
    def _on_send(self, event=None):
        """Called when the user presses Enter or clicks Send."""
        raw = self.input_var.get().strip()
        if not raw:
            return

        # Clear input immediately
        self.input_var.set("")
        self.input_field.focus_set()

        # Display user message
        self._append_user(raw)
        self._log_session("USER", raw)

        # Disable input while bot is responding
        self._set_input_state(False)

        # Show typing indicator, then dispatch bot response in background thread
        self._show_typing()
        thread = threading.Thread(target=self._get_bot_response, args=(raw,), daemon=True)
        thread.start()

    def _get_bot_response(self, user_text: str):
        """Runs in a background thread — calls NLP engine, then schedules UI update."""
        try:
            if self.engine is None:
                response  = "The chatbot engine is not available. Please restart the application."
                intent, confidence, is_fallback, is_ood = "fallback", 0.0, True, False
            else:
                response, intent, confidence, is_fallback = self.engine.get_reply(user_text)
                # Determine OOD from response metadata (detect_ood already ran inside engine)
                from preprocess import detect_ood
                is_ood = detect_ood(user_text)
        except Exception as exc:
            response  = "An unexpected error occurred. Please try again."
            intent, confidence, is_fallback, is_ood = "error", 0.0, True, False
            print(f"[GUI ERROR] {exc}")

        # Schedule UI update back on main thread
        self.root.after(
            0,
            self._display_bot_response,
            response, intent, confidence, is_fallback, is_ood,
        )

    def _display_bot_response(self, response, intent, confidence, is_fallback, is_ood):
        """Called on main thread after bot computes its reply."""
        self._hide_typing()

        self._append_bot(response, intent=intent, confidence=confidence,
                         is_fallback=is_fallback, is_ood=is_ood)
        self._log_session("BOT", response)

        # Update status bar
        if is_ood:
            self._set_status(f"OOD Detected  |  Fallback triggered  |  Input rejected safely")
        elif is_fallback:
            self._set_status(f"Fallback triggered  |  Intent: {intent}  |  Conf: {confidence:.2f}")
        else:
            self._set_status(f"Intent: {intent}  |  Confidence: {confidence:.2f}  |  Ready")

        self._set_input_state(True)
        self.input_field.focus_set()

    # ------------------------------------------------------------------
    # MESSAGE RENDERING
    # ------------------------------------------------------------------
    def _append_user(self, text: str):
        self._insert_line()
        self._write(f"You   {_ts()}\n", "label_user")
        self._write(f"{text}\n", "user_bubble")

    def _append_bot(self, text: str, intent="", confidence=0.0,
                    is_fallback=False, is_ood=False):
        self._insert_line()
        self._write(f"Assistant   {_ts()}\n", "label_bot")
        self._write(f"{text}\n", "bot_bubble")

        # Debug overlay (only shown when debug mode is ON)
        if self.debug_mode.get() and intent:
            ood_tag  = "  [OOD]" if is_ood else ""
            fb_tag   = "  [FALLBACK]" if is_fallback else ""
            dbg_line = f"  Intent: {intent}  |  Conf: {confidence:.2f}{ood_tag}{fb_tag}\n"
            self._write(dbg_line, "debug_info")

    def _append_system(self, text: str):
        self._insert_line()
        self._write(f"{text}\n", "sys_bubble")

    def _write(self, text: str, tag: str):
        """Thread-safe write to the read-only Text widget."""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, text, tag)
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _insert_line(self):
        """Insert a blank separator line."""
        self._write("\n", "timestamp")

    # ------------------------------------------------------------------
    # TYPING INDICATOR
    # ------------------------------------------------------------------
    def _show_typing(self):
        self._typing_visible = True
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\nAssistant is typing...\n", "typing")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _hide_typing(self):
        if not self._typing_visible:
            return
        self._typing_visible = False
        self.chat_display.configure(state=tk.NORMAL)
        content = self.chat_display.get("1.0", tk.END)
        indicator = "\nAssistant is typing...\n"
        if content.endswith(indicator):
            idx = f"end - {len(indicator)}c"
            self.chat_display.delete(idx, tk.END)
        self.chat_display.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # INPUT STATE
    # ------------------------------------------------------------------
    def _set_input_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.input_field.configure(state=state)
        self.send_btn.configure(state=state)

    # ------------------------------------------------------------------
    # STATUS BAR
    # ------------------------------------------------------------------
    def _set_status(self, msg: str):
        self.status_var.set(f"  {msg}")

    # ------------------------------------------------------------------
    # SESSION LOGGING
    # ------------------------------------------------------------------
    def _log_session(self, role: str, text: str):
        """Append a plain-text record to logs/chat_session.txt."""
        try:
            os.makedirs("logs", exist_ok=True)
            with open(SESSION_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{_full_ts()}] {role}: {text}\n")
        except Exception:
            pass  # Never crash the GUI over a log failure

    # ------------------------------------------------------------------
    # WINDOW CLOSE
    # ------------------------------------------------------------------
    def _on_close(self):
        self._log_session("SYSTEM", "Session ended")
        self.root.destroy()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app  = ChatbotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
