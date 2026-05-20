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
from tkinter import messagebox, scrolledtext, filedialog
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
WIN_WIDTH   = 950
WIN_HEIGHT  = 700
MIN_WIDTH   = 750
MIN_HEIGHT  = 550

# Modernized colour palette
C_BG        = "#F4F7F6"
C_CHAT_BG   = "#FFFFFF"
C_USER_BG   = "#E3F2FD"  # softer blue
C_USER_FG   = "#0D47A1"  
C_BOT_BG    = "#F5F5F5"  # soft gray
C_BOT_FG    = "#212121"
C_SYS_BG    = "#FFF8E1"
C_SYS_FG    = "#F57F17"
C_DEBUG_FG  = "#9E9E9E"
C_INPUT_BG  = "#FFFFFF"
C_SEND_BG   = "#1976D2"
C_SEND_FG   = "#FFFFFF"
C_STATUS_BG = "#263238"
C_STATUS_FG = "#B0BEC5"
C_HEADER_BG = "#0D47A1"  # Matches user FG for branding
C_HEADER_FG = "#FFFFFF"
C_TYPING_FG = "#9E9E9E"
C_BTN_BG    = "#E0E0E0"  # For quick actions
C_BTN_FG    = "#424242"

# Font definitions (resolved after root window exists)
FONT_HEADER  = ("Segoe UI", 16, "bold")
FONT_CHAT    = ("Segoe UI", 12)
FONT_CHAT_TS = ("Segoe UI", 9)
FONT_INPUT   = ("Segoe UI", 12)
FONT_SEND    = ("Segoe UI", 11, "bold")
FONT_STATUS  = ("Segoe UI", 10)
FONT_DEBUG   = ("Segoe UI", 10, "italic")
FONT_BTN     = ("Segoe UI", 10)

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
        self._show_welcome(log=True)

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
        self._build_status_bar()
        self._build_input_area()
        self._build_chat_area()

    def _build_header(self):
        header = tk.Frame(self.root, bg=C_HEADER_BG, pady=12)
        header.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            header,
            text="  Student Information Chatbot",
            font=FONT_HEADER,
            bg=C_HEADER_BG,
            fg=C_HEADER_FG,
        ).pack(side=tk.LEFT, padx=14)

        # Right-aligned buttons
        right_frame = tk.Frame(header, bg=C_HEADER_BG)
        right_frame.pack(side=tk.RIGHT, padx=14)

        # Debug Mode toggle
        debug_cb = tk.Checkbutton(
            right_frame,
            text="Debug Mode",
            variable=self.debug_mode,
            bg=C_HEADER_BG,
            fg=C_HEADER_FG,
            selectcolor="#1565C0", # Active state
            activebackground=C_HEADER_BG,
            activeforeground=C_HEADER_FG,
            font=("Segoe UI", 10),
            cursor="hand2",
        )
        debug_cb.pack(side=tk.RIGHT, padx=(10, 0))

        # Export Chat
        btn_export = tk.Button(
            right_frame, text="Export Chat", font=("Segoe UI", 10),
            bg="#283593", fg=C_HEADER_FG, relief=tk.FLAT, cursor="hand2",
            padx=10, pady=2, command=self._on_export_chat
        )
        btn_export.pack(side=tk.RIGHT, padx=5)

        # Clear Chat
        btn_clear = tk.Button(
            right_frame, text="Clear Chat", font=("Segoe UI", 10),
            bg="#C62828", fg=C_HEADER_FG, relief=tk.FLAT, cursor="hand2",
            padx=10, pady=2, command=self._on_clear_chat
        )
        btn_clear.pack(side=tk.RIGHT, padx=5)

    def _build_chat_area(self):
        """Scrollable Text widget used as the chat display."""
        frame = tk.Frame(self.root, bg=C_BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(10, 0))

        # Scrollbar
        scrollbar = tk.Scrollbar(frame, cursor="arrow")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_display = tk.Text(
            frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
            bg=C_CHAT_BG,
            relief=tk.FLAT,
            bd=1,
            highlightbackground="#E0E0E0", highlightthickness=1,
            font=FONT_CHAT,
            padx=16,
            pady=14,
            spacing1=6,
            spacing3=6,
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
            lmargin1=150, lmargin2=150,   # push right
            rmargin=15,
            spacing1=8, spacing3=8,
        )
        self.chat_display.tag_config(
            "bot_bubble",
            background=C_BOT_BG,
            foreground=C_BOT_FG,
            font=FONT_CHAT,
            lmargin1=15, lmargin2=15,
            rmargin=150,
            spacing1=8, spacing3=8,
        )
        self.chat_display.tag_config(
            "sys_bubble",
            background=C_SYS_BG,
            foreground=C_SYS_FG,
            font=("Segoe UI", 10, "italic"),
            lmargin1=80, lmargin2=80,
            rmargin=80,
            justify=tk.CENTER,
            spacing1=6, spacing3=6,
        )
        self.chat_display.tag_config(
            "timestamp",
            foreground="#999999",
            font=FONT_CHAT_TS,
        )
        self.chat_display.tag_config(
            "label_user",
            foreground=C_USER_FG,
            font=("Segoe UI", 10, "bold"),
            lmargin1=150,
        )
        self.chat_display.tag_config(
            "label_bot",
            foreground=C_BOT_FG,
            font=("Segoe UI", 10, "bold"),
            lmargin1=15,
        )
        self.chat_display.tag_config(
            "debug_info",
            foreground=C_DEBUG_FG,
            font=FONT_DEBUG,
            lmargin1=15, lmargin2=15,
            spacing1=0, spacing3=6,
        )
        self.chat_display.tag_config(
            "typing",
            foreground=C_TYPING_FG,
            font=("Segoe UI", 11, "italic"),
            lmargin1=15,
        )

    def _build_input_area(self):
        """Input row: Quick actions + Multi-line text entry + Send button."""
        bottom_container = tk.Frame(self.root, bg=C_BG)
        # Pack this before chat_area so it sticks to the bottom
        bottom_container.pack(fill=tk.X, side=tk.BOTTOM, padx=14, pady=(8, 14))

        # Quick Actions Bar
        qa_frame = tk.Frame(bottom_container, bg=C_BG)
        qa_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 8))

        topics = ["Registration", "Fees", "Exams", "Courses", "Locations", "Scholarships"]
        for t in topics:
            btn = tk.Button(
                qa_frame, text=t, font=FONT_BTN,
                bg=C_BTN_BG, fg=C_BTN_FG,
                relief=tk.FLAT, cursor="hand2", padx=10, pady=4,
                activebackground="#BDBDBD", activeforeground=C_BTN_FG,
                command=lambda topic=t: self._send_quick_action(topic)
            )
            btn.pack(side=tk.LEFT, padx=(0, 8))

        # Input row: Text + Send Button
        input_frame = tk.Frame(bottom_container, bg=C_BG)
        input_frame.pack(fill=tk.X, side=tk.TOP)

        self.input_field = tk.Text(
            input_frame, height=2, font=FONT_INPUT,
            bg=C_INPUT_BG, fg="#212529", insertbackground="#212529",
            relief=tk.FLAT, bd=1, highlightbackground="#CCCCCC", highlightthickness=1
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=6)
        
        # Keybindings
        self.input_field.bind("<Return>", self._on_enter_pressed)
        self.input_field.bind("<Shift-Return>", self._on_shift_enter)
        self.root.bind("<Control-l>", self._on_clear_chat)
        self.root.bind("<Control-L>", self._on_clear_chat)
        
        self.input_field.focus_set()

        self.send_btn = tk.Button(
            input_frame, text="Send", font=FONT_SEND,
            bg=C_SEND_BG, fg=C_SEND_FG, activebackground="#1565C0", activeforeground=C_SEND_FG,
            relief=tk.FLAT, padx=22, pady=12, cursor="hand2", bd=0,
            command=self._on_send
        )
        self.send_btn.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_status_bar(self):
        """Persistent status bar at the very bottom."""
        self.status_var = tk.StringVar(value="Initializing...")
        bar = tk.Frame(self.root, bg=C_STATUS_BG, pady=4)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(
            bar,
            textvariable=self.status_var,
            bg=C_STATUS_BG,
            fg=C_STATUS_FG,
            font=FONT_STATUS,
            anchor="w",
        ).pack(side=tk.LEFT, padx=14)

    # ------------------------------------------------------------------
    # EVENT HANDLERS
    # ------------------------------------------------------------------
    def _on_enter_pressed(self, event):
        """Send message on Enter key (without Shift)."""
        self._on_send()
        return "break" # Prevent default newline insertion

    def _on_shift_enter(self, event):
        """Allow default behavior to insert newline."""
        return None

    def _on_clear_chat(self, event=None):
        """Clear the chat window and reset."""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self._log_session("SYSTEM", "User cleared chat")
        self._show_welcome(log=False)

    def _on_export_chat(self, event=None):
        """Export the chat display text to a local file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Export Chat Log"
        )
        if not filepath:
            return
        
        content = self.chat_display.get("1.0", tk.END)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Export Successful", f"Chat exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"An error occurred while saving:\n{e}")

    def _send_quick_action(self, topic):
        """Instantly populates the input with a relevant query and sends it."""
        self.input_field.delete("1.0", tk.END)
        queries = {
            "Registration": "How do I register for classes?",
            "Fees": "What is the tuition fee structure?",
            "Exams": "When are the upcoming exams?",
            "Courses": "Tell me about available courses.",
            "Locations": "Where is the library located?",
            "Scholarships": "What scholarships are available?"
        }
        query = queries.get(topic, f"Tell me about {topic}.")
        
        self.input_field.insert(tk.END, query)
        self._on_send()

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
    def _show_welcome(self, log=True):
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
        if log:
            self._log_session("SYSTEM", "Session started")

    # ------------------------------------------------------------------
    # SEND / RECEIVE FLOW
    # ------------------------------------------------------------------
    def _on_send(self, event=None):
        """Called when the user presses Enter or clicks Send."""
        raw = self.input_field.get("1.0", tk.END).strip()
        if not raw:
            return

        # Clear input immediately
        self.input_field.delete("1.0", tk.END)
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
        if enabled:
            self.input_field.configure(bg=C_INPUT_BG)
        else:
            self.input_field.configure(bg="#E9ECEF")
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
