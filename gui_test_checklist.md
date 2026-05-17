# GUI Test Checklist — Student Information Chatbot (Phase 9)

Use this checklist to manually verify the GUI before submission.
Run `gui.py` with: `.venv\Scripts\python.exe gui.py`

---

## 1. Launch & Startup

| # | Test | Expected Result | Pass/Fail |
|---|------|----------------|-----------|
| 1.1 | Window launches | 900×650 window opens with title "Student Information Chatbot" | |
| 1.2 | Header visible | Deep blue header with app name and "Debug Mode" checkbox | |
| 1.3 | Model loads | Status bar shows "System Ready \| Confidence Threshold: 0.65 \| Model: Loaded" | |
| 1.4 | Welcome message | System welcome block + bot greeting appear in chat area | |
| 1.5 | Input field focused | Cursor is in text input on startup | |

---

## 2. Message Display

| # | Test | Expected Result | Pass/Fail |
|---|------|----------------|-----------|
| 2.1 | User message styling | Right-aligned, soft green background, labelled "You HH:MM" | |
| 2.2 | Bot message styling | Left-aligned, soft blue background, labelled "Assistant HH:MM" | |
| 2.3 | System message styling | Centered, soft yellow background, monospace-style welcome block | |
| 2.4 | Timestamps | Each message has a time label (HH:MM format) | |
| 2.5 | Multiline bot response | Long bot responses wrap neatly and remain readable | |

---

## 3. Input Handling

| # | Test | Expected Result | Pass/Fail |
|---|------|----------------|-----------|
| 3.1 | Enter key sends | Pressing Enter submits the message | |
| 3.2 | Send button sends | Clicking Send submits the message | |
| 3.3 | Empty message ignored | Pressing Enter or Send with empty input does nothing | |
| 3.4 | Input clears after send | Text field empties after message is sent | |
| 3.5 | Input disabled during response | Field and button are greyed out while bot is processing | |
| 3.6 | Input re-enabled after response | Field and button become active again after bot replies | |

---

## 4. Typing Indicator

| # | Test | Expected Result | Pass/Fail |
|---|------|----------------|-----------|
| 4.1 | Indicator appears | "Assistant is typing..." appears in grey italics after sending | |
| 4.2 | Indicator disappears | Typing indicator is removed once the bot's reply is displayed | |
| 4.3 | UI remains responsive | Window can be scrolled/resized while bot is typing | |

---

## 5. Scrolling & History

| # | Test | Expected Result | Pass/Fail |
|---|------|----------------|-----------|
| 5.1 | Scrollbar visible | Vertical scrollbar on the right of the chat area | |
| 5.2 | Auto-scroll to bottom | Latest message always visible automatically | |
| 5.3 | Manual scroll works | User can scroll up to read earlier messages | |
| 5.4 | 20+ messages | Chat remains stable and readable with many messages | |

---

## 6. NLP Pipeline Integration

| # | Test | Expected Result | Pass/Fail |
|---|------|----------------|-----------|
| 6.1 | "where is the library" | Bot replies with library location | |
| 6.2 | "tuition fees" | Bot replies with fee information | |
| 6.3 | "exam schedule" | Bot replies with exam schedule info | |
| 6.4 | "bitcoin price" | Bot replies with a professional fallback/rejection | |
| 6.5 | "weather today" | Bot replies with a professional fallback/rejection | |
| 6.6 | Follow-up context | After asking about SE dept, "what time does it open" gets correct hours | |

---

## 7. Fallback & OOD Handling

| # | Test | Expected Result | Pass/Fail |
|---|------|----------------|-----------|
| 7.1 | OOD query status bar | Status bar shows "OOD Detected \| Fallback triggered" | |
| 7.2 | Fallback message display | Bot shows a professional domain-restriction message | |
| 7.3 | Fallback rotation | Sending 3 OOD queries shows different fallback messages | |
| 7.4 | Low-confidence query | Vague queries like "yo" trigger fallback, not a wrong intent | |

---

## 8. Debug Mode

| # | Test | Expected Result | Pass/Fail |
|---|------|----------------|-----------|
| 8.1 | Debug OFF (default) | No intent/confidence line under bot messages | |
| 8.2 | Debug ON | Each bot message shows "Intent: X \| Conf: 0.XX" below it | |
| 8.3 | OOD in debug mode | Debug line shows "[OOD]" tag | |
| 8.4 | Fallback in debug mode | Debug line shows "[FALLBACK]" tag | |
| 8.5 | Toggle works live | Enabling/disabling debug mid-conversation takes effect immediately | |

---

## 9. Status Bar

| # | Test | Expected Result | Pass/Fail |
|---|------|----------------|-----------|
| 9.1 | Startup status | "System Ready \| Confidence Threshold: 0.65 \| Model: Loaded" | |
| 9.2 | After valid response | "Intent: X \| Confidence: Y \| Ready" | |
| 9.3 | After OOD query | "OOD Detected \| Fallback triggered \| Input rejected safely" | |
| 9.4 | After fallback | "Fallback triggered \| Intent: fallback \| Conf: X.XX" | |

---

## 10. Session Logging

| # | Test | Expected Result | Pass/Fail |
|---|------|----------------|-----------|
| 10.1 | `chat_session.txt` created | File appears in `logs/` after first message | |
| 10.2 | USER lines logged | Each user message recorded as `[timestamp] USER: ...` | |
| 10.3 | BOT lines logged | Each bot reply recorded as `[timestamp] BOT: ...` | |
| 10.4 | Session start/end | "Session started" and "Session ended" markers present | |
| 10.5 | CSV logging active | `conversation_history.csv` updated with Phase 8 schema | |

---

## 11. Error Handling & Stability

| # | Test | Expected Result | Pass/Fail |
|---|------|----------------|-----------|
| 11.1 | No crash on long input | Pasting 500+ chars doesn't crash | |
| 11.2 | No crash on special chars | Emoji, symbols, non-ASCII input handled gracefully | |
| 11.3 | No crash on rapid sends | Clicking Send rapidly does not break the UI | |
| 11.4 | Window close | Closing the window saves "Session ended" to log cleanly | |
| 11.5 | Resize stable | Resizing the window keeps all elements properly laid out | |

---

## Notes

- Run checklist after every significant code change
- Record the date of testing below

**Test Date:** _______________  
**Tester:** _______________  
**Result:** All Pass / Partially Passing / Failing — ___ / 45 tests passed
