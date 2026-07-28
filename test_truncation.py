import re

def attempt_truncation_repair(snippet: str):
    s = snippet
    stack = []
    in_string = False
    escape = False
    last_safe_cut = 0

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
                last_safe_cut = i + 1
            continue
        if ch == '"':
            in_string = True
        elif ch in '{[':
            stack.append(ch)
        elif ch in '}]':
            if stack:
                stack.pop()
            last_safe_cut = i + 1
        elif ch == ',':
            last_safe_cut = i  # cut BEFORE comma

    if in_string:
        s = s[:last_safe_cut]
        print(f"DEBUG: after cut, s is:\n{s}\n---")
        # recompute stack
        stack = []
        in_string = False
        escape = False
        for ch in s:
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in '{[':
                stack.append(ch)
            elif ch in '}]':
                if stack:
                    stack.pop()

    s = re.sub(r'[,:]\s*$', '', s.rstrip())
    s = re.sub(r'"(?:[^"\\]|\\.)*"\s*:?\s*$', '', s)
    s = re.sub(r'[,:]\s*$', '', s.rstrip())

    closers = {'{': '}', '[': ']'}
    for opener in reversed(stack):
        s += closers[opener]

    return s if s.strip() else None

raw = """{
  "caption": "A construction site...",
  "rule_3_violation": null,
  "rule_4_violation": {
    "reason": "The excavator is positioned"""

print(attempt_truncation_repair(raw))
