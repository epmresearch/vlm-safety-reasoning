with open("C:/Users/Nabeel/.gemini/antigravity/brain/ac032cbb-5b3a-4d5e-ab4c-e09e96fbda70/task.md", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("[ ]", "[x]")

with open("C:/Users/Nabeel/.gemini/antigravity/brain/ac032cbb-5b3a-4d5e-ab4c-e09e96fbda70/task.md", "w", encoding="utf-8") as f:
    f.write(content)
