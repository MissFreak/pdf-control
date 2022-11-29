'''
合并多个文件夹中的同类型文件
'''

import os
root = r"D:\tyg_code_from_web\course-main\chapters\zh-CN\chapter2"
file_ending = ".mdx"
total_content = ""
saved_file = os.path.join(root, "merged_file.md")

for path, subdirs, files in os.walk(root):
    for name in files:
        if name.endswith(file_ending) and "8" not in name:
            with open(os.path.join(path, name), "r", encoding="utf-8") as f:
                 total_content += f.read()

import re
total_content = re.sub("\n<[^<(>)]+\/>", "", total_content)
total_content = re.sub("(?<=\W\n)\{[^{}]+\}\n", "", total_content)
total_content = re.sub("```py.*", "```python", total_content)

total_content = re.sub("\n+", "\n", total_content)

print(total_content[:500])
with open(saved_file, "w", encoding="utf-8") as f:
    f.write(total_content)
