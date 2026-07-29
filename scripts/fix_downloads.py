"""Fix all generated download_data.py scripts with Windows path support."""
import os

PROJECTS_DIR = "projects"
fixed_count = 0

for root, dirs, files in os.walk(PROJECTS_DIR):
    if "download_data.py" in files:
        path = os.path.join(root, "download_data.py")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Skip if already fixed (contains backslash checks or FOLDER_MAP)
        if "FOLDER_MAP" in content or "\\train\\" in content:
            print(f"  Already fixed: {path}")
            continue

        # Fix path separator checks: /train/ -> \\train\\ or /train/
        content = content.replace(
            "if \"/train\" in path_str: split = \"train\"",
            "if \"\\\\train\\\\\" in path_str or \"/train/\" in path_str: split = \"train\""
        )
        content = content.replace(
            "elif \"/val\" in path_str or \"/valid\" in path_str: split = \"val\"",
            "elif \"\\\\val\\\\\" in path_str or \"/val/\" in path_str: split = \"val\""
        )
        content = content.replace(
            "elif \"/test\" in path_str: split = \"test\"",
            "elif \"\\\\test\\\\\" in path_str or \"/test/\" in path_str: split = \"test\""
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"  Fixed: {path}")
        fixed_count += 1

print(f"\nFixed {fixed_count} download scripts")
