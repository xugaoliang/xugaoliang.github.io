import os
import re


def rename_file(file_dir, old_name, new_name):
    old_just_name = os.path.splitext(old_name)[0]
    new_just_name = os.path.splitext(new_name)[0]

    src = os.path.join(file_dir, old_name)
    tgt = os.path.join(file_dir, new_name)
    assert os.path.isfile(src)
    os.rename(src, tgt)
    # 重命名资源文件
    src = os.path.join(file_dir, ".assets", old_just_name)
    tgt = os.path.join(file_dir, ".assets", new_just_name)
    if os.path.isdir(src):
        os.rename(src, tgt)
    # 重命名文件内的引用资源
    tgt = os.path.join(file_dir, new_name)
    with open(tgt, "r") as f:
        s = f.read()
    src_txt = f"(.assets/{old_just_name}/"
    tgt_txt = f"(.assets/{new_just_name}/"
    ns = s.replace(src_txt, tgt_txt)
    if ns != s:
        with open(tgt, "w") as f:
            f.write(ns)


def rename(files_dir):
    g = os.walk(files_dir)
    for dirpath, dirnames, filenames in g:
        for filename in filenames:
            if filename.endswith('.md'):
                if re.match("\d{8}_", filename):
                    y = filename[:4]
                    m = filename[4:6]
                    d = filename[6:8]
                    f = filename[9:]
                    new_filename = f"{y}-{m}-{d}-{f}"
                    rename_file(dirpath, filename, new_filename)


def main():
    rename("日记")


if __name__ == "__main__":
    main()
