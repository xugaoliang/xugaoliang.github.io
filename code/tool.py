#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2018-09-23 18:15:49
@author: wind
'''

import argparse
import os
import re


def get_file_name(filename):
    assert filename[-3:] == ".md"
    if re.match("\d{4}-\d{2}-\d{2}", filename):
        return filename[11:-3]
    else:
        return filename[:-3]


def get_markdown_list(name):
    result = ""
    g = os.walk(f'{name}/')
    for dirpath, dirnames, filenames in g:
        tmp = dirpath.strip("/").split("/")
        l = len(tmp)+1
        t = f"{'  '*(l-1)}*"
        n = tmp[-1]
        result += f"{t} {n}\n\n"

        filenames = sorted(filenames)
        for filename in filenames:
            if filename.endswith('.md'):
                p = os.path.join(dirpath, filename)
                n = get_file_name(filename)
                # n = filename[:-3].replace(" ","&nbsp;")
                # p = filepath.replace(" ","&nbsp;")
                result += f"{'  '*l}* [{n}]({p})\n"

        result += "\n"
    return result


def write_content(file_handle, content):
    file_handle.write(content)


def create_summary():
    with open('./SUMMARY.md', 'w', encoding='utf-8') as f:
        write_content(f, """
# 成长

* [序言](README.md)

""")

        write_content(f, get_markdown_list("工具"))
        write_content(f, get_markdown_list("读书"))
        write_content(f, get_markdown_list("AI"))
        write_content(f, get_markdown_list("理财"))
        write_content(f, get_markdown_list("健康"))
        write_content(f, get_markdown_list("日记"))
        write_content(f, get_markdown_list("草稿"))


def main():
    # parser = argparse.ArgumentParser(description="文件解析工具")
    # parser.add_argument('-s','--summary',action='store_true', default=False,help='是否更新SUMMARY内容')

    # args = parser.parse_args()
    create_summary()


if __name__ == '__main__':
    main()


# 更新summary
# python code/tool.py -s
