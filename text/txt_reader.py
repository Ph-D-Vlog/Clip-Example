def read_txt(file_name):

    non_empty_lines = []

    # 打开文件
    with open(file_name, 'r') as file:

        # 逐行读取文件内容
        for line in file:
            # 去除每行两边的空白字符（包括换行符）
            line = line.strip()
            # 如果行不为空则添加到列表中
            if line:
                non_empty_lines.append(line)

    return non_empty_lines


if __name__ == '__main__':

    pass
