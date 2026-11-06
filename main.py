# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


if __name__ == '__main__':
    # 1 + 1 = 2
    # f"{x}" => String.format
    print(f"1 + 1 = {1+1}")

    score = 85
    if score >= 90:
        print('A')
        # elif => else if
    elif score >= 80:
        print('B')
    else:
        print('C')

    print("-------------")
    # Python — 没有 C 风格 for 循环！
    # range() 是最常用的替代
    # for i in range(5):  # range(stop)
    # for i in range(1, 5):  # range(start, stop) → 1, 2, 3, 4
    # for i in range(0, 10, 2):  # range(start, stop, step) → 0, 2, 4, 6, 8

    # for 循环 — 遍历可迭代对象
    for i in range(5):  # 0, 1, 2, 3, 4
        print(i)

    for item in ["a", "b", "c"]:
        print(item)
    print("-------------")
    # while 循环
    count = 0
    while count < 3:
        print(count)
        count += 1
    print("-------------")

    print("-------for-else用法--------")
    # Python 的循环可以带 else —— 循环正常结束（没被 break）时执行
    for n in range(2, 10):
        for x in range(2, n):
            if n % x == 0:
                break
        else:  # ← 注意：这个 else 属于 for，不是 if！
            print(f"{n} 是质数")
    # 输出：2 是质数、3 是质数、5 是质数、7 是质数


def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    print_info(name="Alice", role="Engineer")
    # name: Alice
    # role: Engineer


