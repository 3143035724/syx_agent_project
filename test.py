class Agent:
    """AI Agent 基类"""

    # 类属性（所有实例共享）
    category = "AI"

# self 必须显式出现在每个实例方法的第一个参数位置,"显式优于隐式"

    # 初始化方法（构造函数）
    def __init__(self, name, model="gpt-4"):
        self.name = name          # 实例属性
        self.model = model
        self._memory = []         # 单下划线：约定为"内部使用" java->private

    # 实例方法
    def introduce(self):
        return f"我是 {self.name}，使用 {self.model} 模型"

    # 特殊方法 — 定义"字符串表示"
    def __str__(self):
        return f"Agent({self.name})"

    def __repr__(self):
        return f"Agent(name='{self.name}', model='{self.model}')"


# 继承
class CodingAgent(Agent):
    def __init__(self, name, model="gpt-4", languages=None):
        super().__init__(name, model)     # 调用父类 __init__
        self.languages = languages or ["Python"]

    def code_review(self, code: str) -> str:
        return f"[{self.name}] 代码审查完成，发现 [{code}]个优化建议"

# 使用
if __name__ == '__main__':
    agent = CodingAgent("CodeBuddy", languages=["Python", "TypeScript"])
    print(agent.introduce())   # 我是 CodeBuddy，使用 gpt-4 模型
    print(agent.category)      # AI （继承自父类）
    print(agent.code_review("print('hello')")) # [CodeBuddy] 代码审查完成，发现 [print('hello')]个优化建议


def get_coordinates():
    return (116.40, 39.90)  # 实际返回的是一个元组

if __name__ == '__main__':
    # 元组()
    single = (1,) # 单个元组，必须加逗号
    lat, lng = get_coordinates()  # 解包（unpacking）
    print(lat, lng)
    # 列表 []
    mixed = [1, "hello", True, None]
    # 集合{}
    tags = {"python", "agent", "llm"}

    # 推导式
    # ========== 列表推导式 ==========
    squares = []
    for i in range(10):
        # 计算平方，如1*1、2*2、3*3
        squares.append(i ** 2)
    squares = [i ** 2 for i in range(10)]
    # 带条件过滤
    evens = [i for i in range(20) if i % 2 == 0]
    # 嵌套循环（按顺序：外层在前）
    pairs = [(x, y) for x in "ab" for y in "12"]
    # 结果：[('a', '1'), ('a', '2'), ('b', '1'), ('b', '2')]

    # ========== 字典推导式 ==========
    # 构建一个工具名→工具描述映射
    tool_map = {
        "search": "搜索网络",
        "read": "读取文件",
        "write": "写入文件",
    }
    # 调转键值对
    reversed_map = {v: k for k, v in tool_map.items()}
    # 结果：{'搜索网络': 'search', '读取文件': 'read', '写入文件': 'write'}

    # ========== 集合推导式 ==========
    unique_lengths = {len(word) for word in ["ai", "agent", "python", "code"]}
    # 结果：{2, 4, 5, 6}

    # ========== 生成器表达式（惰性求值） ==========
    # 把方括号换成圆括号 — 不会一次性加载全部数据
    # 处理大文件、大批 API 响应时，用 (...) 代替 [...] 能省大量内存
    big_sum = sum(i * i for i in range(1_000_000))  # 内存友好

    # *打包成元组，**打包成字典

import time

def timing_decorator(func):
    """测量函数执行时间的装饰器"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)   # 执行原函数
        elapsed = time.time() - start
        print(f"[计时] {func.__name__} 执行耗时：{elapsed:.3f} 秒")
        return result
    return wrapper

# 使用装饰器 — 等价于 slow_function = timing_decorator(slow_function)
@timing_decorator
def slow_function():
    time.sleep(1.5)
    return "完成"
if __name__ == '__main__':
    print(slow_function())
# 输出：
# [计时] slow_function 执行耗时：1.501 秒
# 完成



def retry(max_attempts=3, delay=1):
    """重试装饰器 — 调用失败时自动重试"""
    import time

    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    # 执行原函数
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise  # 最后一次还失败，抛出异常
                    print(f"第 {attempt + 1} 次失败：{e}，{delay}秒后重试...")
                    time.sleep(delay)
        return wrapper
    return decorator

# 使用
@retry(max_attempts=3, delay=2)
def call_unstable_api():
    # ... 可能失败的 API 调用 ...
    pass

# 场景 1：记录每次工具调用的输入输出
def log_tool_call(func):
    def wrapper(*args, **kwargs):
        print(f"[Tool] 调用 {func.__name__}(args={args}, kwargs={kwargs})")
        result = func(*args, **kwargs)
        print(f"[Tool] {func.__name__} 返回 {result[:50]}...")
        return result
    return wrapper

@log_tool_call
def search_web(query: str) -> str:
    return f"搜索结果：关于 '{query}' 的 10 条相关信息..."

# 场景 2：限流保护
# max_calls_per_second（每秒允许的最大调用次数）
def rate_limit(max_calls_per_second):
    import time
    # 算出两次调用之间的最小间隔
    interval = 1.0 / max_calls_per_second
    last_called = [0]

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 距离上次调用已经过去了多少秒
            elapsed = time.time() - last_called[0]
            if elapsed < interval:
                time.sleep(interval - elapsed)
            result = func(*args, **kwargs)
            # 用调用刚结束的时刻更新「上次调用时间」，供下一次 wrapper 计算间隔
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator
# @rate_limit(2)，单线程、按「调用结束时间」节流的简单实现；多线程下不安全，且没有队列/令牌桶等更精细的策略。
