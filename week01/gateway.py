"""StreamingAgentGateway：面向 DeepSeek 的流式 Agent 运行网关。"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import Callable, Generator, Iterable, Literal, Optional, Union

from openai import OpenAI

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


# ---------------------------------------------------------------------------
# RunEvent：创建 / 获取 / 停止 三类独立事件
# ---------------------------------------------------------------------------


@dataclass
class RunCreateEvent:
    """创建一次 Agent Run。"""

    run_id: str
    input: str
    model: str = DEFAULT_MODEL
    status: Literal["created"] = "created"
    type: Literal["run.create"] = "run.create"
    created_at: float = field(default_factory=time.time)


@dataclass
class RunGetEvent:
    """获取一次 Agent Run 的当前状态。"""

    run_id: str
    status: Literal["created", "running", "completed", "stopped", "failed", "not_found"]
    output: Optional[str] = None
    error: Optional[str] = None
    type: Literal["run.get"] = "run.get"
    retrieved_at: float = field(default_factory=time.time)


@dataclass
class RunStopEvent:
    """停止一次进行中的 Agent Run。"""

    run_id: str
    status: Literal["stopping", "stopped", "not_found", "already_terminal"]
    reason: Optional[str] = None
    type: Literal["run.stop"] = "run.stop"
    stopped_at: float = field(default_factory=time.time)


RunEvent = Union[RunCreateEvent, RunGetEvent, RunStopEvent]

_EVENT_TYPES: dict[str, type] = {
    "run.create": RunCreateEvent,
    "run.get": RunGetEvent,
    "run.stop": RunStopEvent,
}


# ---------------------------------------------------------------------------
# SSE 编解码
# ---------------------------------------------------------------------------


class SSECodecError(ValueError):
    """SSE 帧格式或事件载荷不合法。"""


def encode_sse(event: RunEvent, *, event_id: Optional[str] = None) -> str:
    """将 RunEvent 编码为一条 SSE 帧。

    格式：
        id: <optional>
        event: run.create|run.get|run.stop
        data: <json>

    """
    payload = asdict(event)
    event_name = payload.get("type") or getattr(event, "type")
    data_lines = json.dumps(payload, ensure_ascii=False).splitlines() or [""]

    parts: list[str] = []
    if event_id is not None:
        parts.append(f"id: {event_id}")
    parts.append(f"event: {event_name}")
    for line in data_lines:
        parts.append(f"data: {line}")
    return "\n".join(parts) + "\n\n"


def encode_sse_stream(events: Iterable[RunEvent]) -> str:
    """将多个 RunEvent 编码为连续 SSE 文本。"""
    return "".join(encode_sse(event, event_id=str(i)) for i, event in enumerate(events))


def _parse_sse_frame(frame: str) -> RunEvent:
    """解析单条 SSE 帧（不含末尾空行）。"""
    event_name: Optional[str] = None
    data_parts: list[str] = []

    for raw_line in frame.splitlines():
        line = raw_line.rstrip("\r")
        if not line or line.startswith(":"):
            # 空行或注释行
            continue
        if line.startswith("event:"):
            event_name = line[6:].lstrip()
        elif line.startswith("data:"):
            data_parts.append(line[5:].lstrip())
        # id: / retry: 等字段忽略，不影响事件重建

    if not data_parts:
        raise SSECodecError(f"SSE 帧缺少 data 字段: {frame!r}")

    try:
        payload = json.loads("\n".join(data_parts))
    except json.JSONDecodeError as exc:
        raise SSECodecError(f"SSE data 不是合法 JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise SSECodecError("SSE data 必须是 JSON 对象")

    type_name = event_name or payload.get("type")
    if not type_name:
        raise SSECodecError("无法确定事件类型（缺少 event 字段与 data.type）")

    cls = _EVENT_TYPES.get(type_name)
    if cls is None:
        raise SSECodecError(f"未知事件类型: {type_name}")

    payload = {**payload, "type": type_name}
    allowed = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in payload.items() if k in allowed}
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise SSECodecError(f"无法还原 {cls.__name__}: {exc}") from exc


def decode_sse(text: str) -> RunEvent:
    """解码恰好一条 SSE 帧；若含多条则抛错。"""
    events = list(iter_sse(text))
    if len(events) != 1:
        raise SSECodecError(f"期望恰好 1 条 SSE 事件，实际得到 {len(events)} 条")
    return events[0]


def iter_sse(text: str) -> Generator[RunEvent, None, None]:
    """从 SSE 文本中按帧迭代解码为 RunEvent。"""
    # 统一换行，按空行分帧
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = normalized.split("\n\n")
    for chunk in chunks:
        frame = chunk.strip("\n")
        if not frame:
            continue
        # 纯注释帧跳过
        if all(line.startswith(":") or not line.strip() for line in frame.splitlines()):
            continue
        yield _parse_sse_frame(frame)


def decode_sse_stream(text: str) -> list[RunEvent]:
    """解码整段 SSE 文本为 RunEvent 列表。"""
    return list(iter_sse(text))


class SSEDecoder:
    """增量 SSE 解码器：适合从网络流逐块喂入。"""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> list[RunEvent]:
        """喂入文本块，返回本轮可完整解析出的事件。"""
        self._buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
        events: list[RunEvent] = []
        while True:
            sep = self._buffer.find("\n\n")
            if sep < 0:
                break
            frame = self._buffer[:sep]
            self._buffer = self._buffer[sep + 2 :]
            frame = frame.strip("\n")
            if not frame:
                continue
            if all(line.startswith(":") or not line.strip() for line in frame.splitlines()):
                continue
            events.append(_parse_sse_frame(frame))
        return events

    def flush(self) -> list[RunEvent]:
        """结束时冲刷缓冲区中残留的不完整帧（若可解析）。"""
        leftover = self._buffer.strip()
        self._buffer = ""
        if not leftover:
            return []
        return [_parse_sse_frame(leftover)]


# ---------------------------------------------------------------------------
# ModelAdapter：厂商 SDK → 统一流式增量
# ---------------------------------------------------------------------------


class ModelAdapter(ABC):
    """模型适配器：把厂商客户端差异屏蔽成统一的 token 增量流。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """适配后的模型标识。"""

    @abstractmethod
    def stream_chunks(
        self,
        input: str,
        *,
        system: str = "You are a helpful assistant.",
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Generator[str, None, None]:
        """产出文本增量；should_stop() 为 True 时应尽快结束。"""


class DeepSeekAdapter(ModelAdapter):
    """DeepSeek（OpenAI 兼容 Chat Completions）适配器。"""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
    ) -> None:
        key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("请先设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY，或传入 api_key")

        self._model = model
        self._max_tokens = max_tokens
        self._client = OpenAI(api_key=key, base_url=base_url)

    @property
    def name(self) -> str:
        return self._model

    def stream_chunks(
        self,
        input: str,
        *,
        system: str = "You are a helpful assistant.",
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Generator[str, None, None]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": input},
            ],
            stream=True,
            max_tokens=self._max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )

        for chunk in response:
            if should_stop and should_stop():
                break
            delta = chunk.choices[0].delta if chunk.choices else None
            text = getattr(delta, "content", None) if delta else None
            if text:
                yield text


class EchoAdapter(ModelAdapter):
    """本地回声适配器，便于离线测试。"""

    def __init__(self, *, model: str = "echo") -> None:
        self._model = model

    @property
    def name(self) -> str:
        return self._model

    def stream_chunks(
        self,
        input: str,
        *,
        system: str = "You are a helpful assistant.",
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Generator[str, None, None]:
        for ch in f"echo:{input}":
            if should_stop and should_stop():
                break
            yield ch


# ---------------------------------------------------------------------------
# StreamingModel：对接 Adapter 的统一模型门面
# ---------------------------------------------------------------------------


class StreamingModel:
    """流式模型门面：对网关暴露统一 stream 接口，内部委托 ModelAdapter。"""

    def __init__(self, adapter: ModelAdapter) -> None:
        self._adapter = adapter

    @classmethod
    def from_adapter(cls, adapter: ModelAdapter) -> "StreamingModel":
        return cls(adapter)

    @classmethod
    def deepseek(
        cls,
        *,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
    ) -> "StreamingModel":
        return cls(
            DeepSeekAdapter(
                api_key=api_key,
                base_url=base_url,
                model=model,
                max_tokens=max_tokens,
            )
        )

    @property
    def name(self) -> str:
        return self._adapter.name

    @property
    def adapter(self) -> ModelAdapter:
        return self._adapter

    def stream(
        self,
        input: str,
        *,
        system: str = "You are a helpful assistant.",
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Generator[str, None, None]:
        yield from self._adapter.stream_chunks(
            input,
            system=system,
            should_stop=should_stop,
        )


# 兼容旧名称
DeepSeekStreamingModel = StreamingModel.deepseek


# ---------------------------------------------------------------------------
# RunStore：Run 状态存储
# ---------------------------------------------------------------------------

RunStatus = Literal["created", "running", "completed", "stopped", "failed"]
_TERMINAL_STATUSES = frozenset({"completed", "stopped", "failed"})


@dataclass
class RunRecord:
    """一次 Agent Run 的持久化视图。"""

    run_id: str
    input: str
    model: str
    status: RunStatus = "created"
    output: str = ""
    error: Optional[str] = None
    stop_requested: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class RunStore(ABC):
    """Run 状态存储抽象。"""

    @abstractmethod
    def create(self, record: RunRecord) -> RunRecord:
        """写入一条新 Run；若 run_id 已存在则抛错。"""

    @abstractmethod
    def get(self, run_id: str) -> Optional[RunRecord]:
        """按 id 读取；不存在返回 None。返回值为快照副本。"""

    @abstractmethod
    def save(self, record: RunRecord) -> RunRecord:
        """覆盖写入已有 Run；不存在则抛错。"""

    @abstractmethod
    def delete(self, run_id: str) -> bool:
        """删除；存在返回 True，否则 False。"""

    @abstractmethod
    def list(self) -> list[RunRecord]:
        """列出全部 Run 快照。"""

    def exists(self, run_id: str) -> bool:
        return self.get(run_id) is not None

    def update(
        self,
        run_id: str,
        **fields: object,
    ) -> Optional[RunRecord]:
        """部分更新字段；不存在返回 None。"""
        record = self.get(run_id)
        if record is None:
            return None
        for key, value in fields.items():
            if not hasattr(record, key):
                raise AttributeError(f"RunRecord 无字段: {key}")
            setattr(record, key, value)
        record.updated_at = time.time()
        return self.save(record)

    def request_stop(self, run_id: str) -> Optional[RunRecord]:
        """标记停止请求；已终态则原样返回快照。"""
        record = self.get(run_id)
        if record is None:
            return None
        if record.status in _TERMINAL_STATUSES:
            return record
        return self.update(run_id, stop_requested=True, status="stopped")


class InMemoryRunStore(RunStore):
    """线程安全的内存 RunStore。"""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.RLock()

    def create(self, record: RunRecord) -> RunRecord:
        with self._lock:
            if record.run_id in self._runs:
                raise ValueError(f"run 已存在: {record.run_id}")
            stored = self._copy(record)
            self._runs[record.run_id] = stored
            return self._copy(stored)

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            record = self._runs.get(run_id)
            return None if record is None else self._copy(record)

    def save(self, record: RunRecord) -> RunRecord:
        with self._lock:
            if record.run_id not in self._runs:
                raise KeyError(f"run 不存在: {record.run_id}")
            record.updated_at = time.time()
            stored = self._copy(record)
            self._runs[record.run_id] = stored
            return self._copy(stored)

    def delete(self, run_id: str) -> bool:
        with self._lock:
            return self._runs.pop(run_id, None) is not None

    def list(self) -> list[RunRecord]:
        with self._lock:
            return [self._copy(r) for r in self._runs.values()]

    @staticmethod
    def _copy(record: RunRecord) -> RunRecord:
        return RunRecord(
            run_id=record.run_id,
            input=record.input,
            model=record.model,
            status=record.status,
            output=record.output,
            error=record.error,
            stop_requested=record.stop_requested,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


# ---------------------------------------------------------------------------
# 订阅 / 取消订阅接口
# ---------------------------------------------------------------------------

EventHandler = Callable[[RunEvent], None]


@dataclass
class Subscription:
    """一次事件订阅；调用 cancel() 或经 bus.unsubscribe 取消。"""

    subscription_id: str
    run_id: Optional[str]
    handler: EventHandler
    created_at: float = field(default_factory=time.time)
    cancelled: bool = False
    _bus: Optional["RunEventBus"] = field(default=None, repr=False, compare=False)

    def cancel(self) -> bool:
        """取消本订阅；已取消返回 False。"""
        if self._bus is None:
            if self.cancelled:
                return False
            self.cancelled = True
            return True
        return self._bus.unsubscribe(self.subscription_id)


class RunEventBus:
    """RunEvent 发布/订阅总线：支持按 run_id 过滤与取消订阅。"""

    def __init__(self) -> None:
        self._subs: dict[str, Subscription] = {}
        self._lock = threading.RLock()

    def subscribe(
        self,
        handler: EventHandler,
        *,
        run_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
    ) -> Subscription:
        """订阅事件。run_id=None 表示订阅全部 Run。"""
        sub_id = subscription_id or f"sub_{uuid.uuid4().hex[:12]}"
        sub = Subscription(
            subscription_id=sub_id,
            run_id=run_id,
            handler=handler,
            _bus=self,
        )
        with self._lock:
            if sub_id in self._subs and not self._subs[sub_id].cancelled:
                raise ValueError(f"subscription 已存在: {sub_id}")
            self._subs[sub_id] = sub
        return sub

    def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅；成功返回 True，不存在或已取消返回 False。"""
        with self._lock:
            sub = self._subs.get(subscription_id)
            if sub is None or sub.cancelled:
                return False
            sub.cancelled = True
            del self._subs[subscription_id]
            return True

    def cancel(self, subscription_id: str) -> bool:
        """unsubscribe 的别名。"""
        return self.unsubscribe(subscription_id)

    def cancel_all(self, *, run_id: Optional[str] = None) -> int:
        """取消匹配的订阅；run_id=None 时取消全部。返回取消数量。"""
        with self._lock:
            targets = [
                sid
                for sid, sub in self._subs.items()
                if run_id is None or sub.run_id == run_id
            ]
        return sum(1 for sid in targets if self.unsubscribe(sid))

    def publish(self, event: RunEvent) -> int:
        """向匹配订阅推送事件，返回成功投递数。"""
        event_run_id = getattr(event, "run_id", None)
        with self._lock:
            targets = [
                sub
                for sub in self._subs.values()
                if not sub.cancelled and (sub.run_id is None or sub.run_id == event_run_id)
            ]
        delivered = 0
        for sub in targets:
            if sub.cancelled:
                continue
            try:
                sub.handler(event)
                delivered += 1
            except Exception:  # noqa: BLE001 — 单个订阅失败不影响其他订阅
                continue
        return delivered

    def list_subscriptions(self, *, run_id: Optional[str] = None) -> list[Subscription]:
        with self._lock:
            return [
                sub
                for sub in self._subs.values()
                if not sub.cancelled and (run_id is None or sub.run_id == run_id)
            ]


# ---------------------------------------------------------------------------
# AgentLoop：单次 Run 的执行循环
# ---------------------------------------------------------------------------


class AgentLoop:
    """驱动一次 Run 的核心循环：读 Store → 调 StreamingModel → 写回 Store。

    当前为单步生成循环（max_steps=1）；后续可在 step 内扩展工具调用。
    """

    def __init__(
        self,
        model: StreamingModel,
        store: RunStore,
        *,
        bus: Optional[RunEventBus] = None,
        max_steps: int = 1,
        emit_deltas: bool = False,
    ) -> None:
        """emit_deltas=False（默认）：只在状态变化时发 run.get（running → completed）。

        流式过程仍会写入 RunStore.output，前端/轮询可看到增量；
        设为 True 时每个 token 都会再发一条 run.get（容易刷屏）。
        """
        if max_steps < 1:
            raise ValueError("max_steps 必须 >= 1")
        self.model = model
        self.store = store
        self.bus = bus
        self.max_steps = max_steps
        self.emit_deltas = emit_deltas

    def run(
        self,
        run_id: str,
        *,
        system: str = "You are a helpful assistant.",
    ) -> Optional[RunRecord]:
        """同步跑完整个循环，返回终态快照。"""
        for _ in self.iter_events(run_id, system=system):
            pass
        return self.store.get(run_id)

    def iter_events(
        self,
        run_id: str,
        *,
        system: str = "You are a helpful assistant.",
    ) -> Generator[RunEvent, None, None]:
        """执行循环并产出状态事件（以 RunGetEvent 反映进度）。"""
        record = self.store.get(run_id)
        if record is None:
            yield self._emit(RunGetEvent(run_id=run_id, status="not_found"))
            return

        if record.stop_requested or record.status in _TERMINAL_STATUSES:
            if record.stop_requested and record.status not in _TERMINAL_STATUSES:
                record = self.store.update(run_id, status="stopped") or record
            yield self._emit(self._to_get_event(record))
            return

        record = self.store.update(run_id, status="running") or record
        yield self._emit(self._to_get_event(record))

        try:
            output_parts: list[str] = [record.output] if record.output else []
            for step in range(self.max_steps):
                if self._should_stop(run_id):
                    break

                prompt = self._build_step_input(record.input, output_parts, step)
                for delta in self.model.stream(
                    prompt,
                    system=system,
                    should_stop=lambda: self._should_stop(run_id),
                ):
                    if self._should_stop(run_id):
                        break
                    output_parts.append(delta)
                    updated = self.store.update(run_id, output="".join(output_parts))
                    if self.emit_deltas and updated is not None:
                        yield self._emit(self._to_get_event(updated))

                # 单步场景下生成完毕即结束；多步预留扩展点
                if self.max_steps == 1:
                    break

            final = self.store.get(run_id)
            if final is None:
                yield self._emit(RunGetEvent(run_id=run_id, status="not_found"))
                return

            status: RunStatus = "stopped" if final.stop_requested else "completed"
            final = self.store.update(
                run_id,
                output="".join(output_parts),
                status=status,
            ) or final
            yield self._emit(self._to_get_event(final))
        except Exception as exc:  # noqa: BLE001 — 循环需要把失败写回 store
            failed = self.store.update(run_id, error=str(exc), status="failed")
            if failed is not None:
                yield self._emit(self._to_get_event(failed))
            else:
                yield self._emit(RunGetEvent(run_id=run_id, status="failed", error=str(exc)))

    def _emit(self, event: RunEvent) -> RunEvent:
        if self.bus is not None:
            self.bus.publish(event)
        return event

    def _should_stop(self, run_id: str) -> bool:
        current = self.store.get(run_id)
        return current is None or current.stop_requested

    @staticmethod
    def _build_step_input(user_input: str, output_parts: list[str], step: int) -> str:
        if step == 0 or not output_parts:
            return user_input
        # 多步时把已有输出作为上下文续写（工具循环可替换此逻辑）
        so_far = "".join(output_parts)
        return f"{user_input}\n\n[previous_output]\n{so_far}\n\n[continue]"

    @staticmethod
    def _to_get_event(record: RunRecord) -> RunGetEvent:
        return RunGetEvent(
            run_id=record.run_id,
            status=record.status,
            output=record.output or None,
            error=record.error,
        )


# ---------------------------------------------------------------------------
# StreamingAgentGateway
# ---------------------------------------------------------------------------


class StreamingAgentGateway:
    """基于 StreamingModel / ModelAdapter / RunStore / AgentLoop 的流式 Agent 网关。

    提供 create / get / stop 以及 subscribe / unsubscribe 订阅取消接口。
    """

    def __init__(
        self,
        model: Optional[StreamingModel] = None,
        *,
        adapter: Optional[ModelAdapter] = None,
        store: Optional[RunStore] = None,
        loop: Optional[AgentLoop] = None,
        bus: Optional[RunEventBus] = None,
    ) -> None:
        if model is not None and adapter is not None:
            raise ValueError("model 与 adapter 只能传其一")
        if adapter is not None:
            model = StreamingModel.from_adapter(adapter)
        self.model = model or StreamingModel.deepseek()
        self.store = store or InMemoryRunStore()
        self.bus = bus or RunEventBus()
        self.loop = loop or AgentLoop(
            self.model,
            self.store,
            bus=self.bus,
            emit_deltas=False,
        )
        if self.loop.bus is None:
            self.loop.bus = self.bus

    def create(
        self,
        input: str,
        *,
        system: str = "You are a helpful assistant.",
        background: bool = True,
    ) -> RunCreateEvent:
        """创建一次 Run；默认后台由 AgentLoop 执行。"""
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        record = self.store.create(
            RunRecord(run_id=run_id, input=input, model=self.model.name)
        )
        created = RunCreateEvent(
            run_id=record.run_id,
            input=record.input,
            model=record.model,
            created_at=record.created_at,
        )
        self.bus.publish(created)

        if background:
            thread = threading.Thread(
                target=self.loop.run,
                kwargs={"run_id": run_id, "system": system},
                daemon=True,
                name=f"run-{run_id}",
            )
            thread.start()
        else:
            self.loop.run(run_id, system=system)

        return created

    def get(self, run_id: str) -> RunGetEvent:
        """获取指定 Run 的当前状态。"""
        record = self.store.get(run_id)
        if record is None:
            return RunGetEvent(run_id=run_id, status="not_found")

        return RunGetEvent(
            run_id=record.run_id,
            status=record.status,
            output=record.output or None,
            error=record.error,
        )

    def stop(self, run_id: str, *, reason: Optional[str] = "client_stop") -> RunStopEvent:
        """请求停止指定 Run。"""
        record = self.store.get(run_id)
        if record is None:
            return RunStopEvent(run_id=run_id, status="not_found", reason=reason)

        if record.status in _TERMINAL_STATUSES:
            return RunStopEvent(
                run_id=run_id,
                status="already_terminal",
                reason=reason,
            )

        self.store.request_stop(run_id)
        stopped = RunStopEvent(run_id=run_id, status="stopping", reason=reason)
        self.bus.publish(stopped)
        return stopped

    def subscribe(
        self,
        handler: EventHandler,
        *,
        run_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
    ) -> Subscription:
        """订阅 Run 事件；可指定 run_id，不传则订阅全部。"""
        return self.bus.subscribe(
            handler,
            run_id=run_id,
            subscription_id=subscription_id,
        )

    def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅。"""
        return self.bus.unsubscribe(subscription_id)

    def cancel_subscription(self, subscription_id: str) -> bool:
        """取消订阅（unsubscribe 别名）。"""
        return self.bus.cancel(subscription_id)

    def cancel_subscriptions(self, *, run_id: Optional[str] = None) -> int:
        """批量取消订阅；可按 run_id 过滤。"""
        return self.bus.cancel_all(run_id=run_id)

    def stream_events(
        self,
        run_id: str,
        *,
        poll_interval: float = 0.2,
    ) -> Generator[RunEvent, None, None]:
        """轮询并产出该 Run 相关的事件序列（获取 → … → 终态获取）。"""
        create_like = self.get(run_id)
        if create_like.status == "not_found":
            yield create_like
            return

        # 首次获取
        yield create_like

        while True:
            time.sleep(poll_interval)
            event = self.get(run_id)
            yield event
            if event.status in {"completed", "stopped", "failed", "not_found"}:
                break

    def stream_sse(
        self,
        run_id: str,
        *,
        poll_interval: float = 0.2,
    ) -> Generator[str, None, None]:
        """以 SSE 帧形式流式产出事件（已编码文本）。"""
        for index, event in enumerate(self.stream_events(run_id, poll_interval=poll_interval)):
            yield encode_sse(event, event_id=str(index))

    def run_loop(
        self,
        run_id: str,
        *,
        system: str = "You are a helpful assistant.",
    ) -> Generator[RunEvent, None, None]:
        """直接驱动 AgentLoop 并实时产出事件（前台执行，不轮询）。"""
        yield from self.loop.iter_events(run_id, system=system)


if __name__ == "__main__":
    # SSE 编解码自检（不依赖网络）
    sample = RunCreateEvent(run_id="run_demo", input="hello")
    frame = encode_sse(sample, event_id="0")
    print("ENCODED:\n", frame, sep="")
    restored = decode_sse(frame)
    print("DECODED:", restored)
    assert restored == sample

    partial = SSEDecoder()
    mid = frame[: len(frame) // 2]
    assert partial.feed(mid) == []
    assert partial.feed(frame[len(frame) // 2 :]) == [sample]
    print("SSE incremental decode OK")

    # 订阅 / 取消 + AgentLoop（Echo 离线验证）
    received: list[RunEvent] = []
    gateway = StreamingAgentGateway(adapter=EchoAdapter())
    sub = gateway.subscribe(lambda e: received.append(e))
    created = gateway.create("hi", background=False)
    print("CREATE:", created)
    print("subscribed events:", len(received), "last=", received[-1])
    assert gateway.unsubscribe(sub.subscription_id) is True
    assert gateway.cancel_subscription(sub.subscription_id) is False
    print("unsubscribe OK")
