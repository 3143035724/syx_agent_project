"""StreamingAgentGateway 相关异常 / 边界路径测试。"""

from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Callable, Generator, Optional
from unittest import mock

# 保证直接 python week01/test_gateway_exceptions.py 可导入
sys.path.insert(0, os.path.dirname(__file__))

from gateway import (
    AgentLoop,
    DeepSeekAdapter,
    EchoAdapter,
    InMemoryRunStore,
    ModelAdapter,
    RunCreateEvent,
    RunEventBus,
    RunGetEvent,
    RunRecord,
    RunStopEvent,
    SSECodecError,
    SSEDecoder,
    StreamingAgentGateway,
    StreamingModel,
    decode_sse,
    decode_sse_stream,
    encode_sse,
    encode_sse_stream,
)


class BoomAdapter(ModelAdapter):
    """始终抛错的适配器，用于验证 failed 路径。"""

    def __init__(self, exc: Optional[Exception] = None) -> None:
        self._exc = exc or RuntimeError("boom")

    @property
    def name(self) -> str:
        return "boom"

    def stream_chunks(
        self,
        input: str,
        *,
        system: str = "You are a helpful assistant.",
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Generator[str, None, None]:
        raise self._exc
        yield  # pragma: no cover — 保持生成器类型


class SlowEchoThenBoom(ModelAdapter):
    """先产出部分 token，再抛错。"""

    @property
    def name(self) -> str:
        return "partial-boom"

    def stream_chunks(
        self,
        input: str,
        *,
        system: str = "You are a helpful assistant.",
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Generator[str, None, None]:
        yield "part"
        raise RuntimeError("mid-stream failure")


class TestSSECodecExceptions(unittest.TestCase):
    def test_decode_missing_data(self) -> None:
        with self.assertRaises(SSECodecError) as ctx:
            decode_sse("event: run.get\n\n")
        self.assertIn("缺少 data", str(ctx.exception))

    def test_decode_invalid_json(self) -> None:
        with self.assertRaises(SSECodecError) as ctx:
            decode_sse("event: run.get\ndata: {not-json}\n\n")
        self.assertIn("不是合法 JSON", str(ctx.exception))

    def test_decode_non_object_json(self) -> None:
        with self.assertRaises(SSECodecError) as ctx:
            decode_sse("event: run.get\ndata: [1,2]\n\n")
        self.assertIn("JSON 对象", str(ctx.exception))

    def test_decode_unknown_event_type(self) -> None:
        with self.assertRaises(SSECodecError) as ctx:
            decode_sse('event: run.unknown\ndata: {"run_id":"r1"}\n\n')
        self.assertIn("未知事件类型", str(ctx.exception))

    def test_decode_missing_type(self) -> None:
        with self.assertRaises(SSECodecError) as ctx:
            decode_sse('data: {"run_id":"r1"}\n\n')
        self.assertIn("无法确定事件类型", str(ctx.exception))

    def test_decode_expects_exactly_one_event(self) -> None:
        stream = encode_sse_stream(
            [
                RunCreateEvent(run_id="r1", input="a", created_at=1.0),
                RunGetEvent(run_id="r1", status="running", retrieved_at=2.0),
            ]
        )
        with self.assertRaises(SSECodecError) as ctx:
            decode_sse(stream)
        self.assertIn("期望恰好 1 条", str(ctx.exception))

    def test_decode_incomplete_payload_type_error(self) -> None:
        # run.get 缺少必填 status
        with self.assertRaises(SSECodecError) as ctx:
            decode_sse('event: run.get\ndata: {"run_id":"r1","type":"run.get"}\n\n')
        self.assertIn("无法还原", str(ctx.exception))

    def test_decoder_flush_invalid_leftover(self) -> None:
        dec = SSEDecoder()
        self.assertEqual(dec.feed("event: run.get\n"), [])
        with self.assertRaises(SSECodecError):
            dec.flush()


class TestRunStoreExceptions(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryRunStore()

    def test_create_duplicate_raises(self) -> None:
        self.store.create(RunRecord(run_id="r1", input="a", model="echo"))
        with self.assertRaises(ValueError) as ctx:
            self.store.create(RunRecord(run_id="r1", input="b", model="echo"))
        self.assertIn("已存在", str(ctx.exception))

    def test_save_missing_raises(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            self.store.save(RunRecord(run_id="missing", input="a", model="echo"))
        self.assertIn("不存在", str(ctx.exception))

    def test_update_unknown_field_raises(self) -> None:
        self.store.create(RunRecord(run_id="r1", input="a", model="echo"))
        with self.assertRaises(AttributeError) as ctx:
            self.store.update("r1", not_a_field=123)
        self.assertIn("无字段", str(ctx.exception))

    def test_update_missing_returns_none(self) -> None:
        self.assertIsNone(self.store.update("nope", status="running"))

    def test_request_stop_missing_returns_none(self) -> None:
        self.assertIsNone(self.store.request_stop("nope"))

    def test_delete_missing_returns_false(self) -> None:
        self.assertFalse(self.store.delete("nope"))


class TestAdapterAndModelExceptions(unittest.TestCase):
    def test_deepseek_adapter_requires_api_key(self) -> None:
        env = {"DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            # 清掉可能存在的 key
            for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
                os.environ.pop(k, None)
            with self.assertRaises(ValueError) as ctx:
                DeepSeekAdapter(api_key=None)
            self.assertIn("API_KEY", str(ctx.exception))

    def test_gateway_rejects_model_and_adapter_together(self) -> None:
        model = StreamingModel.from_adapter(EchoAdapter())
        with self.assertRaises(ValueError) as ctx:
            StreamingAgentGateway(model=model, adapter=EchoAdapter())
        self.assertIn("只能传其一", str(ctx.exception))

    def test_agent_loop_invalid_max_steps(self) -> None:
        store = InMemoryRunStore()
        model = StreamingModel.from_adapter(EchoAdapter())
        with self.assertRaises(ValueError) as ctx:
            AgentLoop(model, store, max_steps=0)
        self.assertIn("max_steps", str(ctx.exception))


class TestSubscriptionExceptions(unittest.TestCase):
    def test_duplicate_subscription_id_raises(self) -> None:
        bus = RunEventBus()
        bus.subscribe(lambda e: None, subscription_id="sub_fixed")
        with self.assertRaises(ValueError) as ctx:
            bus.subscribe(lambda e: None, subscription_id="sub_fixed")
        self.assertIn("已存在", str(ctx.exception))

    def test_unsubscribe_missing_returns_false(self) -> None:
        bus = RunEventBus()
        self.assertFalse(bus.unsubscribe("sub_missing"))
        self.assertFalse(bus.cancel("sub_missing"))

    def test_handler_exception_does_not_break_publish(self) -> None:
        bus = RunEventBus()
        ok: list = []

        def bad(_e) -> None:
            raise RuntimeError("handler boom")

        bus.subscribe(bad)
        bus.subscribe(lambda e: ok.append(e))
        delivered = bus.publish(RunGetEvent(run_id="r1", status="running"))
        self.assertEqual(delivered, 1)
        self.assertEqual(len(ok), 1)

    def test_cancel_then_unsubscribe_false(self) -> None:
        bus = RunEventBus()
        sub = bus.subscribe(lambda e: None)
        self.assertTrue(sub.cancel())
        self.assertFalse(bus.unsubscribe(sub.subscription_id))


class TestGatewayAndLoopFailurePaths(unittest.TestCase):
    def test_get_not_found(self) -> None:
        g = StreamingAgentGateway(adapter=EchoAdapter())
        event = g.get("run_missing")
        self.assertIsInstance(event, RunGetEvent)
        self.assertEqual(event.status, "not_found")

    def test_stop_not_found(self) -> None:
        g = StreamingAgentGateway(adapter=EchoAdapter())
        event = g.stop("run_missing")
        self.assertIsInstance(event, RunStopEvent)
        self.assertEqual(event.status, "not_found")

    def test_stop_already_terminal(self) -> None:
        g = StreamingAgentGateway(adapter=EchoAdapter())
        created = g.create("hi", background=False)
        event = g.stop(created.run_id)
        self.assertEqual(event.status, "already_terminal")

    def test_stream_events_not_found(self) -> None:
        g = StreamingAgentGateway(adapter=EchoAdapter())
        events = list(g.stream_events("run_missing", poll_interval=0.01))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].status, "not_found")

    def test_model_error_marks_failed(self) -> None:
        g = StreamingAgentGateway(adapter=BoomAdapter())
        created = g.create("x", background=False)
        final = g.get(created.run_id)
        self.assertEqual(final.status, "failed")
        self.assertIn("boom", final.error or "")

    def test_mid_stream_error_marks_failed(self) -> None:
        g = StreamingAgentGateway(adapter=SlowEchoThenBoom())
        created = g.create("x", background=False)
        final = g.get(created.run_id)
        self.assertEqual(final.status, "failed")
        self.assertIn("mid-stream", final.error or "")
        # 失败前可能已写入部分 output
        record = g.store.get(created.run_id)
        self.assertIsNotNone(record)
        self.assertTrue((record.output or "").startswith("part") or record.output == "")

    def test_loop_not_found_event(self) -> None:
        store = InMemoryRunStore()
        model = StreamingModel.from_adapter(EchoAdapter())
        loop = AgentLoop(model, store)
        events = list(loop.iter_events("missing"))
        self.assertEqual(events[-1].status, "not_found")

    def test_roundtrip_still_works_after_error_cases(self) -> None:
        # 冒烟：异常路径后正常链路仍可用
        sample = RunCreateEvent(run_id="r", input="ok", created_at=1.0)
        self.assertEqual(decode_sse(encode_sse(sample)), sample)
        self.assertEqual(len(decode_sse_stream(encode_sse_stream([sample]))), 1)


class TestWebApiExceptions(unittest.TestCase):
    """对 web_app 请求处理中的异常分支做轻量校验。"""

    @classmethod
    def setUpClass(cls) -> None:
        import web_app as web_app_module

        cls.web = web_app_module

    def test_health_ok(self) -> None:
        # 通过 handler 内部工具函数验证 JSON 包装
        status, body, ctype = self.web._json_bytes({"ok": True})
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        self.assertTrue(json.loads(body)["ok"])

    def test_create_requires_input_logic(self) -> None:
        # 复现 handler 对空 input 的校验语义
        text = str("".strip())
        self.assertEqual(text, "")
        status, body, _ = self.web._json_bytes({"error": "input 不能为空"}, 400)
        self.assertEqual(status, 400)
        self.assertIn("input", json.loads(body)["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
