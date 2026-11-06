"""StreamingAgentGateway Web 端：浏览器可访问的本地页面 + SSE。"""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

try:
    from gateway import (
        DeepSeekAdapter,
        EchoAdapter,
        RunEvent,
        StreamingAgentGateway,
        encode_sse,
    )
except ImportError:  # python -m week01.web_app
    from week01.gateway import (
        DeepSeekAdapter,
        EchoAdapter,
        RunEvent,
        StreamingAgentGateway,
        encode_sse,
    )

HOST = os.getenv("GATEWAY_HOST", "127.0.0.1")
PORT = int(os.getenv("GATEWAY_PORT", "8765"))
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _load_dotenv() -> None:
    """从 week01/.env 加载 KEY（不覆盖已有环境变量）。"""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def build_gateway() -> StreamingAgentGateway:
    """默认走 DeepSeek；仅当 GATEWAY_ECHO=1 时才用本地 Echo。"""
    if os.getenv("GATEWAY_ECHO", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        print("[web] GATEWAY_ECHO=1，使用 EchoAdapter（仅本地回声）")
        return StreamingAgentGateway(adapter=EchoAdapter())

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "未设置 DEEPSEEK_API_KEY / OPENAI_API_KEY。\n"
            "请先: export DEEPSEEK_API_KEY='你的key'\n"
            "若仅需本地联调回声模式: export GATEWAY_ECHO=1"
        )

    print("[web] 使用 DeepSeekAdapter → https://api.deepseek.com")
    return StreamingAgentGateway(adapter=DeepSeekAdapter(api_key=api_key))


GATEWAY = build_gateway()


def _json_bytes(payload: object, status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return status, body, "application/json; charset=utf-8"


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "StreamingAgentGatewayWeb/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[web] {self.address_string()} - {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body 必须是对象")
        return data

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._serve_index()
            return
        if path == "/api/health":
            status, body, ctype = _json_bytes(
                {
                    "ok": True,
                    "model": GATEWAY.model.name,
                    "adapter": type(GATEWAY.model.adapter).__name__,
                }
            )
            self._send(status, body, ctype)
            return
        if path.startswith("/api/runs/") and path.endswith("/events"):
            run_id = path[len("/api/runs/") : -len("/events")]
            self._sse_events(run_id)
            return
        if path.startswith("/api/runs/"):
            run_id = path[len("/api/runs/") :]
            event = GATEWAY.get(run_id)
            status, body, ctype = _json_bytes(asdict(event))
            self._send(status, body, ctype)
            return
        self._send(404, b'{"error":"not_found"}', "application/json; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
        except (json.JSONDecodeError, ValueError) as exc:
            status, body, ctype = _json_bytes({"error": str(exc)}, 400)
            self._send(status, body, ctype)
            return

        if path == "/api/runs":
            text = str(payload.get("input") or "").strip()
            if not text:
                status, body, ctype = _json_bytes({"error": "input 不能为空"}, 400)
                self._send(status, body, ctype)
                return
            system = str(payload.get("system") or "You are a helpful assistant.")
            created = GATEWAY.create(text, system=system, background=True)
            status, body, ctype = _json_bytes(asdict(created), 201)
            self._send(status, body, ctype)
            return

        if path.startswith("/api/runs/") and path.endswith("/stop"):
            run_id = path[len("/api/runs/") : -len("/stop")]
            reason = payload.get("reason") or "web_stop"
            stopped = GATEWAY.stop(run_id, reason=str(reason))
            status, body, ctype = _json_bytes(asdict(stopped))
            self._send(status, body, ctype)
            return

        self._send(404, b'{"error":"not_found"}', "application/json; charset=utf-8")

    def _serve_index(self) -> None:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            self._send(500, b"index.html missing", "text/plain; charset=utf-8")
            return
        body = index.read_bytes()
        self._send(200, body, "text/html; charset=utf-8")

    def _sse_events(self, run_id: str) -> None:
        """订阅指定 Run，以 SSE 推送事件；客户端断开即取消订阅。"""
        q: queue.Queue = queue.Queue()
        closed = threading.Event()

        def handler(event: RunEvent) -> None:
            if closed.is_set():
                return
            q.put(event)

        sub = GATEWAY.subscribe(handler, run_id=run_id)
        # 仅补一帧快照；进行中的增量靠 store 轮询/终态事件，避免多个 running 刷屏
        snapshot = GATEWAY.get(run_id)
        q.put(snapshot)
        # 若已是终态，不再等待后续 bus 事件
        already_terminal = snapshot.status in {
            "completed",
            "stopped",
            "failed",
            "not_found",
        }

        self.close_connection = True  # 流结束后关闭，避免客户端一直挂起
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        seq = 0
        terminal_seen = already_terminal
        last_sent_status: Optional[str] = None
        last_sent_output: Optional[str] = None
        try:
            if already_terminal:
                try:
                    self.wfile.write(
                        encode_sse(snapshot, event_id=str(seq)).encode("utf-8")
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                return

            while not closed.is_set():
                try:
                    event = q.get(timeout=0.25)
                except queue.Empty:
                    if terminal_seen:
                        break
                    # 进行中：最多推送「状态变化」或「输出有更新」的一帧，合并刷屏
                    record = GATEWAY.store.get(run_id)
                    if record is None:
                        break
                    if record.status in {"completed", "stopped", "failed"}:
                        snap = GATEWAY.get(run_id)
                        try:
                            self.wfile.write(
                                encode_sse(snap, event_id=str(seq)).encode("utf-8")
                            )
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            break
                        break
                    if (
                        record.status != last_sent_status
                        or (record.output or None) != last_sent_output
                    ):
                        snap = GATEWAY.get(run_id)
                        # running 阶段：用最新快照覆盖，不堆多条
                        try:
                            self.wfile.write(
                                encode_sse(snap, event_id=str(seq)).encode("utf-8")
                            )
                            self.wfile.flush()
                            seq += 1
                            last_sent_status = snap.status
                            last_sent_output = snap.output
                        except (BrokenPipeError, ConnectionResetError):
                            break
                    else:
                        try:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            break
                    continue

                status = getattr(event, "status", None)
                output = getattr(event, "output", None)
                # 过滤连续重复的 running 帧
                if (
                    status == "running"
                    and last_sent_status == "running"
                    and output == last_sent_output
                ):
                    continue
                if status == "running" and last_sent_status == "running":
                    # 同为 running 但 output 变了：仍只发一帧最新（覆盖式）
                    pass

                frame = encode_sse(event, event_id=str(seq)).encode("utf-8")
                seq += 1
                try:
                    self.wfile.write(frame)
                    self.wfile.flush()
                    last_sent_status = status
                    last_sent_output = output
                except (BrokenPipeError, ConnectionResetError):
                    break

                if status in {"completed", "stopped", "failed", "not_found", "already_terminal"}:
                    terminal_seen = True
                    if not output:
                        snap = GATEWAY.get(run_id)
                        if getattr(snap, "output", None):
                            try:
                                self.wfile.write(
                                    encode_sse(snap, event_id=str(seq)).encode("utf-8")
                                )
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                break
                    break
        finally:
            closed.set()
            GATEWAY.unsubscribe(sub.subscription_id)


def main(host: str = HOST, port: int = PORT) -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), GatewayHandler)
    adapter_name = type(GATEWAY.model.adapter).__name__
    print(f"StreamingAgentGateway Web")
    print(f"  model   : {GATEWAY.model.name} ({adapter_name})")
    print(f"  open    : http://{host}:{port}/")
    print("  Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
