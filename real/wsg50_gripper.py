import argparse
import socket
import sys
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from real.config import (  # noqa: E402
    WSG50_GRIP_FORCE_N,
    WSG50_GRIP_SPEED_MM_S,
    WSG50_IP,
    WSG50_OPEN_WIDTH_MM,
    WSG50_PORT,
    WSG50_RELEASE_DISTANCE_MM,
)


class WSG50Error(RuntimeError):
    pass


class WSG50ProtocolError(WSG50Error):
    pass


class WSG50CommandError(WSG50Error):
    def __init__(self, line: str):
        super().__init__(line)
        self.line = line
        self.command = ""
        self.code: int | None = None

        parts = line.split()
        if len(parts) >= 2:
            self.command = parts[1].upper()
        if len(parts) >= 3:
            try:
                self.code = int(parts[2])
            except ValueError:
                self.code = None


class WSG50Gripper:
    ENCODING = "ascii"

    def __init__(
        self,
        host: str = WSG50_IP,
        port: int = WSG50_PORT,
        timeout: float = 5.0,
        open_width_mm: float = WSG50_OPEN_WIDTH_MM,
        default_speed_mm_s: float = WSG50_GRIP_SPEED_MM_S,
        socket_factory: Callable[..., socket.socket] = socket.socket,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.open_width_mm = float(open_width_mm)
        self.default_speed_mm_s = float(default_speed_mm_s)
        self._socket_factory = socket_factory
        self._socket: Optional[socket.socket] = None
        self._recv_buffer = b""

    def __enter__(self) -> "WSG50Gripper":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        if self._socket is not None:
            return
        sock = self._socket_factory(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        self._socket = sock
        self._recv_buffer = b""

    def disconnect(self) -> None:
        if self._socket is None:
            return
        try:
            try:
                self.bye()
            except WSG50Error:
                pass
            self._socket.close()
        finally:
            self._socket = None
            self._recv_buffer = b""

    def home(self) -> list[str]:
        return self.command("HOME")

    def set_verbose(self, enabled: bool = True) -> str:
        return self.set_value("VERBOSE", 1 if enabled else 0)

    def move(self, width_mm: float, speed_mm_s: Optional[float] = None) -> list[str]:
        speed = self.default_speed_mm_s if speed_mm_s is None else float(speed_mm_s)
        return self.command("MOVE", float(width_mm), speed)

    def grip(
        self,
        force_n: float = WSG50_GRIP_FORCE_N,
        width_mm: Optional[float] = None,
        speed_mm_s: Optional[float] = None,
    ) -> list[str]:
        args: list[float] = [float(force_n)]
        if width_mm is not None:
            args.append(float(width_mm))
        if speed_mm_s is not None:
            if width_mm is None:
                raise ValueError("width_mm must be provided when speed_mm_s is provided")
            args.append(float(speed_mm_s))
        return self.command("GRIP", *args)

    def release(
        self,
        distance_mm: float = WSG50_RELEASE_DISTANCE_MM,
        speed_mm_s: Optional[float] = None,
    ) -> list[str]:
        args = [float(distance_mm)]
        if speed_mm_s is not None:
            args.append(float(speed_mm_s))
        return self.command("RELEASE", *args)

    def open(self, speed_mm_s: Optional[float] = None) -> list[str]:
        return self.move(self.open_width_mm, speed_mm_s)

    def close(
        self,
        force_n: float = WSG50_GRIP_FORCE_N,
        width_mm: Optional[float] = None,
        speed_mm_s: Optional[float] = None,
    ) -> list[str]:
        return self.grip(force_n=force_n, width_mm=width_mm, speed_mm_s=speed_mm_s)

    def stop(self) -> list[str]:
        return self.command("STOP", wait_for_fin=False)

    def fast_stop_ack(self) -> list[str]:
        return self.command("FSACK", wait_for_fin=False)

    def bye(self) -> list[str]:
        return self.command("BYE", wait_for_fin=False)

    def set_value(self, name: str, value: float | int) -> str:
        if self._socket is None:
            raise WSG50ProtocolError("WSG50 socket is not connected")

        command_name = name.upper()
        raw_command = f"{command_name}={self._format_arg(value)}\n"
        self._socket.sendall(raw_command.encode(self.ENCODING))
        line = self._readline()
        if line.upper().startswith("ERR "):
            raise WSG50CommandError(line)
        if not line.upper().startswith(f"{command_name}="):
            raise WSG50ProtocolError(f"Unexpected response for {command_name}: {line}")
        return line

    def command(self, name: str, *args: float, wait_for_fin: bool = True) -> list[str]:
        if self._socket is None:
            raise WSG50ProtocolError("WSG50 socket is not connected")

        command_name = name.upper()
        rendered_args = ", ".join(self._format_arg(arg) for arg in args)
        raw_command = f"{command_name}({rendered_args})\n"
        self._socket.sendall(raw_command.encode(self.ENCODING))

        lines: list[str] = []
        saw_ack = False
        while True:
            line = self._readline()
            lines.append(line)
            parts = line.split()
            if not parts:
                continue

            status = parts[0].upper()
            response_command = parts[1].upper() if len(parts) > 1 else ""
            if status == "ERR":
                raise WSG50CommandError(line)
            if response_command and response_command != command_name:
                raise WSG50ProtocolError(
                    f"Unexpected response for {command_name}: {line}"
                )
            if status == "ACK":
                saw_ack = True
                if not wait_for_fin:
                    return lines
                continue
            if status == "FIN":
                if not saw_ack:
                    raise WSG50ProtocolError(
                        f"Command {command_name} finished before ACK: {lines}"
                    )
                return lines

    def _readline(self) -> str:
        if self._socket is None:
            raise WSG50ProtocolError("WSG50 socket is not connected")

        while b"\n" not in self._recv_buffer:
            try:
                chunk = self._socket.recv(1024)
            except socket.timeout as exc:
                raise WSG50ProtocolError("Timed out waiting for WSG50 response") from exc
            if not chunk:
                raise WSG50ProtocolError("WSG50 socket closed while waiting for response")
            self._recv_buffer += chunk

        raw_line, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
        return raw_line.decode(self.ENCODING, errors="replace").strip()

    @staticmethod
    def _format_arg(value: float) -> str:
        return f"{float(value):g}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control a WSG50 gripper over TCP GCL.")
    parser.add_argument("--ip", default=WSG50_IP, help="WSG50 IP address.")
    parser.add_argument("--port", type=int, default=WSG50_PORT, help="WSG50 TCP port.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Socket timeout in seconds.")
    parser.add_argument("--home", action="store_true", help="Run HOME() before other commands.")
    parser.add_argument("--open", action="store_true", help="Open to configured width.")
    parser.add_argument("--close", action="store_true", help="Run GRIP() with configured defaults.")
    parser.add_argument("--grip-test", action="store_true", help="Open, grip, then release.")
    parser.add_argument("--open-width-mm", type=float, default=WSG50_OPEN_WIDTH_MM)
    parser.add_argument("--force-n", type=float, default=WSG50_GRIP_FORCE_N)
    parser.add_argument("--speed-mm-s", type=float, default=WSG50_GRIP_SPEED_MM_S)
    parser.add_argument("--release-distance-mm", type=float, default=WSG50_RELEASE_DISTANCE_MM)
    return parser.parse_args()


def _print_response(label: str, lines: list[str]) -> None:
    print(f"{label}: {' | '.join(lines)}")


def main() -> None:
    args = _parse_args()
    with WSG50Gripper(
        host=args.ip,
        port=args.port,
        timeout=args.timeout,
        open_width_mm=args.open_width_mm,
        default_speed_mm_s=args.speed_mm_s,
    ) as gripper:
        if args.home:
            _print_response("HOME", gripper.home())
        if args.open:
            _print_response("OPEN", gripper.open(args.speed_mm_s))
        if args.close:
            _print_response("GRIP", gripper.grip(args.force_n, speed_mm_s=None))
        if args.grip_test:
            _print_response("OPEN", gripper.open(args.speed_mm_s))
            _print_response("GRIP", gripper.grip(args.force_n, speed_mm_s=None))
            _print_response(
                "RELEASE",
                gripper.release(args.release_distance_mm, args.speed_mm_s),
            )


if __name__ == "__main__":
    main()
