import socket

import pytest

from real.wsg50_gripper import (
    WSG50CommandError,
    WSG50Gripper,
    WSG50ProtocolError,
)


class FakeSocket:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
        self.connected_to = None
        self.timeout = None
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, address):
        self.connected_to = address

    def sendall(self, data):
        self.sent.append(data)

    def recv(self, size):
        if not self.responses:
            raise socket.timeout()
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


def _factory(fake_socket):
    def make_socket(*args, **kwargs):
        return fake_socket

    return make_socket


def test_move_sends_gcl_command_and_waits_for_ack_fin():
    fake = FakeSocket([b"ACK MOVE\nFIN MOVE\n"])
    gripper = WSG50Gripper(
        host="10.0.0.2",
        port=1000,
        socket_factory=_factory(fake),
    )

    gripper.connect()
    lines = gripper.move(80.0, 50.0)

    assert fake.connected_to == ("10.0.0.2", 1000)
    assert fake.sent == [b"MOVE(80, 50)\n"]
    assert lines == ["ACK MOVE", "FIN MOVE"]


def test_grip_sends_force_width_and_speed():
    fake = FakeSocket([b"ACK GRIP\n", b"FIN GRIP\n"])
    gripper = WSG50Gripper(socket_factory=_factory(fake))

    gripper.connect()
    gripper.grip(force_n=40.0, width_mm=70.0, speed_mm_s=30.0)

    assert fake.sent == [b"GRIP(40, 70, 30)\n"]


def test_grip_can_send_force_only():
    fake = FakeSocket([b"ACK GRIP\n", b"FIN GRIP\n"])
    gripper = WSG50Gripper(socket_factory=_factory(fake))

    gripper.connect()
    gripper.grip(force_n=40.0)

    assert fake.sent == [b"GRIP(40)\n"]


def test_command_raises_on_err_response():
    fake = FakeSocket([b"ACK HOME\nERR HOME FAULT\n"])
    gripper = WSG50Gripper(socket_factory=_factory(fake))

    gripper.connect()

    with pytest.raises(WSG50CommandError):
        gripper.home()


def test_stop_returns_after_ack_without_fin():
    fake = FakeSocket([b"ACK STOP\n"])
    gripper = WSG50Gripper(socket_factory=_factory(fake))

    gripper.connect()

    assert gripper.stop() == ["ACK STOP"]
    assert fake.sent == [b"STOP()\n"]


def test_motion_command_times_out_cleanly_waiting_for_fin():
    fake = FakeSocket([b"ACK MOVE\n"])
    gripper = WSG50Gripper(socket_factory=_factory(fake))

    gripper.connect()

    with pytest.raises(WSG50ProtocolError, match="Timed out"):
        gripper.move(10.0)


def test_fast_stop_ack_uses_fsack_ack_only_command():
    fake = FakeSocket([b"ACK FSACK\n"])
    gripper = WSG50Gripper(socket_factory=_factory(fake))

    gripper.connect()

    assert gripper.fast_stop_ack() == ["ACK FSACK"]
    assert fake.sent == [b"FSACK()\n"]


def test_set_verbose_uses_assignment_response():
    fake = FakeSocket([b"VERBOSE=1\n"])
    gripper = WSG50Gripper(socket_factory=_factory(fake))

    gripper.connect()

    assert gripper.set_verbose(True) == "VERBOSE=1"
    assert fake.sent == [b"VERBOSE=1\n"]
