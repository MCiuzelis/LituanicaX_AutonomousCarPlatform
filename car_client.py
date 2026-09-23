#!/usr/bin/env python3
"""PC-side remote control for the RC car: keyboard -> tcp link over WiFi.

    python car_client.py --host 192.168.88.126        (Windows, macOS, Linux; standard library only)

Health lines from the Pi (battery, speed, active link, dead-man state) are shown live.

Keys (window must have focus):  W forward   S brake   A / D steer   space stop
                                + / - throttle level (5 % steps, max 20 %)   Esc quit
"""
import argparse
import json
import socket
import threading
import time
import tkinter as tk

RATE_HZ = 20


class Link(threading.Thread):
    """Reconnecting line link. send() queues the latest command; health lines land in .health."""

    def __init__(self, name, connect):
        super().__init__(daemon=True)
        self.name, self._connect = name, connect
        self.sock, self.health, self.last_rx, self.error = None, None, 0.0, ""
        self._cmd = None
        self.start()

    def send(self, line):
        self._cmd = line

    def run(self):
        while True:
            try:
                self.sock = self._connect()
                self.sock.settimeout(0.05)
                self.error, buf, nxt = "", b"", time.monotonic()
                while True:
                    if time.monotonic() >= nxt:
                        nxt += 1.0 / RATE_HZ
                        if self._cmd:
                            self.sock.sendall(self._cmd)
                    try:
                        chunk = self.sock.recv(4096)
                        if not chunk:
                            raise OSError("closed by peer")
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            try:
                                self.health, self.last_rx = json.loads(line), time.monotonic()
                            except ValueError:
                                pass
                    except socket.timeout:
                        pass
            except OSError as e:
                self.error = str(e).splitlines()[0][:60]
                if self.sock:
                    self.sock.close()
                self.sock = None
                time.sleep(1.0)

    @property
    def up(self):
        return self.sock is not None and time.monotonic() - self.last_rx < 1.5


def tcp_connect(host, port):
    def f():
        s = socket.create_connection((host, port), timeout=3)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return s
    return f


class App:
    def __init__(self, links, level):
        self.links, self.level, self.keys, self._pressed = links, level, set(), {}
        self.root = tk.Tk()
        self.root.title("RC car client")
        self.root.geometry("460x300")
        self.text = tk.Label(self.root, font=("Consolas", 12), justify="left", anchor="nw")
        self.text.pack(fill="both", expand=True, padx=10, pady=10)
        self.root.bind("<KeyPress>", self.press)
        self.root.bind("<KeyRelease>", self.release)
        self.root.bind("<FocusOut>", lambda e: self.keys.clear())
        self.root.after(int(1000 / RATE_HZ), self.tick)

    def press(self, e):
        k = e.keysym.lower()
        if k == "escape":
            self.root.destroy()
        elif k in ("plus", "equal"):
            self.level = min(0.20, self.level + 0.05)
        elif k in ("minus", "underscore"):
            self.level = max(0.05, self.level - 0.05)
        else:
            self.keys.add(k)
            self._pressed[k] = time.monotonic()

    def release(self, e):
        # key auto-repeat can emit release/press pairs: drop the key only if no press follows
        k, t = e.keysym.lower(), time.monotonic()
        self.root.after(40, lambda: self.keys.discard(k) if self._pressed.get(k, 0) <= t else None)

    def command(self):
        t = s = 0.0
        if "w" in self.keys:
            t = self.level
        if "s" in self.keys:
            t = -1.0
        if "a" in self.keys:
            s -= 1.0
        if "d" in self.keys:
            s += 1.0
        if "space" in self.keys:
            t = s = 0.0
        return t, s

    def tick(self):
        t, s = self.command()
        line = (json.dumps({"t": t, "s": s}) + "\n").encode()
        for l in self.links:
            l.send(line)
        h = max(self.links, key=lambda l: l.last_rx).health or {}
        lines = ["cmd  throttle %+.0f %%   steer %+.1f   (level %.0f %%, +/- to change)" % (t * 100, s, self.level * 100), ""]
        for l in self.links:
            lines.append("%-4s %s %s" % (l.name, "UP  " if l.up else "DOWN", l.error if not l.up else ""))
        lines += ["", "car  batt %s V  temp %s C  %s rev/s  pulse %s us  steer %s" % (
            h.get("v"), h.get("temp"), h.get("rps"), h.get("us"), h.get("st")),
            "     active link: %s   %s" % (h.get("act"), "STOPPED (dead-man)" if h.get("stop") else "driving"),
            "     pi sees cmds: %s" % h.get("lk"), "", "W fwd  S brake  A/D steer  space stop  Esc quit"]
        self.text.config(text="\n".join(lines))
        self.root.after(int(1000 / RATE_HZ), self.tick)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="rp5", help="Pi hostname/IP for the WiFi link")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--level", type=float, default=0.10, help="forward throttle for W (0.05..0.20)")
    a = ap.parse_args()
    links = [Link("wifi", tcp_connect(a.host, a.port))]
    App(links, min(0.20, max(0.05, a.level))).root.mainloop()


if __name__ == "__main__":
    main()
