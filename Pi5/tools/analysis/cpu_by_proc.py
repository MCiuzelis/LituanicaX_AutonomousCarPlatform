"""CPU per process and per thread over N seconds, from /proc (on the Pi; no pidstat needed).
2026-10-11: dashboard ~95 % (detection 120 fps), +190 % while recording 1536x864@60 (x264), drive server 3 %,
map worker 8-15 %, BNO085 worker 10-30 %.

    python3 tools/analysis/cpu_by_proc.py 8
"""
import os, time, sys
hz = os.sysconf("SC_CLK_TCK")
def snap():
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit(): continue
        try:
            for tid in os.listdir("/proc/%s/task" % pid):
                st = open("/proc/%s/task/%s/stat" % (pid, tid)).read()
                name = st[st.index("(")+1:st.rindex(")")]
                f = st[st.rindex(")")+2:].split()
                cmd = open("/proc/%s/cmdline" % pid).read().replace("\0", " ")[:60] or name
                out[(pid, tid)] = (cmd, name, int(f[11]) + int(f[12]))
        except OSError: pass
    return out
a = snap(); time.sleep(float(sys.argv[1])); b = snap()
proc, thr = {}, {}
for k, (cmd, name, t) in b.items():
    if k in a:
        d = (t - a[k][2]) / hz / float(sys.argv[1]) * 100
        proc[cmd] = proc.get(cmd, 0) + d
        thr[(cmd[:30], name)] = thr.get((cmd[:30], name), 0) + d
for c, v in sorted(proc.items(), key=lambda x: -x[1])[:8]: print("%6.1f%%  %s" % (v, c))
print("-- threads")
for c, v in sorted(thr.items(), key=lambda x: -x[1])[:12]: print("%6.1f%%  %s / %s" % (v, c[0], c[1]))
