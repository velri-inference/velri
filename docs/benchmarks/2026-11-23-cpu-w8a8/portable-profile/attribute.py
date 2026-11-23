"""Attribute sampled Release IPs using the matching DWARF object."""
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from pathlib import Path
import re
from statistics import median
import subprocess
import sys

root = Path(sys.argv[1])
kernels = ('velri_slang_avx512_linear_w8a8_f32', 'velri_slang_avx512_linear_w8a8_f32_packed')
categories = ('GEMM_F32', 'W_expand_pack', 'X_expand_pack', 'I32_accumulate', 'epilogue',
              'GEMV_I8', 'kernel_other', 'pool_spin_sync', 'dispatch_other', 'DSO')

def lines(*command):
    return subprocess.check_output(command, text=True).splitlines()

def symbols(file):
    table = {}
    for row in lines('nm', '-S', '-n', root/file):
        fields = row.split()
        if len(fields) == 4 and fields[2] in ('T', 't'):
            table[fields[3]] = (int(fields[0], 16), int(fields[1], 16))
    return table

prod = symbols('graph')
debug = symbols('dwarf.o')
ordered = sorted((address, size, sym) for sym, (address, size) in prod.items())
starts = [address for address, _, _ in ordered]

def symbol(pc):
    index = bisect_right(starts, pc) - 1
    if index >= 0:
        address, size, name = ordered[index]
        if pc < address + size:
            return name, pc - address
    return 'plt/unknown', pc

def inner_loops(name):
    # Optimized DWARF omits a source line for some of the activation broadcasts.
    # Identify each branch-free inner FMA loop from its backward branch.
    instructions = []
    for line in lines('objdump', '-dw', '--no-show-raw-insn', f'--disassemble={name}', root/'graph'):
        match = re.match(r'\s*([0-9a-f]+):\s+(\w+)\s*(.*)', line)
        if match:
            instructions.append((int(match[1], 16), match[2], match[3]))
    ranges = []
    for address, op, operands in instructions:
        target = re.match(r'([0-9a-f]+) <', operands)
        if not op.startswith('j') or not target:
            continue
        start = int(target[1], 16)
        if not 0 < address - start < 4096:
            continue
        body = [op for pc, op, _ in instructions if start <= pc < address]
        if body and not any(op.startswith('j') for op in body):
            ranges.append((start - prod[name][0], address - prod[name][0], Counter(body)))
    return ranges

loops = {name: inner_loops(name) for name in kernels}
fma = {name: [(lo, hi) for lo, hi, body in loops[name] if any(op.startswith('vfmadd') for op in body)]
       for name in kernels}
samples = {}
headers = {}
addresses = set()
for file in sorted(root.glob('profile-*.ip')):
    data = file.read_text().splitlines()
    headers[file] = dict(re.findall(r'(\w+)=([\d.]+)', data[0]))
    counts = Counter()
    for row in data[1:]:
        obj, address, count = row.split()
        if obj == 'thread':
            continue
        name, offset = symbol(int(address, 16)) if obj == 'graph' else ('external', 0)
        counts[name, offset] += int(count)
        if name in kernels:
            addresses.add(debug[name][0] + offset)
    samples[file] = counts

frames = {}
for line in lines('addr2line', '-a', '-f', '-i', '-e', root/'dwarf.o', *map(hex, sorted(addresses))):
    if line.startswith('0x'):
        current = int(line, 16)
        frames[current] = []
    else:
        frames[current].append(line)
frames = {ip: list(zip(values[::2], values[1::2])) for ip, values in frames.items()}

# Resolve merge and epilogue scopes in the exact Slang source we profiled.
# The debugger's inlined microkernel call site cross-checks the snapshot.
source = (root/'linear_cpu.snapshot.slang').read_text().splitlines()
def find_line(marker, start=0):
    return next(i for i, line in enumerate(source, 1) if i > start and marker in line)
entry_line = find_line('public void linear_w8a8_f32_slice_cpu')
dot_call = find_line('gemm_mk_f32<', entry_line)
merge_begin = find_line('for (uint r =', dot_call)
merge_end = find_line('c_i32[dst] =', merge_begin)
ep_begin = find_line('for (uint rm =', find_line('k0 += kc', merge_end))
ep_end = find_line('EpQuantScale::apply(', ep_begin)
for chain in frames.values():
    for (child, _), (caller, location) in zip(chain, chain[1:]):
        if 'gemm_mk_f32' in child and 'linear_w8a8_f32_slice_cpu' in caller:
            line = re.search(r':(\d+)', location)
            if line and int(line[1]) != dot_call:
                raise ValueError('DWARF does not match linear_cpu.snapshot.slang')

def category(name, offset):
    if name == 'external':
        return 'DSO'
    if name not in kernels:
        if name in ('velri_os_cpu_relax', 'velri_os_cond_wait', 'worker_loop', 'velri_pool_parallel'):
            return 'pool_spin_sync'
        return 'dispatch_other'
    if any(lo <= offset <= hi for lo, hi in fma[name]):
        return 'GEMM_F32'
    for function, location in frames[debug[name][0] + offset]:
        for fragment, label in (('pack_b_f32', 'W_expand_pack'), ('unpack_w8a8_panel', 'W_expand_pack'),
                                ('pack_a_f32', 'X_expand_pack'), ('gemm_mk_f32', 'GEMM_F32'),
                                ('w8a8_gemv_columns', 'GEMV_I8'), ('EpQuantScale', 'epilogue')):
            if fragment in function:
                return label
        if 'linear_w8a8_f32_slice_cpu' in function:
            line = re.search(r':(\d+)', location)
            if line:
                position = int(line[1])
                if merge_begin <= position <= merge_end:
                    return 'I32_accumulate'
                if ep_begin <= position <= ep_end:
                    return 'epilogue'
    return 'kernel_other'

by_case = defaultdict(list)
hot = defaultdict(Counter)
with (root/'categories.csv').open('w') as file:
    table = csv.writer(file)
    table.writerow(['rep', 'rows', 'layout', 'samples', 'overruns', 'wall_s', 'user_s', 'sys_s', 'calls', *categories])
    for profile, hits in samples.items():
        _, rep, rows, layout = profile.stem.split('-')
        case = int(rows), layout
        counts = Counter()
        for (name, offset), count in hits.items():
            label = category(name, offset)
            counts[label] += count
            if name != 'external':
                hot[case][name, offset, label] += count
        h = headers[profile]
        calls = re.search(r'profiled (\d+) hot calls', profile.with_suffix('.log').read_text())[1]
        n = int(h['sample'])
        assert n == counts.total()
        by_case[case].append((n, counts))
        table.writerow([rep, rows, layout, n, h['overruns'], h['wall'], h['user'], h['system'], calls,
                        *(counts[label] for label in categories)])

with (root/'summary.csv').open('w') as file:
    table = csv.writer(file)
    table.writerow(['rows', 'layout', 'runs', 'samples', 'category', 'median_pct', 'min_pct', 'max_pct'])
    for (rows, layout), runs in sorted(by_case.items()):
        for label in categories:
            fractions = [100 * counts[label] / n for n, counts in runs]
            table.writerow([rows, layout, len(runs), sum(n for n, _ in runs), label,
                            *(f'{fn(fractions):.3f}' for fn in (median, min, max))])

with (root/'hot-ips.csv').open('w') as file:
    table = csv.writer(file)
    table.writerow(['rows', 'layout', 'symbol', 'offset', 'category', 'samples'])
    for (rows, layout), counts in sorted(hot.items()):
        for (name, offset, label), n in counts.most_common():
            table.writerow([rows, layout, name, hex(offset), label, n])

with (root/'hot-loops.csv').open('w') as file:
    table = csv.writer(file)
    table.writerow(['rows', 'layout', 'symbol', 'start_offset', 'back_edge_offset',
                    'loop_samples', 'all_samples', 'static_loop_body'])
    for (rows, layout), counts in sorted(hot.items()):
        name = kernels[int(layout == 'packed')]
        candidates = []
        for start, end, ops in loops[name]:
            n = sum(n for (sym, off, _), n in counts.items() if sym == name and start <= off <= end)
            candidates.append((n, start, end, ops))
        n, start, end, ops = max(candidates, key=lambda block: block[0])
        total = sum(n for n, _ in by_case[rows, layout])
        table.writerow([rows, layout, name, hex(start), hex(end), n, total,
                        ' '.join(f'{op}={n}' for op, n in ops.items())])
