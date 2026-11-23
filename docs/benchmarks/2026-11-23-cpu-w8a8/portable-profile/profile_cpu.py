"""Run the W8A8 CPU-clock profile from a native standalone Ninja Release/LLVM_IR build."""
import argparse
import csv
import fcntl
import hashlib
import os
from pathlib import Path
import random
import shlex
import subprocess

tools_dir = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build', required=True, type=Path)
parser.add_argument('--graph', required=True, type=Path, help='W8A8 benchmark graph.c')
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--cpus', default='0-15')
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--rows', type=int, choices=[1, 64, 128, 256], help='Select one row count')
parser.add_argument('--layout', choices=['raw','packed'], help='Select a weight layout')
args = parser.parse_args()
build, graph_source, out = args.build.resolve(), args.graph.resolve(), args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()):
    parser.error('--output must point to a new or empty directory')
if args.repeats < 1:
    parser.error('--repeats must be positive')

def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, text=True, **kwargs)

# Rebuild the exact production Slang object with DWARF to resolve inlined loops.
run(['ninja', '-C', build, 'libvelri.a'])
commands = run(['ninja', '-C', build, '-t', 'commands', 'libvelri.a'], capture_output=True).stdout
command = next(c for c in commands.splitlines() if '-DVELRI_ISA=avx512 ' in c and '-target llvm-ir' in c)
stages = [shlex.split(part.strip()) for part in command.split(' && ')]
slang, strip_ir, clang = stages[-3:]
entry = next(Path(x) for x in slang if x.endswith('entry_cpu.slang'))
lib_source = entry.parents[3]
old_raw = slang[slang.index('-o') + 1]
old_ir = clang[clang.index('-c') + 1]
old_object = Path(clang[clang.index('-o') + 1])
library = build / 'libvelri.a'
paths = {old_raw: str(out/'dwarf.raw.ll'), old_ir: str(out/'dwarf.ll'), str(old_object): str(out/'dwarf.o')}
for stage in [[*slang, '-g3'], strip_ir, clang]:
    arguments = []
    for argument in stage:
        for old_path, new_path in paths.items():
            argument = argument.replace(old_path, new_path)
        arguments.append(argument)
    run(arguments, cwd=build)
with (out/'release-member.o').open('wb') as member:
    subprocess.run(['ar', 'p', library, old_object.name], check=True, stdout=member)
for obj, target in [(out/'release-member.o', 'release.text'), (out/'dwarf.o', 'dwarf.text')]:
    run(['objcopy', '--dump-section', f'.text={out/target}', obj, os.devnull])
release_text = (out/'release.text').read_bytes()
if release_text != (out/'dwarf.text').read_bytes():
    raise RuntimeError('Release and DWARF code differ. Instruction attribution needs inspection.')
linear_source = entry.parent/'ops/linear_cpu.slang'
(out/'linear_cpu.snapshot.slang').write_bytes(linear_source.read_bytes())
paths_to_record = [entry, linear_source, graph_source, library, out/'release.text', out/'dwarf.text']
def checksums():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths_to_record}
before = checksums()
print('Matched .text SHA256', hashlib.sha256(release_text).hexdigest(), flush=True)

source = graph_source.read_text()
marker = '    double times[9];'
if source.count(marker) != 1: raise RuntimeError('Benchmark sampling hook moved')
(out/'graph-profile.c').write_text(source.replace(marker, '''    extern void profile_start(void), profile_stop(void);
    size_t calls = 0;
    double t = now_sec();
    profile_start();
    do {
        check_ok(velri_slot_run(slot, nullptr, nullptr), "sampled slot_run");
        ++calls;
    } while (now_sec() - t < 1.5);
    profile_stop();
    fprintf(stderr, "profiled %zu hot calls\\n", calls);
'''+marker))
common = [clang[0], '-std=c23', '-O3', '-no-pie', '-DVELRI_BENCH_PACKED_API',
          '-DVELRI_BENCH_FIXED_WARMUP', '-I', lib_source/'include', '-I', lib_source/'src/platform']
run([*common, out/'graph-profile.c', tools_dir/'sample_cpu.c', library, '-Wl,--wrap=pthread_create',
     '-lm', '-ldl', '-pthread', '-o', out/'graph'])
run([*common, graph_source, library, '-lm', '-ldl', '-pthread', '-o', out/'timing'])
cases = [(128, 2048, 2048, 1, 64), (64, 4096, 4096, 1, 32),
         (256, 4096, 4096, 1, 16), (1, 4096, 4096, 12, 48)]
labels = [(case, layout) for case in cases for layout in ('raw', 'packed')
          if (not args.rows or case[0] == args.rows) and (not args.layout or layout == args.layout)]
with (out/'timings.csv').open('w', buffering=1) as log:
    csvout = csv.writer(log)
    csvout.writerow(['rep','rows','m','k','ring','iterations','layout','tops','ms','effective_W_GB_s'])
    for mode in ('profile', 'time'):
        for rep in range(args.repeats):
            random.Random(1123 + rep).shuffle(labels)
            for (rows,m,k,ring,it), layout in labels:
                name = out/f'profile-{rep}-{rows}-{layout}'
                env = os.environ.copy()
                if mode == 'profile': env['VELRI_SAMPLE_FILE'] = str(name)+'.ip'
                exe = out/('graph' if mode == 'profile' else 'timing')
                command = ['taskset','-c',args.cpus,exe,rows,m,k,16,ring,it if mode=='time' else ring]
                if layout == 'packed': command.append('packed')
                with open('/tmp/velri-w8-perf.lock','a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    result = run(command, env=env, capture_output=True)
                if mode == 'profile':
                    name.with_suffix('.log').write_text(result.stderr)
                else:
                    record = result.stdout.rstrip().split(',')
                    csvout.writerow([rep,rows,m,k,ring,it,layout,record[11],record[8],record[12]])
if before != checksums():
    raise RuntimeError('Source or Release object changed during measurement')
paths_to_record.extend([out/'graph', out/'timing'])
(out/'SHA256SUMS').write_text(''.join(f'{digest}  {path}\n' for path, digest in checksums().items()))
run(['python3', tools_dir/'attribute.py', out])
