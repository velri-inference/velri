"""Build the long-context harnesses against the measured engine libraries."""
import argparse
from pathlib import Path
import shlex
import subprocess as sp

p = argparse.ArgumentParser(description=__doc__)
for option in ('velri-cpu-build','velri-gpu-build','llama-src','llama-vulkan-lib','llama-cuda-lib','out'):
    p.add_argument('--'+option,type=Path,required=True)
a = p.parse_args()
src = Path(__file__).resolve().parent
for device in ['cpu','gpu']:
    folder=(a.out/device).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    build=(a.velri_gpu_build if device=='gpu' else a.velri_cpu_build).resolve()
    commands=sp.check_output(['ninja','-C',build,'-t','commands'],text=True).splitlines()
    test=shlex.split(next(x for x in commands if 'test_engine_runtime.c.o -c ' in x))
    root=Path(test[test.index('-c')+1]).parents[1]
    flags=test[:test.index('-MD')]+['-D_DEFAULT_SOURCE','-I',str(root/'tests'),'-I',str(root/'velri-lib/include')]
    for name,bits in ([('velri-f32',32),('velri-mixed',16)] if device=='gpu' else [('velri',32)]):
        defs=[f'-DBENCH_BITS={bits}',f'-DBENCH_BITS_TEXT="{bits}"','-DBENCH_CHUNK=1024'] if device=='gpu' else []
        obj=folder/(name+'.o')
        sp.run([*flags,*defs,'-c',src/f'velri-{device}.c','-o',obj],check=True,cwd=build)
        sp.run([flags[0],obj,build/'libvelri_engine.a',build/'velri-lib/libvelri.a','-ldl','-lm','-pthread','-o',folder/name],check=True)
    for name,libdir in ([('llama-cuda',a.llama_cuda_lib),('llama-vulkan',a.llama_vulkan_lib)] if device=='gpu' else [('llama',a.llama_vulkan_lib)]):
        lib=libdir.resolve()
        sp.run(['c++','-O3','-std=c++17','-march=native','-I'+str(a.llama_src/'include'),'-I'+str(a.llama_src/'ggml/include'),
                src/f'llama-{device}.cpp','-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lllama','-lggml','-lggml-base','-o',folder/name],check=True)
