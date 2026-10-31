"""Independent scalar oracle; Python standard library only. Run from any directory.

Two layers, hidden=4, FF=6, 2 query heads / 1 KV head, key dim=4,
value dim=3, partial rotate-half RoPE, full then sliding-window attention.
Rounds every operation output to f32, as the runtime's tensor boundaries do.
"""
import json
import math
import struct
from pathlib import Path


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


weights = {}

def weight(name, n, norm=False):
    seed = sum((i + 1) * ord(c) for i, c in enumerate(name))
    values = [f32((1 if norm else 0) + (((seed + i * 17) % 37) - 18) / 70) for i in range(n)]
    weights[name] = values
    return values


weight('embedding', 5 * 4)
for layer in range(2):
    prefix = f'hidden-layers/{layer}/'
    for name, n in [('attention/norm', 4), ('mlp/norm', 4)]:
        weight(prefix + name, n, norm=True)
    for name, n in [('attention/q', 8*4), ('attention/k', 4*4),
                    ('attention/v', 3*4), ('attention/output', 4*6),
                    ('mlp/gate', 6*4), ('mlp/up', 6*4), ('mlp/down', 4*6)]:
        weight(prefix + name, n)
weight('final-norm', 4, norm=True)
weight('head', 5*4)


def linear(x, name, m):
    w = weights[name]
    return [[f32(sum(a * b for a, b in zip(row, w[j*len(row):(j+1)*len(row)])))
             for j in range(m)] for row in x]


def norm(x, name):
    return [[f32(v / math.sqrt(sum(t*t for t in row)/len(row) + f32(1e-5)) * w)
             for v, w in zip(row, weights[name])] for row in x]


def rotary(x, heads):
    out = []
    for pos, row in enumerate(x):
        row = row.copy()
        for h in range(heads):
            a, b = row[h*4:h*4+2]
            row[h*4] = f32(a*math.cos(pos) - b*math.sin(pos))
            row[h*4+1] = f32(b*math.cos(pos) + a*math.sin(pos))
        out.append(row)
    return out


def residual(x, y):
    return [[f32(a+b) for a,b in zip(u,v)] for u,v in zip(x,y)]


tokens = [0, 3, 1, 4, 2, 1]
x = [weights['embedding'][t*4:t*4+4] for t in tokens]
for layer in range(2):
    p = f'hidden-layers/{layer}/'
    a = norm(x, p+'attention/norm')
    q = rotary(linear(a, p+'attention/q', 8), 2)
    k = rotary(linear(a, p+'attention/k', 4), 1)
    v = linear(a, p+'attention/v', 3)
    mixed = []
    for pos in range(len(tokens)):
        first = 0 if layer == 0 else max(0, pos-1)
        row = []
        for h in range(2):
            scores = [sum(a*b for a,b in zip(q[pos][h*4:h*4+4], k[j])) * 0.5
                      for j in range(first, pos+1)]
            probs = [math.exp(s - max(scores)) for s in scores]
            row += [f32(sum(p*v[j][d] for p,j in zip(probs, range(first,pos+1)))/sum(probs)) for d in range(3)]
        mixed.append(row)
    x = residual(x, linear(mixed, p+'attention/output', 4))
    a = norm(x, p+'mlp/norm')
    gate, up = linear(a, p+'mlp/gate', 6), linear(a, p+'mlp/up', 6)
    active = [[f32((g/(1+math.exp(-g)))*u) for g,u in zip(gg,uu)] for gg,uu in zip(gate,up)]
    x = residual(x, linear(active, p+'mlp/down', 4))
logits = linear(norm(x, 'final-norm'), 'head', 5)
Path(__file__).with_name('decoder.json').write_text(json.dumps({'tokens': tokens, 'weights': weights, 'logits': logits}, indent=2)+'\n')
