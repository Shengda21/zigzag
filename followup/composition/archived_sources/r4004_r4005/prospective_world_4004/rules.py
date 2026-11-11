"""Independent implementations of five r4004 digit-state operators."""
import random

RULES = {
    'A': {'kind': 'rotate', 'positions': list(range(8)), 'amount': 1},
    'B': {'kind': 'rotate', 'positions': [0, 1, 2], 'amount': 1},
    'C': {'kind': 'add', 'position': 4, 'value': 3},
    'D': {'kind': 'conditional_swap', 'positions': [5, 7],
          'control': [0, 1], 'predicate': 'first_greater_than_second'},
    'E': {'kind': 'reverse', 'positions': [0, 1, 2, 3]},
}


def reference(root, word):
    x = tuple(root)
    trace = []
    for op in word:
        a = list(x)
        if op == 'A':
            x = x[1:] + x[:1]
        elif op == 'B':
            x = (x[1], x[2], x[0]) + x[3:]
        elif op == 'C':
            a[4] = (x[4] + 3) & 7
            x = tuple(a)
        elif op == 'D':
            if x[0] > x[1]:
                a[5], a[7] = a[7], a[5]
            x = tuple(a)
        elif op == 'E':
            x = x[:4][::-1] + x[4:]
        else:
            raise ValueError(op)
        trace.append(x)
    return trace


def execute(root, word):
    x = list(root)
    trace = []
    for op in word:
        rule = RULES[op]
        old = list(x)
        kind = rule['kind']
        if kind in ('rotate', 'reverse'):
            pos = rule['positions']
            values = [old[i] for i in pos]
            values = values[::-1] if kind == 'reverse' else \
                values[rule['amount']:] + values[:rule['amount']]
            for i, value in zip(pos, values):
                x[i] = value
        elif kind == 'conditional_swap':
            if old[rule['control'][0]] > old[rule['control'][1]]:
                a, b = rule['positions']
                x[a], x[b] = old[b], old[a]
        else:
            p = rule['position']
            x[p] = (old[p] + rule['value']) % 8
        trace.append(tuple(x))
    return trace


def audit():
    rng = random.Random(400420261110)
    for _ in range(10000):
        root = tuple(rng.randrange(8) for _ in range(8))
        word = ''.join(rng.choice('ABCDE') for _ in range(rng.randrange(1, 9)))
        assert reference(root, word) == execute(root, word)
    orders = {}
    for op in 'ABCDE':
        witnessed = []
        for _ in range(32):
            root = tuple(rng.randrange(8) for _ in range(8))
            state = root
            for n in range(1, 9):
                state = reference(state, op)[0]
                if state == root:
                    witnessed.append(n)
                    break
        orders[op] = max(witnessed)
    assert orders == {'A': 8, 'B': 3, 'C': 8, 'D': 2, 'E': 2}, orders
    return orders


if __name__ == '__main__':
    print(audit())
