import os


def print_eval(results, summary, log):
    log(f"\n    {'Category':<35} {'Corr':>6} {'Tot':>6} {'Skip':>6} {'Acc':>8}")
    log(f"    {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")

    for c, r in sorted(results.items()):
        if r['is_semantic']:
            log(f"    {c:<35} {r['correct']:>6} {r['total']:>6} "
                f"{r['skipped']:>6} {r['accuracy']*100:>7.1f}%")
    s = summary['semantic']
    log(f"    {'SEMANTIC TOTAL':<35} {s['correct']:>6} {s['total']:>6} "
        f"{s['skipped']:>6} {s['accuracy']*100:>7.1f}%")
    log("")

    for c, r in sorted(results.items()):
        if not r['is_semantic']:
            log(f"    {c:<35} {r['correct']:>6} {r['total']:>6} "
                f"{r['skipped']:>6} {r['accuracy']*100:>7.1f}%")
    s = summary['syntactic']
    log(f"    {'SYNTACTIC TOTAL':<35} {s['correct']:>6} {s['total']:>6} "
        f"{s['skipped']:>6} {s['accuracy']*100:>7.1f}%")

    s = summary['overall']
    log(f"    {'=== OVERALL ===':<35} {s['correct']:>6} {s['total']:>6} "
        f"{s['skipped']:>6} {s['accuracy']*100:>7.1f}%")


def save_all(rdir, results, summary, nn, anex, log_lines):
    with open(os.path.join(rdir, 'analogy_results.txt'), 'w') as f:
        for c, r in sorted(results.items()):
            f.write(f"{c}: {r['correct']}/{r['total']} = "
                    f"{r['accuracy']*100:.1f}% (skip {r['skipped']})\n")
        f.write(f"\nSemantic: {summary['semantic']['accuracy']*100:.1f}%\n")
        f.write(f"Syntactic: {summary['syntactic']['accuracy']*100:.1f}%\n")
        f.write(f"Overall: {summary['overall']['accuracy']*100:.1f}%\n")

    with open(os.path.join(rdir, 'nearest_neighbors.txt'), 'w') as f:
        for w, ns in nn.items():
            f.write(f"{w}: " + ', '.join(f"{n}({s:.3f})" for n, s in ns) + '\n')

    with open(os.path.join(rdir, 'analogy_examples.txt'), 'w') as f:
        for a, b, c, ps in anex:
            f.write(f"{a} : {b} :: {c} : ?  ->  "
                    + ', '.join(f"{w}({s:.3f})" for w, s in ps) + '\n')

    with open(os.path.join(rdir, 'full_log.txt'), 'w') as f:
        f.write('\n'.join(log_lines) + '\n')
