import csv

with open('study/results/benchmark_comparison.csv') as f:
    rows = list(csv.DictReader(f))

base = [r for r in rows if 'Baseline' in r['model']]
robust = [r for r in rows if 'Robust' in r['model']]

print(f"{'DATASET':<12} {'PLATFORM':<12} {'BASE bACC':<12} {'ROBUST bACC':<12} {'BASE AUC':<12} {'ROBUST AUC':<12}")
print("-" * 75)

for b, r in zip(base, robust):
    b_acc = float(b['bACC']) * 100
    r_acc = float(r['bACC']) * 100
    b_auc = float(b['AUC']) * 100
    r_auc = float(r['AUC']) * 100
    print(f"{b['dataset']:<12} {b['platform']:<12} {b_acc:>6.2f}%      {r_acc:>6.2f}%      {b_auc:>6.2f}%      {r_auc:>6.2f}%")
