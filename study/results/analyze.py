import csv

with open('study/results/benchmark_comparison.csv', 'r') as f:
    rows = list(csv.DictReader(f))

models = ['Baseline (Pretrained)', 'Approach 2 (No Aug)', 'Approach 2 (Mild Aug)']
metrics = ['AUC', 'bACC_030', 'bACC_050', 'bACC_dual', 'TPR_030', 'TNR_030', 'TPR_050', 'TNR_050']

print('=== OVERALL AVERAGES across 15 conditions ===')
for m in models:
    m_rows = [r for r in rows if r['model'] == m]
    res = {k: sum(float(r[k]) for r in m_rows)/len(m_rows)*100 for k in metrics}
    print(f"{m:<26}: AUC={res['AUC']:.2f}% | bACC(0.30)={res['bACC_030']:.2f}% | bACC(0.50)={res['bACC_050']:.2f}% | Dual={res['bACC_dual']:.2f}%")

datasets = ['CocoGlide', 'Columbia', 'CASIA1.0']
for ds in datasets:
    print(f"\n=== Dataset: {ds} ===")
    for m in models:
        m_rows = [r for r in rows if r['model'] == m and r['dataset'] == ds]
        res = {k: sum(float(r[k]) for r in m_rows)/len(m_rows)*100 for k in metrics}
        print(f"  {m:<24}: AUC={res['AUC']:.2f}% | bACC(0.30)={res['bACC_030']:.2f}% | bACC(0.50)={res['bACC_050']:.2f}% | Dual={res['bACC_dual']:.2f}%")
platforms = ['original', 'whatsapp', 'instagram', 'facebook', 'telegram']

print("\n=== COMPLETE 15-CONDITION AUC COMPARISON ===")
print("| Dataset | Platform | Baseline AUC | App2 (No Aug) | App2 (Mild Aug) | Diff vs No Aug | Diff vs Baseline |")
print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
for ds in datasets:
    for p in platforms:
        b_r = next(r for r in rows if r['model'] == models[0] and r['dataset'] == ds and r['platform'] == p)
        n_r = next(r for r in rows if r['model'] == models[1] and r['dataset'] == ds and r['platform'] == p)
        w_r = next(r for r in rows if r['model'] == models[2] and r['dataset'] == ds and r['platform'] == p)
        b_auc = float(b_r['AUC']) * 100
        n_auc = float(n_r['AUC']) * 100
        w_auc = float(w_r['AUC']) * 100
        d_noaug = w_auc - n_auc
        d_base = w_auc - b_auc
        print(f"| {ds:<9} | {p:<9} | {b_auc:6.2f}% | {n_auc:6.2f}% | {w_auc:6.2f}% | {d_noaug:+6.2f}% | {d_base:+6.2f}% |")

print("\n=== COMPLETE 15-CONDITION bACC (tau=0.30) COMPARISON ===")
print("| Dataset | Platform | Baseline bACC | App2 (No Aug) | App2 (Mild Aug) | Diff vs No Aug | Diff vs Baseline |")
print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
for ds in datasets:
    for p in platforms:
        b_r = next(r for r in rows if r['model'] == models[0] and r['dataset'] == ds and r['platform'] == p)
        n_r = next(r for r in rows if r['model'] == models[1] and r['dataset'] == ds and r['platform'] == p)
        w_r = next(r for r in rows if r['model'] == models[2] and r['dataset'] == ds and r['platform'] == p)
        b_val = float(b_r['bACC_030']) * 100
        n_val = float(n_r['bACC_030']) * 100
        w_val = float(w_r['bACC_030']) * 100
        d_noaug = w_val - n_val
        d_base = w_val - b_val
        print(f"| {ds:<9} | {p:<9} | {b_val:6.2f}% | {n_val:6.2f}% | {w_val:6.2f}% | {d_noaug:+6.2f}% | {d_base:+6.2f}% |")

print("\n=== COMPLETE 15-CONDITION bACC (tau=0.50) COMPARISON ===")
print("| Dataset | Platform | Baseline bACC | App2 (No Aug) | App2 (Mild Aug) | Diff vs No Aug | Diff vs Baseline |")
print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
for ds in datasets:
    for p in platforms:
        b_r = next(r for r in rows if r['model'] == models[0] and r['dataset'] == ds and r['platform'] == p)
        n_r = next(r for r in rows if r['model'] == models[1] and r['dataset'] == ds and r['platform'] == p)
        w_r = next(r for r in rows if r['model'] == models[2] and r['dataset'] == ds and r['platform'] == p)
        b_val = float(b_r['bACC_050']) * 100
        n_val = float(n_r['bACC_050']) * 100
        w_val = float(w_r['bACC_050']) * 100
        d_noaug = w_val - n_val
        d_base = w_val - b_val
        print(f"| {ds:<9} | {p:<9} | {b_val:6.2f}% | {n_val:6.2f}% | {w_val:6.2f}% | {d_noaug:+6.2f}% | {d_base:+6.2f}% |")

