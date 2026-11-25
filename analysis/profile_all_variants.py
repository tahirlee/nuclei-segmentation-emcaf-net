"""
Profile ALL EMCAF-Net variants: Params, GFLOPs, FPS.
Run from the project root: python analysis/profile_all_variants.py
Copy all nucleinet_*.py files into model/ before running.
Requires: pip install thop
"""
import warnings
warnings.filterwarnings("ignore")          # suppress all UserWarning / FutureWarning
import logging
logging.disable(logging.WARNING)           # suppress timm registry log messages

import sys, time, torch
sys.path.insert(0, '.')

from thop import profile

import copy
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WARMUP = 20
N      = 200

def measure(model):
    """Params and GFLOPs (thop, CPU copy) and GPU FPS: 512x512, batch 1, fp32, forward pass only."""
    model.eval()
    dummy_cpu = torch.randn(1, 3, 512, 512)
    flops, params = profile(copy.deepcopy(model), inputs=(dummy_cpu,), verbose=False)
    model = model.to(DEVICE)
    dummy = dummy_cpu.to(DEVICE)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(dummy)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N):
            model(dummy)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
    fps = N / (time.time() - t0)
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return params / 1e6, flops / 1e9, fps

print(f'Device: {DEVICE}' + (f' ({torch.cuda.get_device_name(0)})' if DEVICE.type == 'cuda' else ''))

results = {}

# 1. Full model (proposed EMCAF-Net)
try:
    from model.nucleinet import nucleinet
    results['EMCAF-Net (Full)'] = measure(nucleinet())
    print('[1/7] Full model done')
except Exception as e:
    print(f'[1/7] Full model FAILED: {e}')

# 2. w/o Channel Attention  (EMF sub-component ablation)
try:
    from model.nucleinet_no_ca import nucleinet as m2
    results['w/o Channel Attention'] = measure(m2())
    print('[2/7] No-CA done')
except Exception as e:
    print(f'[2/7] No-CA FAILED: {e}')

# 3. w/o Edge Gate  (EMF sub-component ablation)
try:
    from model.nucleinet_no_edge import nucleinet as m3
    results['w/o Edge Gate'] = measure(m3())
    print('[3/7] No-Edge done')
except Exception as e:
    print(f'[3/7] No-Edge FAILED: {e}')

# 4. w/o Dilated Conv  (EMF sub-component ablation)
try:
    from model.nucleinet_no_dilated import nucleinet as m4
    results['w/o Dilated Conv'] = measure(m4())
    print('[4/7] No-Dilated done')
except Exception as e:
    print(f'[4/7] No-Dilated FAILED: {e}')

# 5. Plain Conv replacement for EMF  (comparable-capacity conventional baseline)
try:
    from model.nucleinet_plain_emf import nucleinet as m5
    results['Plain Conv (EMF replaced)'] = measure(m5())
    print('[5/7] Plain-EMF done')
except Exception as e:
    print(f'[5/7] Plain-EMF FAILED: {e}')

# 6. AFF replacing CAF  (CAF replacement comparison)
try:
    from model.nucleinet_aff_caf import nucleinet as m6
    results['AFF (CAF replaced)'] = measure(m6())
    print('[6/7] AFF-CAF done')
except Exception as e:
    print(f'[6/7] AFF-CAF FAILED: {e}')

# 7. EMCAD-style block replacing EMF  (EMF replacement comparison)
try:
    from model.nucleinet_emcad_emf import nucleinet as m7
    results['EMCAD-style (EMF replaced)'] = measure(m7())
    print('[7/7] EMCAD-EMF done')
except Exception as e:
    print(f'[7/7] EMCAD-EMF FAILED: {e}')

# ── Print table ───────────────────────────────────────────────────────────────
print('\n' + '=' * 68)
print(f"{'Model':<33} {'Params (M)':>10} {'GMACs':>10} {'FPS':>10}")
print('=' * 68)
for name, (p, g, f) in results.items():
    marker = ' *' if name == 'EMCAF-Net (Full)' else ''
    print(f"{name:<33} {p:>10.3f} {g:>10.3f} {f:>10.2f}{marker}")
print('=' * 68)
print('* = proposed method')
print('GMACs = multiply-accumulate operations counted by thop (1 MAC = 2 FLOPs).')
print('FPS   = forward pass, 512 x 512, batch 1, fp32, on ' +
      (torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'))