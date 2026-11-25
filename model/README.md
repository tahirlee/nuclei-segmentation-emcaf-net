# model/

Put the network definitions here. Every file defines a class named `nucleinet`, which `train.py`,
`train_dual.py` and `test_instance.py` load with `--arch <file name without .py>`.

| File | Used for |
|---|---|
| `nucleinet.py` | EMCAF-Net (full model) |
| `pvtv2.py` | PVTv2 encoder used by every EMCAF-Net variant |
| `pvt_v2_b2.pth` | ImageNet weights of PVTv2-B2 (download, see the main README; not stored in the repository) |
| `nucleinet_no_emf.py` | ablation: without EMF (encoder features go to CAF unchanged) |
| `nucleinet_no_caf.py` | ablation: without CAF (plain summation of the two inputs) |
| `nucleinet_no_ca.py` | ablation: EMF without channel attention |
| `nucleinet_no_edge.py` | ablation: EMF without edge gate |
| `nucleinet_no_dilated.py` | ablation: EMF without dilated convolutions |
| `nucleinet_plain_emf.py` | ablation: EMF replaced by plain convolutions |
| `nucleinet_aff_caf.py` | ablation: CAF replaced by AFF |
| `nucleinet_emcad_emf.py` | ablation: EMF replaced by an EMCAD-style block |
| `dual_decoder_unet.py` | Dual decoder U-Net baseline (retrained under the same protocol) |
