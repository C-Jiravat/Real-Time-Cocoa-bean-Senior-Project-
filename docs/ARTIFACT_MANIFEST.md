# Artifact Manifest

This acquisition manifest identifies the nine physical model and dataset
components observed locally at the SP-000 baseline. They are deliberately
excluded from Git and must be acquired or verified separately before a runtime
profile uses them. It intentionally does not enumerate every large local binary,
archive, report, or research asset in the workspace.

| Role | Observed local path | SHA-256 | Acquisition source | License | Provenance/status | Runtime note |
| --- | --- | --- | --- | --- | --- | --- |
| Cocoa bean detector (PyTorch) | `yolo11n_run/weights/best.pt` | `DD9B512F8D820AA5EA0B7FFBEACD2B3B4F654BEEB4C18F9F500F210A7D351278` | Not recorded | Not recorded | Observed local copy; provenance Not recorded | Legacy detector candidate; validate the supported detector contract before use. |
| Cocoa bean detector (ONNX) | `yolo11n_run/weights/best.onnx` | `D97F6CA92E5E2099BD399469F252E5DD8CBF1E86F4FE232A28718A39D0A66AA5` | Not recorded | Not recorded | Observed local copy; provenance Not recorded | Legacy detector candidate; inspect before activation. |
| Color classifier (PyTorch) | `Phase2_batch128_color_best.pth` | `11F71845D1532F9B004F7FEFAA1C56C1EB97CA8F5291CFF17CEBB175D33A7E36` | Not recorded | Not recorded | Observed local copy; provenance Not recorded | Validate the Color classifier tensor contract in Model Lab before use. |
| Color classifier (ONNX) | `Phase2_batch128_color_best.onnx` | `BDA85323F30A59E3FE94F8E3B1DC2E06851EDB68EBA35F9F2C2F8629B2FFAC3D` | Not recorded | Not recorded | Observed local copy; provenance Not recorded | Inspect the Color classifier contract before activation. |
| Defect classifier (PyTorch) | `Phase3_WD0.15_best.pth` | `689B733287F632A4ADF28507DCAB0676C199A9F0D76E9B950D2B303492FBEAD1` | Not recorded | Not recorded | Observed local copy; provenance Not recorded | Validate the Defect classifier tensor contract in Model Lab before use. |
| Defect classifier (ONNX bundle graph) | `Phase3_WD0.15_best.onnx` | `A46B4904B5E6FDB1D25FB68D8CDBB663BEF250A2F84B1077233F9D3A29B014B8` | Not recorded | Not recorded | Observed local copy; provenance Not recorded | One component of a two-file bundle; the graph and external data must both match before use. |
| Defect classifier (ONNX bundle external data) | `Phase3_WD0.15_best.onnx.data` | `4109642E75C7BBBA34C9FAF6B5B78E6068F16512282E0FA3440C239CFD09767B` | Not recorded | Not recorded | Observed local copy; provenance Not recorded | One component of a two-file bundle; the graph and external data must both match before use. |
| Color evaluation dataset | `Test_Color.zip` | `F1B7972DB26F16640A2436DEB69FF1B056FC4E093668AB59CC1A15A16CA75D25` | Not recorded | Not recorded | Observed local copy; provenance Not recorded | Import only through the Color Dataset ZIP workflow. |
| Defect evaluation dataset | `test_data_defect.zip` | `97A765D1F3CE3F392BF81265C92AE36980F3BD55AD7341F834BEFA1503DA99B4` | Not recorded | Not recorded | Observed local copy; provenance Not recorded | Import only through the Defect Dataset ZIP workflow. |

## Verification

On Windows PowerShell, verify a downloaded artifact before registering it:

```powershell
Get-FileHash -Algorithm SHA256 <path-to-artifact>
```

The listed hashes identify the observed local copies and were verified with
`Get-FileHash` on 2026-07-28. They are not an approval of model compatibility,
accuracy, licensing, provenance, or production activation. The Model Lab
validates those properties and records the hash in each run.
