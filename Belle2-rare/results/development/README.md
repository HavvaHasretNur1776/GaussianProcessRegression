# How the reconstruction was chosen

These records document earlier design choices. Each row identifies its own configuration or comparison. Several stages used different targets, training budgets, selections or splits. The primary paired validation result for the final 11-input angular GP is in the [main README](../../README.md).

| Study | What to inspect |
|---|---|
| Detector response | [Response diagnostics](response_diagnostics.png) for alternative object acceptance and smearing assumptions |
| Kaon and recoil | [Early angular GP comparison](gpr_comparison.png) and [recoil-information checks](comparison.csv) |
| Alternative target | [Three-component B-momentum fit](test_performance.png) with 12 inputs and 300 inducing points |
| Kaon direction in the training label | [True-kaon versus reconstructed-kaon target](comparison.json) at 300 inducing points |
| Model size | [Earlier inducing-point scan](scan_summary.csv) for the configuration studied at that stage |
| Toy sensitivity exercise | [Template-study results](sensitivity_table.csv) under the assumptions recorded in the table |
| Fixed or trainable locations and recoil composition | [Four paired comparisons](M300_recoil_comparison.csv) from the SWAN run at 300 inducing points |

In the 11-input comparison, allowing the locations to move reduced validation RMSE by 0.000826 GeV². Adding separate charged and photon recoil projections to that trainable model gave a difference consistent with zero in the paired interval. These checks motivated retaining the 11 inputs and trainable locations in the final setup.
