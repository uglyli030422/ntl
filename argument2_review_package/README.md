# Argument 2 review package

本文件夹整理的是论文拆分后的“论点2”相关代码、数据、结果和图件。

## 论点2的定位

论点2不是继续证明“SVI 导致夜光误差”，而是分解论点1中观察到的脆弱性-误差梯度：低实测用电社区更容易被有限信息的夜光代理高估，而低实测用电又与高 SVI 社区在空间上部分重叠。因此，边际上的 SVI-误差梯度主要来自“低用电端统计压缩 + 低用电与脆弱性共聚”，而不是一个单独的 SVI 因果机制。

对应论文表述可概括为：

- 论点1：正向夜光电力代理误差集中暴露在高 SVI 社区。
- 论点2：实测用电控制与同用电分层显示，这个梯度主要由低用电端校准误差和低用电-SVI 共聚解释。
- 论点3：城市形态进一步解释相同实测用电下夜光可见性为何不同。
- 论点4：上述误差会影响脆弱低用电区域的筛查。

## 文件结构

- `scripts/`
  - `build_appendix_a2_experiments.py`：生成 Table S3a-S3c，检验低用电与高 SVI 的共聚，以及低夜光筛查的漏识别集中性。
  - `transfer_validate_core_to_external_cities.py`：用 Tokyo、Amsterdam、London 三个核心城市训练夜光-用电模型，并直接预测 Marseille、Sydney，输出两城外部验证结果。
  - `make_main_figure2_statistical_blooming.py`：生成论点2主图 `Figure2_statistical_blooming_v1`。
- `data/`
  - `city_bias_metrics.csv`：三个核心城市的统一误差分析单元。
  - `marseille_fine_scale_bias.csv`：Marseille 外部验证城市细尺度输入。
  - `sydney_ausgrid_fine_scale_bias.csv`：Sydney 外部验证城市细尺度输入。
  - `result3_multithreshold_luminous_poverty_labels.csv`：用于 A2 低用电-高 SVI 共聚与漏识别诊断的核心城市标签表。
- `outputs/result2_observed_electricity_controls/`
  - `svi_bias_regression_with_observed_control.csv`：SVI-误差梯度在加入实测用电控制前后的变化。
  - `svi_gradient_within_observed_electricity_quintiles.csv`：各城市实测用电五分位内的 SVI 梯度。
  - `svi_gradient_pooled_within_observed_quintiles.csv`：同用电五分位内合并估计的 SVI 梯度。
  - `calibration_by_observed_electricity_decile.csv`：按实测用电十分位的校准曲线数据。
  - `supplementary_observed_electricity_control_and_calibration.*`：对应补充图。
- `outputs/appendix_a2_vulnerability_error_experiments/`
  - `table_s3a_electricity_svi_coupling.csv`：低实测用电和高 SVI 的共聚程度。
  - `table_s3b_screening_baselines.csv`：低夜光规则与随机/同用电中性基线的漏识别比较。
  - `table_s3c_social_concentration_of_missed_low_electricity.csv`：漏识别低用电单元的社会集中性。
  - `appendix_a2_experiment_summary.md`：以上结果的文字摘要。
- `outputs/external_transfer_validation_core_to_new_cities/`
  - `core_to_external_transfer_observed_control_table.csv`：五城 observed-electricity control 表，含 Marseille/Sydney。
  - `core_to_external_transfer_summary.csv`：三核心城市 OOF 与两验证城市 direct-transfer 的残差-SVI 梯度摘要。
  - `core_to_external_transfer_svi_deciles.csv`：按 SVI 十分位的残差摘要。
  - `core_to_external_transfer_unit_predictions.csv`：核心城市和两验证城市的单元级预测结果。
- `figures/main/`
  - `Figure2_statistical_blooming_v1.png`
  - `Figure2_statistical_blooming_v1.svg`
- `figures/supplementary/`
  - `Supplementary_observed_electricity_control_and_calibration.png`
  - `Supplementary_observed_electricity_control_and_calibration.svg`

## 复现命令

在本文件夹的上一级目录运行：

```bash
python argument2_review_package/scripts/build_appendix_a2_experiments.py
python argument2_review_package/scripts/transfer_validate_core_to_external_cities.py
python argument2_review_package/scripts/make_main_figure2_statistical_blooming.py
```

这些脚本已改为读取本包内部的 `data/` 和 `outputs/` 路径，不依赖原电脑上的 `D:\ntl` 或其他固定路径。

## 关于 Marseille 和 Sydney

Marseille 和 Sydney 不参与模型训练。它们在 `transfer_validate_core_to_external_cities.py` 中的角色是 `external_direct`：模型只使用 Tokyo、Amsterdam、London 拟合，然后直接预测 Marseille 和 Sydney 的标准化实测用电，并计算预测残差与 SVI 的关系。

因此，两城在论点2中的作用是检查“低用电端统计压缩和 SVI-残差梯度”是否能在未参与训练的新城市中复现，而不是把两城加入训练集。

## 未包含内容

- 本包不包含已经删除的“非夜光对照模型”相关脚本、表格或注释。
- 本包不包含 147 MB 的 `city_out_of_fold_predictions.csv` 大表，以避免 GitHub 单文件 100 MB 限制。论点2复现所需结果已经通过包内较小的核心表和单元预测表提供。
