# 混淆矩阵诊断报告：prod_v2_b2_39c_c025

- 模型: `D:\RK3588&Orin Nano\ORIN NANO\Model Training\runs\detect\prod_v2_b2\weights\best.pt`
- 评估集: `D:\RK3588&Orin Nano\ORIN NANO\Model Training\datasets_master\production\v1.0\data.yaml`
- conf=0.25  imgsz=256  类别数=39
- GT总框=11182  预测总框=9714  FP=2389  FN=3857

## 一、混淆对 (GT类被判成其它类, Top30)

| GT类 | 错判为 | 次数 | 占GT类比例 |
|---|---|---|---|
| Bright scratch | Pitted surface | 39 | 3.9% |
| Pitted surface | Bright scratch | 34 | 4.5% |
| Oil spot | Inclusion | 12 | 2.0% |
| Water spot | Oil spot | 11 | 2.5% |
| break | fray | 9 | 2.6% |
| Patches | Bright scratch | 8 | 0.9% |
| Inclusion | Oil spot | 7 | 0.4% |
| Water spot | Inclusion | 7 | 1.6% |
| Inclusion | Water spot | 4 | 0.2% |
| Oil spot | Water spot | 4 | 0.7% |
| Rolled pit | Crease | 4 | 7.7% |
| Crease | Welding line | 3 | 4.6% |
| Inclusion | Patches | 3 | 0.2% |
| White rust | Bright scratch | 3 | 1.3% |
| water_slag_mark | slag_skin | 3 | 10.3% |
| longitudinal_crack | Inclusion | 3 | 10.7% |
| Crazing | break | 2 | 0.4% |
| Iron scale compression | Inclusion | 2 | 3.7% |
| Oil spot | Rolled pit | 2 | 0.3% |
| Oil spot | foreign_obj | 2 | 0.3% |
| Patches | Inclusion | 2 | 0.2% |
| Pitted surface | Inclusion | 2 | 0.3% |
| Rolled pit | Water spot | 2 | 3.9% |
| slag_skin | Slag inclusion | 2 | 4.9% |
| slag_skin | water_slag_mark | 2 | 4.9% |
| warp | Dark scratches | 2 | 8.3% |
| blocky_scale | foreign_obj | 2 | 8.0% |
| striated_scale | Inclusion | 2 | 9.1% |
| break | blocky_scale | 2 | 0.6% |
| Bright scratch | Crazing | 1 | 0.1% |

## 二、漏检榜 (GT类 → 无框, 按FN数降序)

| 类名 | FN | GT数 | 漏检率 |
|---|---|---|---|
| Waist folding | 898 | 1153 | 77.9% |
| Bright scratch | 517 | 1010 | 51.2% |
| Inclusion | 409 | 1724 | 23.7% |
| Pitted surface | 342 | 747 | 45.8% |
| Crazing | 289 | 529 | 54.6% |
| Patches | 194 | 861 | 22.5% |
| Rolled in scale | 177 | 554 | 31.9% |
| break | 175 | 343 | 51.0% |
| Oil spot | 114 | 590 | 19.3% |
| Slag inclusion | 104 | 307 | 33.9% |
| Water spot | 103 | 439 | 23.5% |
| Silk spot | 91 | 304 | 29.9% |
| Dark scratches | 90 | 264 | 34.1% |
| White rust | 77 | 232 | 33.2% |
| Secondary rust skin | 59 | 153 | 38.6% |
| Finishing roll printing | 40 | 118 | 33.9% |
| Iron sheet ash | 38 | 139 | 27.3% |
| Crease | 31 | 65 | 47.7% |
| Rolled pit | 28 | 52 | 53.8% |
| wrinkle | 15 | 46 | 32.6% |
| warp | 14 | 24 | 58.3% |
| Welding line | 10 | 219 | 4.6% |
| fray | 7 | 246 | 2.9% |
| external_fold | 6 | 25 | 24.0% |
| slag_skin | 5 | 41 | 12.2% |
| Iron scale compression | 4 | 54 | 7.4% |
| Red iron sheet | 3 | 119 | 2.5% |
| blowhole | 3 | 225 | 1.3% |
| uneven | 3 | 90 | 3.3% |
| Punching | 2 | 165 | 1.2% |
| longitudinal_crack | 2 | 28 | 7.1% |
| blocky_scale | 2 | 25 | 8.0% |
| Crescent gap | 1 | 106 | 0.9% |
| Oxide scale of plate system | 1 | 18 | 5.6% |
| water_slag_mark | 1 | 29 | 3.5% |
| striated_scale | 1 | 22 | 4.5% |
| foreign_obj | 1 | 33 | 3.0% |
| Oxide scale of temperature system | 0 | 60 | 0.0% |
| cutting_opening | 0 | 23 | 0.0% |

## 三、误报榜 (无GT → 框, 按FP数降序)

| 类名 | FP数 |
|---|---|
| Inclusion | 645 |
| Bright scratch | 235 |
| Waist folding | 227 |
| Slag inclusion | 149 |
| Patches | 144 |
| Oil spot | 120 |
| Pitted surface | 120 |
| Rolled in scale | 118 |
| Crazing | 103 |
| Water spot | 87 |
| White rust | 75 |
| Iron sheet ash | 64 |
| Finishing roll printing | 48 |
| Secondary rust skin | 47 |
| Silk spot | 39 |
| Dark scratches | 35 |
| Iron scale compression | 15 |
| blowhole | 13 |
| Crease | 12 |
| Welding line | 12 |
| Crescent gap | 9 |
| Red iron sheet | 9 |
| slag_skin | 8 |
| external_fold | 8 |
| wrinkle | 7 |
| water_slag_mark | 6 |
| fray | 6 |
| blocky_scale | 5 |
| Punching | 4 |
| Rolled pit | 4 |
| foreign_obj | 4 |
| longitudinal_crack | 3 |
| break | 3 |
| Oxide scale of plate system | 2 |
| striated_scale | 2 |
| Oxide scale of temperature system | 1 |
| cutting_opening | 0 |
| warp | 0 |
| uneven | 0 |

## 四、逐类混淆矩阵统计

| ID | 类名 | GT | 预测 | TP | FN | FP | recall | precision |
|---|---|---|---|---|---|---|---|---|
| 0 | Bright scratch | 1010 | 731 | 451 | 517 | 235 | 0.4465 | 0.617 |
| 1 | Crazing | 529 | 342 | 238 | 289 | 103 | 0.4499 | 0.6959 |
| 2 | Crease | 65 | 46 | 30 | 31 | 12 | 0.4615 | 0.6522 |
| 3 | Crescent gap | 106 | 114 | 105 | 1 | 9 | 0.9906 | 0.9211 |
| 4 | Dark scratches | 264 | 212 | 174 | 90 | 35 | 0.6591 | 0.8208 |
| 5 | Finishing roll printing | 118 | 127 | 78 | 40 | 48 | 0.661 | 0.6142 |
| 6 | Inclusion | 1724 | 1978 | 1299 | 409 | 645 | 0.7535 | 0.6567 |
| 7 | Iron scale compression | 54 | 63 | 48 | 4 | 15 | 0.8889 | 0.7619 |
| 8 | Iron sheet ash | 139 | 164 | 100 | 38 | 64 | 0.7194 | 0.6098 |
| 9 | Oil spot | 590 | 594 | 456 | 114 | 120 | 0.7729 | 0.7677 |
| 10 | Oxide scale of plate system | 18 | 19 | 17 | 1 | 2 | 0.9444 | 0.8947 |
| 11 | Oxide scale of temperature system | 60 | 61 | 60 | 0 | 1 | 1.0 | 0.9836 |
| 12 | Patches | 861 | 805 | 657 | 194 | 144 | 0.7631 | 0.8161 |
| 13 | Pitted surface | 747 | 528 | 369 | 342 | 120 | 0.494 | 0.6989 |
| 14 | Punching | 165 | 168 | 163 | 2 | 4 | 0.9879 | 0.9702 |
| 15 | Red iron sheet | 119 | 125 | 116 | 3 | 9 | 0.9748 | 0.928 |
| 16 | Rolled in scale | 554 | 495 | 377 | 177 | 118 | 0.6805 | 0.7616 |
| 17 | Rolled pit | 52 | 24 | 16 | 28 | 4 | 0.3077 | 0.6667 |
| 18 | Secondary rust skin | 153 | 140 | 93 | 59 | 47 | 0.6078 | 0.6643 |
| 19 | Silk spot | 304 | 253 | 213 | 91 | 39 | 0.7007 | 0.8419 |
| 20 | Slag inclusion | 307 | 355 | 202 | 104 | 149 | 0.658 | 0.569 |
| 21 | Waist folding | 1153 | 482 | 255 | 898 | 227 | 0.2212 | 0.529 |
| 22 | Water spot | 439 | 412 | 315 | 103 | 87 | 0.7175 | 0.7646 |
| 23 | Welding line | 219 | 225 | 209 | 10 | 12 | 0.9543 | 0.9289 |
| 24 | White rust | 232 | 227 | 152 | 77 | 75 | 0.6552 | 0.6696 |
| 25 | cutting_opening | 23 | 23 | 23 | 0 | 0 | 1.0 | 1.0 |
| 26 | water_slag_mark | 29 | 31 | 23 | 1 | 6 | 0.7931 | 0.7419 |
| 27 | slag_skin | 41 | 44 | 32 | 5 | 8 | 0.7805 | 0.7273 |
| 28 | longitudinal_crack | 28 | 26 | 23 | 2 | 3 | 0.8214 | 0.8846 |
| 29 | warp | 24 | 7 | 7 | 14 | 0 | 0.2917 | 1.0 |
| 30 | external_fold | 25 | 27 | 19 | 6 | 8 | 0.76 | 0.7037 |
| 31 | wrinkle | 46 | 39 | 31 | 15 | 7 | 0.6739 | 0.7949 |
| 32 | blocky_scale | 25 | 27 | 19 | 2 | 5 | 0.76 | 0.7037 |
| 33 | striated_scale | 22 | 21 | 19 | 1 | 2 | 0.8636 | 0.9048 |
| 34 | foreign_obj | 33 | 40 | 31 | 1 | 4 | 0.9394 | 0.775 |
| 35 | blowhole | 225 | 235 | 222 | 3 | 13 | 0.9867 | 0.9447 |
| 36 | break | 343 | 162 | 157 | 175 | 3 | 0.4577 | 0.9691 |
| 37 | fray | 246 | 255 | 239 | 7 | 6 | 0.9715 | 0.9373 |
| 38 | uneven | 90 | 87 | 87 | 3 | 0 | 0.9667 | 1.0 |