import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Raw log data (truncated here, but we can parse user-provided text)
log_text = """
[epoch 1] step 100/38082 loss=0.5997
[epoch 1] step 200/38082 loss=0.5089
[epoch 1] step 300/38082 loss=0.3058
[epoch 1] step 400/38082 loss=0.4714
[epoch 1] step 500/38082 loss=0.5657
[epoch 1] step 600/38082 loss=0.2807
[epoch 1] step 700/38082 loss=0.1907
[epoch 1] step 800/38082 loss=0.2448
[epoch 1] step 900/38082 loss=0.2482
[epoch 1] step 1000/38082 loss=0.2599
[epoch 1] step 1100/38082 loss=0.3537
[epoch 1] step 1200/38082 loss=0.2110
[epoch 1] step 1300/38082 loss=0.1765
[epoch 1] step 1400/38082 loss=0.2932
[epoch 1] step 1500/38082 loss=0.2712
[epoch 1] step 1600/38082 loss=0.2563
[epoch 1] step 1700/38082 loss=0.2378
[epoch 1] step 1800/38082 loss=0.4259
[epoch 1] step 1900/38082 loss=0.2185
[epoch 1] step 2000/38082 loss=0.2790
[epoch 1] step 2100/38082 loss=0.4128
[epoch 1] step 2200/38082 loss=0.1824
[epoch 1] step 2300/38082 loss=0.3142
[epoch 1] step 2400/38082 loss=0.2449
[epoch 1] step 2500/38082 loss=0.1698
[epoch 1] step 2600/38082 loss=0.2487
[epoch 1] step 2700/38082 loss=0.2855
[epoch 1] step 2800/38082 loss=0.4254
[epoch 1] step 2900/38082 loss=0.3384
[epoch 1] step 3000/38082 loss=0.4053
[epoch 1] step 3100/38082 loss=0.1702
[epoch 1] step 3200/38082 loss=0.2787
[epoch 1] step 3300/38082 loss=0.3242
[epoch 1] step 3400/38082 loss=0.2014
[epoch 1] step 3500/38082 loss=0.2344
[epoch 1] step 3600/38082 loss=0.2293
[epoch 1] step 3700/38082 loss=0.2353
[epoch 1] step 3800/38082 loss=0.1766
[epoch 1] step 3900/38082 loss=0.3255
[epoch 1] step 4000/38082 loss=0.1274
[epoch 1] step 4100/38082 loss=0.3196
[epoch 1] step 4200/38082 loss=0.2328
[epoch 1] step 4300/38082 loss=0.1913
[epoch 1] step 4400/38082 loss=0.0886
[epoch 1] step 4500/38082 loss=0.2626
[epoch 1] step 4600/38082 loss=0.3174
[epoch 1] step 4700/38082 loss=0.1076
[epoch 1] step 4800/38082 loss=0.1996
[epoch 1] step 4900/38082 loss=0.2204
[epoch 1] step 5000/38082 loss=0.1747
[epoch 1] step 5100/38082 loss=0.3165
[epoch 1] step 5200/38082 loss=0.1531
[epoch 1] step 5300/38082 loss=0.2506
[epoch 1] step 5400/38082 loss=0.1430
[epoch 1] step 5500/38082 loss=0.1790
[epoch 1] step 5600/38082 loss=0.1304
[epoch 1] step 5700/38082 loss=0.1813
[epoch 1] step 5800/38082 loss=0.1199
[epoch 1] step 5900/38082 loss=0.2374
[epoch 1] step 6000/38082 loss=0.2130
[epoch 1] step 6100/38082 loss=0.2336
[epoch 1] step 6200/38082 loss=0.1600
[epoch 1] step 6300/38082 loss=0.1764
[epoch 2] step 6400/38082 loss=0.1562
[epoch 2] step 6500/38082 loss=0.2333
[epoch 2] step 6600/38082 loss=0.2002
[epoch 2] step 6700/38082 loss=0.2645
[epoch 2] step 6800/38082 loss=0.2449
[epoch 2] step 6900/38082 loss=0.2343
[epoch 2] step 7000/38082 loss=0.1793
[epoch 2] step 7100/38082 loss=0.3714
[epoch 2] step 7200/38082 loss=0.2000
[epoch 2] step 7300/38082 loss=0.1723
[epoch 2] step 7400/38082 loss=0.2630
[epoch 2] step 7500/38082 loss=0.1999
[epoch 2] step 7600/38082 loss=0.1934
[epoch 2] step 7700/38082 loss=0.2174
[epoch 2] step 7800/38082 loss=0.1579
[epoch 2] step 7900/38082 loss=0.0769
[epoch 2] step 8000/38082 loss=0.1444
[epoch 2] step 8100/38082 loss=0.3234
[epoch 2] step 8200/38082 loss=0.1647
[epoch 2] step 8300/38082 loss=0.2326
[epoch 2] step 8400/38082 loss=0.1986
[epoch 2] step 8500/38082 loss=0.1361
[epoch 2] step 8600/38082 loss=0.2269
[epoch 2] step 8700/38082 loss=0.1661
[epoch 2] step 8800/38082 loss=0.1288
[epoch 2] step 8900/38082 loss=0.1515
[epoch 2] step 9000/38082 loss=0.1188
[epoch 2] step 9100/38082 loss=0.1703
[epoch 2] step 9200/38082 loss=0.2006
[epoch 2] step 9300/38082 loss=0.1487
[epoch 2] step 9400/38082 loss=0.1527
[epoch 2] step 9500/38082 loss=0.2246
[epoch 2] step 9600/38082 loss=0.2335
[epoch 2] step 9700/38082 loss=0.2083
[epoch 2] step 9800/38082 loss=0.2401
[epoch 2] step 9900/38082 loss=0.1888
[epoch 2] step 10000/38082 loss=0.1005
[epoch 2] step 10100/38082 loss=0.2637
[epoch 2] step 10200/38082 loss=0.1853
[epoch 2] step 10300/38082 loss=0.2158
[epoch 2] step 10400/38082 loss=0.1489
[epoch 2] step 10500/38082 loss=0.2678
[epoch 2] step 10600/38082 loss=0.2396
[epoch 2] step 10700/38082 loss=0.2962
[epoch 2] step 10800/38082 loss=0.1258
[epoch 2] step 10900/38082 loss=0.1732
[epoch 2] step 11000/38082 loss=0.1914
[epoch 2] step 11100/38082 loss=0.1985
[epoch 2] step 11200/38082 loss=0.3361
[epoch 2] step 11300/38082 loss=0.0860
[epoch 2] step 11400/38082 loss=0.2156
[epoch 2] step 11500/38082 loss=0.2312
[epoch 2] step 11600/38082 loss=0.1368
[epoch 2] step 11700/38082 loss=0.2737
[epoch 2] step 11800/38082 loss=0.2697
[epoch 2] step 11900/38082 loss=0.2147
[epoch 2] step 12000/38082 loss=0.1476
[epoch 2] step 12100/38082 loss=0.1850
[epoch 2] step 12200/38082 loss=0.2609
[epoch 2] step 12300/38082 loss=0.2050
[epoch 2] step 12400/38082 loss=0.2801
[epoch 2] step 12500/38082 loss=0.1979
[epoch 2] step 12600/38082 loss=0.2325
[epoch 3] step 12700/38082 loss=0.2737
[epoch 3] step 12800/38082 loss=0.0875
[epoch 3] step 12900/38082 loss=0.2710
[epoch 3] step 13000/38082 loss=0.1456
[epoch 3] step 13100/38082 loss=0.1790
[epoch 3] step 13200/38082 loss=0.1547
[epoch 3] step 13300/38082 loss=0.1205
[epoch 3] step 13400/38082 loss=0.1970
[epoch 3] step 13500/38082 loss=0.1492
[epoch 3] step 13600/38082 loss=0.2132
[epoch 3] step 13700/38082 loss=0.2372
[epoch 3] step 13800/38082 loss=0.1134
[epoch 3] step 13900/38082 loss=0.1662
[epoch 3] step 14000/38082 loss=0.1607
[epoch 3] step 14100/38082 loss=0.1888
[epoch 3] step 14200/38082 loss=0.1291
[epoch 3] step 14300/38082 loss=0.2134
[epoch 3] step 14400/38082 loss=0.1075
[epoch 3] step 14500/38082 loss=0.1258
[epoch 3] step 14600/38082 loss=0.1023
[epoch 3] step 14700/38082 loss=0.2335
[epoch 3] step 14800/38082 loss=0.1669
[epoch 3] step 14900/38082 loss=0.1172
[epoch 3] step 15000/38082 loss=0.1900
[epoch 3] step 15100/38082 loss=0.1821
[epoch 3] step 15200/38082 loss=0.2782
[epoch 3] step 15300/38082 loss=0.1111
[epoch 3] step 15400/38082 loss=0.2108
[epoch 3] step 15500/38082 loss=0.1036
[epoch 3] step 15600/38082 loss=0.1965
[epoch 3] step 15700/38082 loss=0.1525
[epoch 3] step 15800/38082 loss=0.0704
[epoch 3] step 15900/38082 loss=0.3163
[epoch 3] step 16000/38082 loss=0.1607
[epoch 3] step 16100/38082 loss=0.2419
[epoch 3] step 16200/38082 loss=0.0689
[epoch 3] step 16300/38082 loss=0.1863
[epoch 3] step 16400/38082 loss=0.2267
[epoch 3] step 16500/38082 loss=0.2602
[epoch 3] step 16600/38082 loss=0.1047
[epoch 3] step 16700/38082 loss=0.1433
[epoch 3] step 16800/38082 loss=0.1934
[epoch 3] step 16900/38082 loss=0.1703
[epoch 3] step 17000/38082 loss=0.1243
[epoch 3] step 17100/38082 loss=0.2296
[epoch 3] step 17200/38082 loss=0.1093
[epoch 3] step 17300/38082 loss=0.1401
[epoch 3] step 17400/38082 loss=0.1004
[epoch 3] step 17500/38082 loss=0.1088
[epoch 3] step 17600/38082 loss=0.1474
[epoch 3] step 17700/38082 loss=0.1374
[epoch 3] step 17800/38082 loss=0.2068
[epoch 3] step 17900/38082 loss=0.1066
[epoch 3] step 18000/38082 loss=0.2042
[epoch 3] step 18100/38082 loss=0.0948
[epoch 3] step 18200/38082 loss=0.1723
[epoch 3] step 18300/38082 loss=0.1463
[epoch 3] step 18400/38082 loss=0.2607
[epoch 3] step 18500/38082 loss=0.1146
[epoch 3] step 18600/38082 loss=0.1816
[epoch 3] step 18700/38082 loss=0.0952
[epoch 3] step 18800/38082 loss=0.1137
[epoch 3] step 18900/38082 loss=0.1784
[epoch 3] step 19000/38082 loss=0.0825
[epoch 4] step 19100/38082 loss=0.1825
[epoch 4] step 19200/38082 loss=0.2476
[epoch 4] step 19300/38082 loss=0.2368
[epoch 4] step 19400/38082 loss=0.1022
[epoch 4] step 19500/38082 loss=0.0840
[epoch 4] step 19600/38082 loss=0.1680
[epoch 4] step 19700/38082 loss=0.0889
[epoch 4] step 19800/38082 loss=0.2303
[epoch 4] step 19900/38082 loss=0.1540
[epoch 4] step 20000/38082 loss=0.1910
[epoch 4] step 20100/38082 loss=0.1205
[epoch 4] step 20200/38082 loss=0.1495
[epoch 4] step 20300/38082 loss=0.0668
[epoch 4] step 20400/38082 loss=0.1476
[epoch 4] step 20500/38082 loss=0.1223
[epoch 4] step 20600/38082 loss=0.1501
[epoch 4] step 20700/38082 loss=0.1969
[epoch 4] step 20800/38082 loss=0.1358
[epoch 4] step 20900/38082 loss=0.2351
[epoch 4] step 21000/38082 loss=0.1606
[epoch 4] step 21100/38082 loss=0.1274
[epoch 4] step 21200/38082 loss=0.2121
[epoch 4] step 21300/38082 loss=0.0935
[epoch 4] step 21400/38082 loss=0.0716
[epoch 4] step 21500/38082 loss=0.0811
[epoch 4] step 21600/38082 loss=0.1615
[epoch 4] step 21700/38082 loss=0.1539
[epoch 4] step 21800/38082 loss=0.1200
[epoch 4] step 21900/38082 loss=0.1791
[epoch 4] step 22000/38082 loss=0.1416
[epoch 4] step 22100/38082 loss=0.1534
[epoch 4] step 22200/38082 loss=0.1010
[epoch 4] step 22300/38082 loss=0.1369
[epoch 4] step 22400/38082 loss=0.1232
[epoch 4] step 22500/38082 loss=0.1157
[epoch 4] step 22600/38082 loss=0.1512
[epoch 4] step 22700/38082 loss=0.2142
[epoch 4] step 22800/38082 loss=0.0941
[epoch 4] step 22900/38082 loss=0.1362
[epoch 4] step 23000/38082 loss=0.0796
[epoch 4] step 23100/38082 loss=0.1201
[epoch 4] step 23200/38082 loss=0.1108
[epoch 4] step 23300/38082 loss=0.1530
[epoch 4] step 23400/38082 loss=0.1062
[epoch 4] step 23500/38082 loss=0.1067
[epoch 4] step 23600/38082 loss=0.1326
[epoch 4] step 23700/38082 loss=0.0896
[epoch 4] step 23800/38082 loss=0.1397
[epoch 4] step 23900/38082 loss=0.1020
[epoch 4] step 24000/38082 loss=0.0820
[epoch 4] step 24100/38082 loss=0.0903
[epoch 4] step 24200/38082 loss=0.1989
[epoch 4] step 24300/38082 loss=0.0939
[epoch 4] step 24400/38082 loss=0.1572
[epoch 4] step 24500/38082 loss=0.1187
[epoch 4] step 24600/38082 loss=0.1098
[epoch 4] step 24700/38082 loss=0.1187
[epoch 4] step 24800/38082 loss=0.1106
[epoch 4] step 24900/38082 loss=0.0724
[epoch 4] step 25000/38082 loss=0.1365
[epoch 4] step 25100/38082 loss=0.1346
[epoch 4] step 25200/38082 loss=0.1458
[epoch 4] step 25300/38082 loss=0.2348
[epoch 5] step 25400/38082 loss=0.1236
[epoch 5] step 25500/38082 loss=0.1619
[epoch 5] step 25600/38082 loss=0.1297
[epoch 5] step 25700/38082 loss=0.1225
[epoch 5] step 25800/38082 loss=0.2232
[epoch 5] step 25900/38082 loss=0.0932
[epoch 5] step 26000/38082 loss=0.0802
[epoch 5] step 26100/38082 loss=0.1540
[epoch 5] step 26200/38082 loss=0.1917
[epoch 5] step 26300/38082 loss=0.1627
[epoch 5] step 26400/38082 loss=0.1238
[epoch 5] step 26500/38082 loss=0.0844
[epoch 5] step 26600/38082 loss=0.1641
[epoch 5] step 26700/38082 loss=0.1493
[epoch 5] step 26800/38082 loss=0.2224
[epoch 5] step 26900/38082 loss=0.0935
[epoch 5] step 27000/38082 loss=0.0717
[epoch 5] step 27100/38082 loss=0.0839
[epoch 5] step 27200/38082 loss=0.1072
[epoch 5] step 27300/38082 loss=0.0255
[epoch 5] step 27400/38082 loss=0.0932
[epoch 5] step 27500/38082 loss=0.2085
[epoch 5] step 27600/38082 loss=0.1206
[epoch 5] step 27700/38082 loss=0.1382
[epoch 5] step 27800/38082 loss=0.0999
[epoch 5] step 27900/38082 loss=0.1803
[epoch 5] step 28000/38082 loss=0.0985
[epoch 5] step 28100/38082 loss=0.1507
[epoch 5] step 28200/38082 loss=0.1184
[epoch 5] step 28300/38082 loss=0.0885
[epoch 5] step 28400/38082 loss=0.1534
[epoch 5] step 28500/38082 loss=0.1126
[epoch 5] step 28600/38082 loss=0.1098
[epoch 5] step 28700/38082 loss=0.1209
[epoch 5] step 28800/38082 loss=0.0663
[epoch 5] step 28900/38082 loss=0.1814
[epoch 5] step 29000/38082 loss=0.0906
[epoch 5] step 29100/38082 loss=0.1036
[epoch 5] step 29200/38082 loss=0.0226
[epoch 5] step 29300/38082 loss=0.0746
[epoch 5] step 29400/38082 loss=0.0868
[epoch 5] step 29500/38082 loss=0.1338
[epoch 5] step 29600/38082 loss=0.1380
[epoch 5] step 29700/38082 loss=0.1023
[epoch 5] step 29800/38082 loss=0.0954
[epoch 5] step 29900/38082 loss=0.0840
[epoch 5] step 30000/38082 loss=0.1286
[epoch 5] step 30100/38082 loss=0.1461
[epoch 5] step 30200/38082 loss=0.1297
[epoch 5] step 30300/38082 loss=0.1516
[epoch 5] step 30400/38082 loss=0.1485
[epoch 5] step 30500/38082 loss=0.0758
[epoch 5] step 30600/38082 loss=0.1736
[epoch 5] step 30700/38082 loss=0.0602
[epoch 5] step 30800/38082 loss=0.1065
[epoch 5] step 30900/38082 loss=0.1601
[epoch 5] step 31000/38082 loss=0.0420
[epoch 5] step 31100/38082 loss=0.1523
[epoch 5] step 31200/38082 loss=0.2956
[epoch 5] step 31300/38082 loss=0.1455
[epoch 5] step 31400/38082 loss=0.0643
[epoch 5] step 31500/38082 loss=0.0638
[epoch 5] step 31600/38082 loss=0.0380
[epoch 5] step 31700/38082 loss=0.1057
[epoch 6] step 31800/38082 loss=0.0956
[epoch 6] step 31900/38082 loss=0.0561
[epoch 6] step 32000/38082 loss=0.0828
[epoch 6] step 32100/38082 loss=0.0993
[epoch 6] step 32200/38082 loss=0.0341
[epoch 6] step 32300/38082 loss=0.2090
[epoch 6] step 32400/38082 loss=0.1286
[epoch 6] step 32500/38082 loss=0.0830
[epoch 6] step 32600/38082 loss=0.0462
[epoch 6] step 32700/38082 loss=0.1281
[epoch 6] step 32800/38082 loss=0.1398
[epoch 6] step 32900/38082 loss=0.0551
[epoch 6] step 33000/38082 loss=0.0881
[epoch 6] step 33100/38082 loss=0.1620
[epoch 6] step 33200/38082 loss=0.1466
[epoch 6] step 33300/38082 loss=0.1056
[epoch 6] step 33400/38082 loss=0.0888
[epoch 6] step 33500/38082 loss=0.0773
[epoch 6] step 33600/38082 loss=0.1973
[epoch 6] step 33700/38082 loss=0.1010
[epoch 6] step 33800/38082 loss=0.1792
[epoch 6] step 33900/38082 loss=0.1371
[epoch 6] step 34000/38082 loss=0.0759
[epoch 6] step 34100/38082 loss=0.1308
[epoch 6] step 34200/38082 loss=0.0296
[epoch 6] step 34300/38082 loss=0.1352
[epoch 6] step 34400/38082 loss=0.0927
[epoch 6] step 34500/38082 loss=0.1195
[epoch 6] step 34600/38082 loss=0.0658
[epoch 6] step 34700/38082 loss=0.1430
[epoch 6] step 34800/38082 loss=0.0541
[epoch 6] step 34900/38082 loss=0.1294
[epoch 6] step 35000/38082 loss=0.1015
[epoch 6] step 35100/38082 loss=0.1085
[epoch 6] step 35200/38082 loss=0.1087
[epoch 6] step 35300/38082 loss=0.1384
[epoch 6] step 35400/38082 loss=0.0715
[epoch 6] step 35500/38082 loss=0.0718
[epoch 6] step 35600/38082 loss=0.0787
[epoch 6] step 35700/38082 loss=0.1313
[epoch 6] step 35800/38082 loss=0.0442
[epoch 6] step 35900/38082 loss=0.0537
[epoch 6] step 36000/38082 loss=0.1079
[epoch 6] step 36100/38082 loss=0.1474
[epoch 6] step 36200/38082 loss=0.1411
[epoch 6] step 36300/38082 loss=0.0793
[epoch 6] step 36400/38082 loss=0.0502
[epoch 6] step 36500/38082 loss=0.1673
[epoch 6] step 36600/38082 loss=0.0661
[epoch 6] step 36700/38082 loss=0.1411
[epoch 6] step 36800/38082 loss=0.0744
[epoch 6] step 36900/38082 loss=0.0405
[epoch 6] step 37000/38082 loss=0.0608
[epoch 6] step 37100/38082 loss=0.2125
[epoch 6] step 37200/38082 loss=0.1302
[epoch 6] step 37300/38082 loss=0.1064
[epoch 6] step 37400/38082 loss=0.2316
[epoch 6] step 37500/38082 loss=0.0919
[epoch 6] step 37600/38082 loss=0.1342
[epoch 6] step 37700/38082 loss=0.0769
[epoch 6] step 37800/38082 loss=0.0924
[epoch 6] step 37900/38082 loss=0.0404
"""

# Regex parse: epoch, step, loss
pattern = r"\[epoch (\d+)\] step (\d+)/\d+ loss=([\d\.]+)"
records = [(int(e), int(s), float(l)) for e, s, l in re.findall(pattern, log_text)]

df = pd.DataFrame(records, columns=["epoch", "step", "loss"])

# Determine output directory (the folder where this script lives)
OUT_DIR = os.path.dirname(__file__) or "."

# Plot 1: Loss curves per epoch with best fit line (all epochs combined)
plt.figure(figsize=(10, 6))
for epoch, subdf in df.groupby("epoch"):
    plt.plot(subdf["step"], subdf["loss"], linestyle="-", linewidth=1.5, label=f"Epoch {epoch}")

# Best fit line across all epochs
X = df["step"].values.reshape(-1, 1)
y = df["loss"].values
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
plt.plot(df["step"], y_pred, "k--", linewidth=2, label="Best Fit (all epochs)")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss Curves with Best Fit Line")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_curves_with_fit.png"), dpi=150)
plt.show()

# Plot 2: Boxplot of losses per epoch
plt.figure(figsize=(8, 6))
df.boxplot(column="loss", by="epoch")
plt.title("Loss Distribution per Epoch")
plt.suptitle("")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_boxplot_by_epoch.png"), dpi=150)
plt.show()

# Epoch-by-epoch stats
stats = []
for epoch, subdf in df.groupby("epoch"):
    mean = subdf["loss"].mean()
    std = subdf["loss"].std()
    cv = std/mean if mean!=0 else np.nan
    # Regression for slope & R²
    X = subdf["step"].values.reshape(-1,1)
    y = subdf["loss"].values
    reg = LinearRegression().fit(X,y)
    slope = reg.coef_[0]
    r2 = reg.score(X,y)
    stats.append((epoch, mean, std, cv, slope, r2))

stats_df = pd.DataFrame(stats, columns=["Epoch", "Mean Loss", "Std Dev", "CV", "Slope", "R²"])

# Save stats to JSON
stats_records = stats_df.to_dict(orient="records")
with open(os.path.join(OUT_DIR, "biencoder_epoch_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats_records, f, indent=2)

print("Saved:")
print(" -", os.path.join(OUT_DIR, "loss_curves_with_fit.png"))
print(" -", os.path.join(OUT_DIR, "loss_boxplot_by_epoch.png"))
print(" -", os.path.join(OUT_DIR, "biencoder_epoch_stats.json"))
