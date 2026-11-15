python src/train_cnn.py

Epoch 01/15 | Train: 0.2205 | Val: 0.8659 | Val F1: 0.8578
Epoch 02/15 | Train: 0.1301 | Val: 0.2627 | Val F1: 0.9327
Epoch 03/15 | Train: 0.1102 | Val: 0.1397 | Val F1: 0.9655
Epoch 04/15 | Train: 0.1038 | Val: 0.3035 | Val F1: 0.9238
Epoch 05/15 | Train: 0.0918 | Val: 0.2839 | Val F1: 0.9361
Epoch 06/15 | Train: 0.0914 | Val: 0.1749 | Val F1: 0.9602
Epoch 07/15 | Train: 0.0828 | Val: 0.1421 | Val F1: 0.9548
Epoch 08/15 | Train: 0.0768 | Val: 0.1782 | Val F1: 0.9603
Epoch 09/15 | Train: 0.0696 | Val: 0.2310 | Val F1: 0.9521
Epoch 10/15 | Train: 0.0660 | Val: 0.1132 | Val F1: 0.9643
Epoch 11/15 | Train: 0.0651 | Val: 0.0946 | Val F1: 0.9773
Epoch 13/15 | Train: 0.0611 | Val: 0.1443 | Val F1: 0.9651
Epoch 14/15 | Train: 0.0558 | Val: 0.1325 | Val F1: 0.9628
Epoch 15/15 | Train: 0.0568 | Val: 0.0834 | Val F1: 0.9780

Split: Validation | Preprocess: original | ClassWeights: False | Augment: False
Accuracy:  0.9676
Precision: 0.9869
Recall:    0.9692
F1:        0.9780
Confusion matrix:
[[130   5]
 [ 12 377]]

---

python src/train_cnn.py --preprocess clahe

Epoch 01/15 | Train: 0.2175 | Val: 0.9048 | Val F1: 0.8521
Epoch 02/15 | Train: 0.1212 | Val: 0.1508 | Val F1: 0.9662
Epoch 03/15 | Train: 0.1028 | Val: 0.2715 | Val F1: 0.9207
Epoch 04/15 | Train: 0.0982 | Val: 0.1140 | Val F1: 0.9708
Epoch 05/15 | Train: 0.0874 | Val: 0.0965 | Val F1: 0.9747
Epoch 06/15 | Train: 0.0788 | Val: 0.0915 | Val F1: 0.9758
Epoch 07/15 | Train: 0.0780 | Val: 0.1233 | Val F1: 0.9558
Epoch 08/15 | Train: 0.0730 | Val: 0.0753 | Val F1: 0.9845
Epoch 09/15 | Train: 0.0639 | Val: 0.2223 | Val F1: 0.9430
Epoch 10/15 | Train: 0.0611 | Val: 0.0632 | Val F1: 0.9858
Epoch 11/15 | Train: 0.0585 | Val: 0.0898 | Val F1: 0.9761
Epoch 12/15 | Train: 0.0581 | Val: 0.0876 | Val F1: 0.9773
Epoch 13/15 | Train: 0.0579 | Val: 0.0687 | Val F1: 0.9833
Epoch 14/15 | Train: 0.0512 | Val: 0.0703 | Val F1: 0.9793
Epoch 15/15 | Train: 0.0468 | Val: 0.0688 | Val F1: 0.9796

Split: Validation | Preprocess: clahe | ClassWeights: False | Augment: False
Accuracy:  0.9790
Precision: 0.9871
Recall:    0.9846
F1:        0.9858
Confusion matrix:
[[130   5]
 [  6 383]]

---

>> python src/train_cnn.py --preprocess hist_eq

Epoch 01/15 | Train: 0.2392 | Val: 1.1655 | Val F1: 0.8521
Epoch 02/15 | Train: 0.1386 | Val: 0.1694 | Val F1: 0.9671
Epoch 03/15 | Train: 0.1134 | Val: 0.3920 | Val F1: 0.9026
Epoch 04/15 | Train: 0.0994 | Val: 0.2184 | Val F1: 0.9407
Epoch 05/15 | Train: 0.0890 | Val: 0.1354 | Val F1: 0.9651
Epoch 06/15 | Train: 0.0867 | Val: 0.3803 | Val F1: 0.9229
Epoch 07/15 | Train: 0.0844 | Val: 0.1050 | Val F1: 0.9766
Epoch 08/15 | Train: 0.0807 | Val: 0.1245 | Val F1: 0.9650
Epoch 09/15 | Train: 0.0706 | Val: 0.2541 | Val F1: 0.9317
Epoch 10/15 | Train: 0.0672 | Val: 0.1560 | Val F1: 0.9616
Epoch 11/15 | Train: 0.0652 | Val: 0.0902 | Val F1: 0.9770
Epoch 12/15 | Train: 0.0596 | Val: 0.1681 | Val F1: 0.9592
Epoch 13/15 | Train: 0.0613 | Val: 0.0815 | Val F1: 0.9821
Epoch 14/15 | Train: 0.0544 | Val: 0.0897 | Val F1: 0.9760
Epoch 15/15 | Train: 0.0564 | Val: 0.1141 | Val F1: 0.9638

Split: Validation | Preprocess: hist_eq | ClassWeights: False | Augment: False
Accuracy:  0.9733
Precision: 0.9747
Recall:    0.9897
F1:        0.9821
Confusion matrix:
[[125  10]
 [  4 385]]

---

python src/train_cnn.py --preprocess clahe --epochs 25 --lr 5e-4 --dropout 0.35 --use-class-weights --augment --eval-test
Epoch 01/25 | Train: 0.4450 | Val: 1.3150 | Val F1: 0.8644
Epoch 02/25 | Train: 0.3018 | Val: 0.2662 | Val F1: 0.9433
Epoch 03/25 | Train: 0.2625 | Val: 0.1950 | Val F1: 0.9369
Epoch 04/25 | Train: 0.2380 | Val: 0.1620 | Val F1: 0.9409
Epoch 05/25 | Train: 0.2149 | Val: 0.1711 | Val F1: 0.9496
Epoch 06/25 | Train: 0.2179 | Val: 0.2007 | Val F1: 0.9528
Epoch 07/25 | Train: 0.2260 | Val: 0.1617 | Val F1: 0.9579
Epoch 08/25 | Train: 0.2163 | Val: 0.1519 | Val F1: 0.9592
Epoch 09/25 | Train: 0.2013 | Val: 0.1094 | Val F1: 0.9256
Epoch 10/25 | Train: 0.1931 | Val: 0.1471 | Val F1: 0.9661
Epoch 11/25 | Train: 0.1867 | Val: 0.1557 | Val F1: 0.9716
Epoch 12/25 | Train: 0.1696 | Val: 0.1203 | Val F1: 0.9506
Epoch 13/25 | Train: 0.1853 | Val: 0.1107 | Val F1: 0.9378
Epoch 14/25 | Train: 0.1671 | Val: 0.0971 | Val F1: 0.9534
Epoch 15/25 | Train: 0.1854 | Val: 0.1404 | Val F1: 0.9742
Epoch 16/25 | Train: 0.1619 | Val: 0.0925 | Val F1: 0.9448
Epoch 17/25 | Train: 0.1584 | Val: 0.1702 | Val F1: 0.9744
Epoch 18/25 | Train: 0.1612 | Val: 0.1064 | Val F1: 0.9699
Epoch 19/25 | Train: 0.1580 | Val: 0.2168 | Val F1: 0.9732
Epoch 20/25 | Train: 0.1561 | Val: 0.0861 | Val F1: 0.9657
Epoch 21/25 | Train: 0.1427 | Val: 0.1060 | Val F1: 0.9767
Epoch 22/25 | Train: 0.1422 | Val: 0.0987 | Val F1: 0.9780
Epoch 23/25 | Train: 0.1361 | Val: 0.1086 | Val F1: 0.9768
Epoch 24/25 | Train: 0.1526 | Val: 0.0796 | Val F1: 0.9545
Epoch 25/25 | Train: 0.1538 | Val: 0.0909 | Val F1: 0.9793

Split: Validation | Preprocess: clahe | ClassWeights: True | Augment: True
Accuracy:  0.9351
Precision: 0.9944
Recall:    0.9177
F1:        0.9545
Confusion matrix:
[[133   2]
 [ 32 357]]

Split: Test | Preprocess: clahe
Accuracy:  0.9038
Precision: 0.8986
Recall:    0.9538
F1:        0.9254
Confusion matrix:
[[192  42]
 [ 18 372]]