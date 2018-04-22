# BilinPL
This is an implemetation of BilinPL in java. 

Step 1:
Run ToyBilinPL to start Gateway server.

Step 2:
Run BilinPL.py to train the model


Data set contains four files namely instance_Scaled, labelFeatures_Scaled
ordering, rankOrders.
There are totally 435 instances (instance_Scaled.csv)
Totally 5000 labels which are represented as feature vector of length 12.
Ranking length is 10.
'0' is encoded for labels which is not considered in top 10 ranking. (rankOrders.csv)
Kendall's Tau-b is used for ranking measurment.
