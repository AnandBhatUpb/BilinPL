This is an implemetation of BilinPL in java. 

Step 1:
Run ToyBilinPL to start Gateway server.

Step 2:
Run BilinPL.py to train the model

1. Data set contains four files namely instance_Scaled, labelFeatures_Scaled
    ordering, rankOrders.
2. There are totally 435 instances (instance_Scaled.csv)
3. Totally 5000 labels which are represented as feature vector of length 12.
4. Ranking length is 10.
5. '0' is encoded for labels which is not considered in top 10 ranking. (rankOrders.csv)
6. Kendall's Tau-b is used for ranking measurment.  
