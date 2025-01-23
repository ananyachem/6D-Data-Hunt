from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F
from pyspark.sql.functions import col

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

spark = SparkSession.builder.getOrCreate()

df = spark.read.option("delimiter", ",").csv("assignment4/space.dat", inferSchema=True)

df = df.withColumnRenamed("_c0", "x") \
       .withColumnRenamed("_c1", "y") \
       .withColumnRenamed("_c2", "z") \
       .withColumnRenamed("_c3", "u") \
       .withColumnRenamed("_c4", "v") \
       .withColumnRenamed("_c5", "w") 

# vector assembler will assemble the features into a single vector column
assembler = VectorAssembler(inputCols=["x", "y", "z", "u", "v", "w"], outputCol="features")
vector_df = assembler.transform(df)

# kmeans with 6 clusters (this file does not include how i decided on 5 clusters, i tested 
# it on a separate file and did not include it here since the code was getting too long)
kmeans = KMeans(featuresCol="features", k=6, seed=101)
model = kmeans.fit(vector_df)
clustered_df = model.transform(vector_df)

# getting the cluster centers 
print("\nCluster Centers")
for center in model.clusterCenters():
    print("Center:", center)

# Number of points in each cluster
print("\nNumber of points in each cluster")
cluster_count = clustered_df.groupBy("prediction").count()
sorted_clusters = cluster_count.orderBy("prediction")
sorted_clusters.show()

# centering the data
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=False, withMean=True)
scaler_model = scaler.fit(clustered_df)
clustered_df = scaler_model.transform(clustered_df)

# each cluster in its own df so we can plot it
cluster_1_df = clustered_df.filter(col("prediction") == 0).withColumn("cluster_label", col("prediction") + 1)
cluster_2_df = clustered_df.filter(col("prediction") == 1).withColumn("cluster_label", col("prediction") + 1)
cluster_3_df = clustered_df.filter(col("prediction") == 2).withColumn("cluster_label", col("prediction") + 1)
cluster_4_df = clustered_df.filter(col("prediction") == 3).withColumn("cluster_label", col("prediction") + 1)
cluster_5_df = clustered_df.filter(col("prediction") == 4).withColumn("cluster_label", col("prediction") + 1)
cluster_6_df = clustered_df.filter(col("prediction") == 5).withColumn("cluster_label", col("prediction") + 1)


# plotting each cluster individually

# CLUSTER 1
print("\nSHAPE 1")

# applying PCA
assembler = VectorAssembler(inputCols=["x", "y", "z", "u", "v", "w"], outputCol="shape_features")
cluster_1_vectors = assembler.transform(cluster_1_df)

pca = PCA(k=6, inputCol="shape_features", outputCol="pca_features")
pca_model = pca.fit(cluster_1_vectors)
pca_df = pca_model.transform(cluster_1_vectors)

# explained variance for each component to approximate the dimensions
explained_variance = pca_model.explainedVariance.toArray()
print("Explained Variance for each Component, Shape 1: ", explained_variance)

# calculating the standard deviation along each axis in Shape 1
std_dev_df = cluster_1_df.select(
    F.stddev("x").alias("std_x"),
    F.stddev("y").alias("std_y"),
    F.stddev("z").alias("std_z"),
    F.stddev("u").alias("std_u"),
    F.stddev("v").alias("std_v"),
    F.stddev("w").alias("std_w")
)
print("Standard deviation along each axis, Shape 1: ")
std_dev_df.show()

# since the points have almost identical standard deviations from each axis, 
# and each principal component contributes evenly to the shape, we can classify 
# shape 1 as a 6D Sphere.

# finding the diameter of the 6D sphere
pca_results = pca_df.select("pca_features").rdd.map(lambda row: row.pca_features.toArray()).collect()
pca_df_pd = pd.DataFrame(pca_results, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"])

pc1_min, pc1_max = pca_df_pd["PC1"].min(), pca_df_pd["PC1"].max()
pc2_min, pc2_max = pca_df_pd["PC2"].min(), pca_df_pd["PC2"].max()
pc3_min, pc3_max = pca_df_pd["PC3"].min(), pca_df_pd["PC3"].max()
pc4_min, pc4_max = pca_df_pd["PC4"].min(), pca_df_pd["PC4"].max()
pc5_min, pc5_max = pca_df_pd["PC5"].min(), pca_df_pd["PC5"].max()
pc6_min, pc6_max = pca_df_pd["PC6"].min(), pca_df_pd["PC6"].max()
l1 = abs(pc1_max - pc1_min)
l2 = abs(pc2_max - pc2_min)
l3 = abs(pc3_max - pc3_min)
l4 = abs(pc4_max - pc4_min)
l5 = abs(pc5_max - pc5_min)
l6 = abs(pc6_max - pc6_min)
print("Max lengths for each side, Shape 1: ", l1, l2, l3, l4, l5, l6)
avg_diameter = (l1+l2+l3+l4+l5+l6)/6
print("Diameter of the 6D Sphere, Shape 1, ", avg_diameter)


# CLUSTER 2
print("\nSHAPE 2")

# applying PCA
assembler = VectorAssembler(inputCols=["x", "y", "z", "u", "v", "w"], outputCol="shape_features")
cluster_2_vectors = assembler.transform(cluster_2_df)

pca = PCA(k=6, inputCol="shape_features", outputCol="pca_features")
pca_model = pca.fit(cluster_2_vectors)
pca_df = pca_model.transform(cluster_2_vectors)

# explained variance for each component to approximate the dimensions
explained_variance = pca_model.explainedVariance.toArray()
print("Explained Variance for each Component, Shape 2: ", explained_variance)
# only the first principal component contributes to this shape, so we can say it is 1 dimensional.

pca_results = pca_df.select("pca_features").rdd.map(lambda row: row.pca_features.toArray()).collect()
pca_df_pd = pd.DataFrame(pca_results, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]) 

pc1_min, pc1_max = pca_df_pd["PC1"].min(), pca_df_pd["PC1"].max()
length = abs(pc1_max - pc1_min)
print("Length of the Line, Shape 2: ", length)

# visualizing the shape
plt.scatter(pca_df_pd['PC1'], np.zeros_like(pca_df_pd['PC1']))
plt.title('1D Plot of shape 2')
plt.xlabel('PCA Component 1')
plt.ylabel('Index')
plt.grid()
plt.savefig("shape_2_1D.png")
print("Saved shape_2_1D.png to device")
plt.close()


# CLUSTER 3
print("\nSHAPE 3")

# applying PCA
assembler = VectorAssembler(inputCols=["x", "y", "z", "u", "v", "w"], outputCol="shape_features")
cluster_3_vectors = assembler.transform(cluster_3_df)

pca = PCA(k=6, inputCol="shape_features", outputCol="pca_features")
pca_model = pca.fit(cluster_3_vectors)
pca_df = pca_model.transform(cluster_3_vectors)

# explained variance for each component to approximate the dimensions
explained_variance = pca_model.explainedVariance.toArray()
print("Explained Variance for each Component, Shape 3: ", explained_variance)
# The first two principal components contribute most to this shape, so we can say it is 2 dimensional.

pca_results = pca_df.select("pca_features").rdd.map(lambda row: row.pca_features.toArray()).collect()
pca_df_pd = pd.DataFrame(pca_results, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]) 

pc1_min, pc1_max = pca_df_pd["PC1"].min(), pca_df_pd["PC1"].max()
length = abs(pc1_max - pc1_min)
print("Length of the Line, Shape 3: ", length)

# visualizing the shape
plt.scatter(pca_df_pd['PC1'], pca_df_pd['PC2'])
plt.title('2D Plot of shape 3')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid()
plt.savefig("shape_3_2D.png")
print("Saved shape_3_2D.png to device")
plt.close()

# finding the size of the rectangle
# setting a manual min value because of the one extra point
pc1_min, pc1_max = -155, pca_df_pd["PC1"].max()  
pc2_min, pc2_max = -46, pca_df_pd["PC2"].max()

length = abs(pc1_min - pc1_max)
breadth = abs(pc2_min - pc2_max)
print("Length and breadth of the Rectangle, Shape 3: ", length, breadth)


# CLUSTER 4
print("\nSHAPE 4")

# applying PCA
assembler = VectorAssembler(inputCols=["x", "y", "z", "u", "v", "w"], outputCol="shape_features")
cluster_4_vectors = assembler.transform(cluster_4_df)

pca = PCA(k=6, inputCol="shape_features", outputCol="pca_features")
pca_model = pca.fit(cluster_4_vectors)
pca_df = pca_model.transform(cluster_4_vectors)

# explained variance for each component to approximate the dimensions
explained_variance = pca_model.explainedVariance.toArray()
print("Explained Variance for each Component, Shape 2: ", explained_variance)
# The first three principal components contribute most to this shape, so we can say it is 3 dimensional.

pca_results = pca_df.select("pca_features").rdd.map(lambda row: row.pca_features.toArray()).collect()
pca_df_pd = pd.DataFrame(pca_results, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]) 

# visualizing the shape
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=15, azim=15)  
scatter = ax.scatter(pca_df_pd['PC1'], pca_df_pd['PC2'], pca_df_pd['PC3'])
ax.set_title('3D Plot of Shape 4')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
plt.grid()
plt.savefig("shape_4_3D.png")
print("Saved shape_4_3D.png to device")
plt.close()

# finding the size of the cuboid
pc1_min, pc1_max = pca_df_pd["PC1"].min(), pca_df_pd["PC1"].max()
pc2_min, pc2_max = pca_df_pd["PC2"].min(), pca_df_pd["PC2"].max()
pc3_min, pc3_max = pca_df_pd["PC3"].min(), pca_df_pd["PC3"].max()

height = abs(pc3_min - pc3_max)
print("Height of the Cuboid, Shape 4: ", height)


# CLUSTER 5
print("\nSHAPE 5")

# applying PCA
assembler = VectorAssembler(inputCols=["x", "y", "z", "u", "v", "w"], outputCol="shape_features")
cluster_5_vectors = assembler.transform(cluster_5_df)

pca = PCA(k=6, inputCol="shape_features", outputCol="pca_features")
pca_model = pca.fit(cluster_5_vectors)
pca_df = pca_model.transform(cluster_5_vectors)

# explained variance for each component to approximate the dimensions
explained_variance = pca_model.explainedVariance.toArray()
print("Explained Variance for each Component, Shape 5: ", explained_variance)
# The first two principal components contribute most to this shape, so we can say it is 2 dimensional.

pca_results = pca_df.select("pca_features").rdd.map(lambda row: row.pca_features.toArray()).collect()
pca_df_pd = pd.DataFrame(pca_results, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]) 

# visualizing the shape
plt.scatter(pca_df_pd['PC1'], pca_df_pd['PC2'])
plt.title('2D Plot of shape 5')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid()
plt.savefig("shape_5_2D.png")
print("Saved shape_5_2D.png to device")
plt.close()

# finding the size of the ellipse
pc2_min, pc2_max = pca_df_pd["PC2"].min(), pca_df_pd["PC2"].max()
length = abs(pc2_max - pc2_min)
print("Length the Ellipse, Shape 5: ", length)


# CLUSTER 6
print("\nSHAPE 6")

# applying PCA
assembler = VectorAssembler(inputCols=["x", "y", "z", "u", "v", "w"], outputCol="shape_features")
cluster_6_vectors = assembler.transform(cluster_6_df)

pca = PCA(k=6, inputCol="shape_features", outputCol="pca_features")
pca_model = pca.fit(cluster_6_vectors)
pca_df = pca_model.transform(cluster_6_vectors)

# explained variance for each component to approximate the dimensions
explained_variance = pca_model.explainedVariance.toArray()
print("Explained Variance for each Component, Shape 6: ", explained_variance)

# calculating the standard deviation along each axis in Shape 6
std_dev_df = cluster_6_df.select(
    F.stddev("x").alias("std_x"),
    F.stddev("y").alias("std_y"),
    F.stddev("z").alias("std_z"),
    F.stddev("u").alias("std_u"),
    F.stddev("v").alias("std_v"),
    F.stddev("w").alias("std_w")
)
print("Standard deviation along each axis, Shape 6: ")
std_dev_df.show()

# finding the length of each side
pca_results = pca_df.select("pca_features").rdd.map(lambda row: row.pca_features.toArray()).collect()
pca_df_pd = pd.DataFrame(pca_results, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"])

pc1_min, pc1_max = pca_df_pd["PC1"].min(), pca_df_pd["PC1"].max()
pc2_min, pc2_max = pca_df_pd["PC2"].min(), pca_df_pd["PC2"].max()
pc3_min, pc3_max = pca_df_pd["PC3"].min(), pca_df_pd["PC3"].max()
pc4_min, pc4_max = pca_df_pd["PC4"].min(), pca_df_pd["PC4"].max()
pc5_min, pc5_max = pca_df_pd["PC5"].min(), pca_df_pd["PC5"].max()
pc6_min, pc6_max = pca_df_pd["PC6"].min(), pca_df_pd["PC6"].max()
l1 = abs(pc1_max - pc1_min)
l2 = abs(pc2_max - pc2_min)
l3 = abs(pc3_max - pc3_min)
l4 = abs(pc4_max - pc4_min)
l5 = abs(pc5_max - pc5_min)
l6 = abs(pc6_max - pc6_min)
print("Max lengths for each side, Shape 6: ", l1, l2, l3, l4, l5, l6)
avg_length = (l1+l2+l3+l4+l5+l6)/6
print("Length of the sides, Shape 6, ", avg_length)