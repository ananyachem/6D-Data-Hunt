# Identifying Shapes in a 6D Space

### Goal
The goal of this project was to identify all the shapes and patterns present in a 6D space. Might sound simple, but here's the catch: The shapes are of different dimensions, and there is no way to visualize a shape that is over 4 dimensions. So how did I find all the shapes? Let's go over that step-by-step.

### K-Means
Since each shape would essentially be multiple points clustered together, the first step I took was to perform K-Means clustering on the raw data. Through trial and error and adjusting the 'K' value multiple times, I concluded that there were 6 shapes.

### Visualizing through PCA
My next thought was to immediately throw PCA onto this data to visualize it. This made me realise something really important- PCA IS NOT JUST FOR VISUALIZING DATA. 

### 2D Plot
![image](https://github.com/user-attachments/assets/bc6b3be9-21b6-484a-9546-559e01523d7a)

### 3D Plot
![image](https://github.com/user-attachments/assets/184fa6f0-e3f0-4b71-b20f-7c276b6cac46)

If you see the images above, you can probably make out 1 - 2 shapes and be quite confident about them. I learned the hard way that this wasnt the right approach. Since PCA is a linear technique, and can only return 'n' number of linearly independant vectors, expecting it to project your 6D data down to 2D and make no mistake is quite literally impossible. So what did I do then?

### Identifying the shapes
I realized that working with the entire data here was the problem. Instead, what I did was, I extracted all the point from each cluster and inspected those individually. In addition to performing PCA on each shape, I also observed the Explained Variance Ratio.

### Chain of thought for Shape 1: 
The explained variance ratio for each principal Component was almost identical ~15% from each dimension. This indicates that this might in fact be a 6D shape, leaning towards a spherical or cubical shape. I plotted a histogram of the Principal Components, which then let me confirm that this shape is infact a 6D Cube.

<img width="500" alt="Screenshot 2025-01-23 at 2 19 36 PM" src="https://github.com/user-attachments/assets/76932151-b57c-4943-95a9-572668eb3c24" />

Similarly, I was able to identify all of the 6 shapes.

### Shape 2 was a Line.

![shape_2_1D](https://github.com/user-attachments/assets/01a3abbf-4716-4995-af64-91edb7ee5e1a)

### Shape 3 was a Rectangle.

![shape_3_2D](https://github.com/user-attachments/assets/b51aa0ef-980a-4a26-ac83-5aded1b589f5)

### Shape 4 was a Cuboid.

![shape_4_3D](https://github.com/user-attachments/assets/736bfbce-a188-4801-8621-8d3491f46588)

### Shape 5:
This was a actually a circle. The reason it appears to be an ellipse is because of the tiny dot you see in the corner. This dot is distorting the figure, since it might be a dot in a higher dimension, and PCA needs to come up with 2 vectors that explains the entire shape as closely as possible. 

![shape_5_2D](https://github.com/user-attachments/assets/f302aac7-35b4-4fe9-b8cf-44416bc4f645)

### Shape 6: 
This was just the 6D coordinates drawn out. Plotting it out in 2D and 3D was not useful whatsoever, since projecting down the image made it look like some sort of an asterisk shape. But, if you take a quick look at the data, you can see that each row is essentially just 1D, so plotting just the raw data was more than enough.

With this, I concluded my project.
