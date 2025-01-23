# Identifying Shapes in a 6D Space

### Goal
The goal of this project was to identify all the shapes and patterns present in a 6D space. Might sound simple, but here's the catch: The shapes are of different dimensions, and there is no way to visualize a shape that is over 4 dimensions. So how did I find all the shapes? Let's go over that step-by-step.

### K-Means
Since each shape would essentially be multiple points clustered together, the first step I took was to perform K-Means clustering on the raw data. Through trial and error and adjusting the 'K' value multiple times, I concluded that there were 6 shapes.

### Visualizing through PCA
My next thought was to immediately throw PCA onto this data to visualize it. This made me realise something really important- PCA IS NOT JUST FOR VISUALIZING DATA. 

![image](https://github.com/user-attachments/assets/bc6b3be9-21b6-484a-9546-559e01523d7a)

If you see the image above, you can probably make out 1 - 2 shapes and be quite confident about them. I learned the hard way that this wasnt the right approach. Since PCA is a linear technique, and can only return 'n' number of linearly independant vectors, expecting it to project your 6D data down to 2D and make no mistake is quite literally impossible. So what did I do then?

### Identifying the shapes
Working with the entire data 
