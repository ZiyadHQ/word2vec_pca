import math
import numpy
import plotly.express as ex

embedding_dict = {}
with open("glove_w2v.txt", "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = list(map(float, values[1:]))
        if len(vector) != 200:
            print(f"ERROR: vector size is not 200, instead: {len(vector)}, named: {word}")
            break
        embedding_dict[word] = vector

def find_distance(vec_a, vec_b):
    
    length = 0
    for i in range(len(vec_a)):
        length += math.pow(vec_a[i] - vec_b[i], 2)
    length = math.sqrt(length)
    
    return length

def add_vec(vec_a, vec_b):
    return [vec_a[i] + vec_b[i] for i in range(len(vec_a))]

def subtract_vec(vec_a, vec_b):
    return [vec_a[i] - vec_b[i] for i in range(len(vec_a))]    

def get_length(vec_a):
    return math.sqrt(sum(x**2 for x in vec_a))

def find_variance(embedding, dimension_index):
    values = [vector[dimension_index] for vector in embedding.values()]
    
    variance = numpy.var(values)
    return variance

# finds n dimensions with the most variance in their data, useful for PCA
def find_top_n_dimensions(embedding, n):
    variances = [ [find_variance(embedding=embedding, dimension_index=i), i] for i in range(200)]
    variances.sort(reverse=True)
    return [dim for _, dim in variances[0 : n]]

# prepares the data for PCA plotting
def prepare_data_points(embedding, dimensions):
    top_dims = find_top_n_dimensions(embedding=embedding, n=dimensions)
    data_points = []
    for vector in embedding.items():
        points = [vector[1][dim] for dim in top_dims]
        word = vector[0]
        data_points.append((word, points))
    
    return data_points

print("prepared the embedding data...")

data = prepare_data_points(embedding=embedding_dict, dimensions=2)

print("found the top n dimensions...")

tokens = [point[0] for point in data]

x = [point[1][0] for point in data]
y = [point[1][1] for point in data]

# Create a DataFrame for Plotly (optional but easier for labeling)
import pandas as pd
df = pd.DataFrame({
    'Token': tokens,
    'Dimension 1': x,
    'Dimension 2': y,
})

print("prepared the data frame...")

fig = ex.scatter(
    df, x='Dimension 1', y='Dimension 2',
    text='Token', title='Word embeddings for word2vec Glove'
)

fig.update_traces(textposition='top center', marker=dict(size=8))
fig.update_layout(height=600, width=800)

fig.show()
