#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 


# In[2]:


num_transmission_lines = 120


# In[3]:


# df_line_loading = pd.read_excel('Y_norm.xlsx')
# df_line_loading = df_line_loading.iloc[:, 2:]
# print(df_line_loading.head())
#

# In[4]:


# y_norm = df_line_loading.to_numpy()
y_norm = np.load("Polish_y_norm.npy")
print("y_norm shape:", y_norm.shape)


# # In[5]:
#
# df_line_connections = pd.read_csv('lines_connections.csv')
# print(df_line_connections.head())
#
# # In[6]:
# df_line_connections = df_line_connections.iloc[:, :-2]
# display(df_line_connections)
#
# # In[7]:
#
# lines_data = list(df_line_connections[['# line', 'from bus', 'to bus']].itertuples(index=False, name=None))
# print(lines_data)
#
# # In[8]:
# # Create a function to find connections
# def find_connected_lines(lines_data):
#     connected_lines = {}
#
#     # Loop over each line
#     for i, (line_num1, from_bus1, to_bus1) in enumerate(lines_data):
#         connected_lines[line_num1] = []
#
#         # Compare with every other line
#         for j, (line_num2, from_bus2, to_bus2) in enumerate(lines_data):
#             if i != j:  # Skip comparing the line with itself
#                 # Check if any bus is common between two lines
#                 if (from_bus1 == from_bus2 or from_bus1 == to_bus2 or
#                     to_bus1 == from_bus2 or to_bus1 == to_bus2):
#                     connected_lines[line_num1].append(line_num2)
#
#     return connected_lines
#
# # In[9]:
#
# # Get the connected lines
# connected_lines = find_connected_lines(lines_data)
# # Print the connected lines
# #for line, connections in connected_lines.items():
# #    print(f"Line {line} is connected to lines: {connections}")
#
# # In[10]:
# connected_lines
# # In[11]:
# # Initialize a num_transmission_linesxnum_transmission_lines adjacency matrix with zeros
# adj_matrix = np.zeros((num_transmission_lines, num_transmission_lines), dtype=int)
#
# # Fill the adjacency matrix based on connected lines
# for line, connections in connected_lines.items():
#     for connected_line in connections:
#         adj_matrix[line - 1][connected_line - 1] = 1  # Subtract 1 for zero-based indexing
#         adj_matrix[connected_line - 1][line - 1] = 1  # Since the connection is undirected
#

adj_matrix = np.load("Polish_adj_matrix.npy")
# Print the adjacency matrix
print("Adjacency Matrix:")
print(adj_matrix)


# In[13]:


adj_matrix[0] # line 1 


# In[14]:


adj_matrix = np.array(adj_matrix)
print("adj_matrix shape:", adj_matrix.shape)


# Self connection 

# In[15]:


adj_matrix_with_identity = adj_matrix + np.eye(adj_matrix.shape[0])
print("adj_matrix_with_identity shape:", adj_matrix_with_identity.shape)


# In[16]:


adj_matrix_with_identity[0]


# Normalizing 

# In[34]:


degree = np.sum(adj_matrix, axis=1)


# In[35]:


degree


# In[36]:


d_inv_sqrt = np.power(degree, -0.5)


# In[38]:


d_mat_inv_sqrt = np.diag(d_inv_sqrt)  


# In[39]:


d_mat_inv_sqrt


# In[41]:


normalized_adj = np.dot(np.dot(d_mat_inv_sqrt, adj_matrix_with_identity), d_mat_inv_sqrt)


# In[42]:


normalized_adj[0]


# # save data 

# In[43]:


np.save('adj_matrix_with_identity.npy', y_norm)
np.save('adj_matrix.npy', adj_matrix)
np.save('normalized_adj_matrix.npy', normalized_adj)


# In[ ]:




