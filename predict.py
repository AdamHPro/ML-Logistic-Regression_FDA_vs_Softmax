import numpy as np

def predict(X, projected_centroid, W):
    """Apply the trained LDA classifier on the test data
    X: test data
    projected_centroid: centroid vectors of each class projected to the new space
    W: projection matrix computed by LDA
    """

    # Project test data onto the LDA space defined by W
    projected_data = np.dot(X, W)
    label = []
    for x in projected_data:
        classe = 0
        distance = np.inner(x - projected_centroid[0], x - projected_centroid[0])
        for i in range(1, len(projected_centroid)) :
            new_dist = np.inner(x - projected_centroid[1], x - projected_centroid[1])
            if distance > new_dist :
                classe = i
                distance = new_dist
        label.append(classe)
  
    # Return the predicted labels of the test data X
    return np.array(label)
