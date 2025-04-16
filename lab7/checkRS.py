from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]

from sklearn.cluster import KMeans

for randomstate in range(0, 43):

    models = KMeans(n_clusters=10, n_init=10)
    models.fit(X)

    y_predicted = models.predict(X)

    from sklearn.metrics import confusion_matrix

    conf_matrix = confusion_matrix(y, y_predicted)

    import numpy as np

    argmaxes = np.argmax(conf_matrix, axis=1)
    unique_sorted = sorted(set(argmaxes))
    print(randomstate, len(unique_sorted))
    
    if len(unique_sorted) == 10:
        print(randomstate)
        break