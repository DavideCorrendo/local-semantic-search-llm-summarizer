import pickle

def save_index(path, vectors, graph, node_levels, state, chunks, fingerprint):
    data = {
        "vectors": vectors,
        "graph": graph,
        "node_levels": node_levels,
        "state": state,
        "chunks": chunks,
        "fingerprint": fingerprint,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_index(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return (
        data["vectors"],
        data["graph"],
        data["node_levels"],
        data["state"],
        data["chunks"],
        data["fingerprint"],
    )