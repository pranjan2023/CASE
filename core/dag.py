import time

class ExperimentDAG:

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.latest = None

    def add_experiment(self, config_id, geometry, stats):

        node_id = config_id

        node_data = {
            "geometry": geometry,
            "mean": stats["mean"],
            "std": stats["std"],
            "n": stats["n"],
            "timestamp": time.time(),
            "run_id": len(self.nodes.get(node_id, []))
        }

        self.nodes.setdefault(node_id, []).append(node_data)

        if self.latest and self.latest != node_id:
            self.edges.setdefault(self.latest, []).append(node_id)

        self.latest = node_id

        return node_id

    def get_latest(self):
        node = self.nodes.get(self.latest)
        return node[-1] if node else None