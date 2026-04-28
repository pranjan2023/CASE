from core.dag import ExperimentDAG


def main():

    dag = ExperimentDAG()

    config1 = {"shape": "cube", "size": 10}

    exp1 = dag.create_experiment(config1)

    print("Experiment:", exp1)

    config2 = {"shape": "cube", "size": 12}

    exp2 = dag.create_experiment(config2, parent=exp1)

    print("Experiment:", exp2)

    print("Latest:", dag.latest())

    print("Node data:", dag.get(exp1))


if __name__ == "__main__":
    main()
