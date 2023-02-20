
import causalgraphicalmodels
import daft
import matplotlib.pyplot as plt


def draw_dag(dag: causalgraphicalmodels.CausalGraphicalModel, coordinates: dict) -> None:
    assert not set(dag.dag.nodes) - coordinates.keys(), "All nodes should have defined coordinates"

    pgm = daft.PGM()
    for node in dag.dag.nodes:
        pgm.add_node(node, node, *coordinates[node])
    for edge in dag.dag.edges:
        pgm.add_edge(*edge)
    with plt.rc_context({"figure.constrained_layout.use": False}):
        pgm.render()
    plt.gca().invert_yaxis()
