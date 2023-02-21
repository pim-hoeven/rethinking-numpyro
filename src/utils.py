
import causalgraphicalmodels
import daft
import numpyro.diagnostics
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


def print_filtered_summary(samples, prob=0.9, group_by_chain=True):
    filtered_samples = {name: values for name, values in samples.items() if len(values.shape) == 1}
    numpyro.diagnostics.print_summary(filtered_samples, prob=prob, group_by_chain=group_by_chain)

