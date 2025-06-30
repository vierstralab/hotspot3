from hotspot3.io.paths import Hotspot3Paths
import networkx as nx

class StepsSolver:

    display_names ={
        'cutcounts': 'Extract cutcounts from BAM',
        'total_cutcounts': 'Calculate total cutcounts',
        'smoothed_signal': 'Smooth cutcounts',
        'fit_params': 'Fit background model',
        'pvals': 'Calculate p-values',
        'fdrs': 'Calculate FDRs',
        'normalized_density': 'Extract normalized density',
        'peak_calling': 'Call hotspots and peaks',
    }

    def __init__(self, paths: Hotspot3Paths, save_density):
        self.paths = paths
        self.save_density = save_density

    @staticmethod
    def resolve_required_steps(outputs, available, graph: nx.DiGraph):
        """
        Given a set of desired outputs and already available nodes,
        return the minimal set of required nodes (topologically sorted).
        """
        required = set()
        visited = set()

        def visit(node):
            if node in visited or node in available:
                return
            visited.add(node)
            for dep in graph.predecessors(node):
                visit(dep)
            required.add(node)

        for out in outputs:
            visit(out)

        return list(nx.topological_sort(graph.subgraph(required)))


    def find_missing_steps(self):
        step_graph = nx.DiGraph()
        step_graph.add_edges_from([
            ("bam", "cutcounts"),
            ("cutcounts", "total_cutcounts"),
            
            # signal smoothing
            ("cutcounts", "smoothed_signal"),
            ("total_cutcounts", "smoothed_signal"),

            # normalized density
            ("smoothed_signal", "normalized_density"),

            # background model fitting
            ("cutcounts", "fit_params"),
            ("total_cutcounts", "fit_params"),

            # per-bp p-values
            ("fit_params", "pvals"),
            ("cutcounts", "pvals"),

            # FDR
            ("pvals", "fdrs"),

            # peak calling
            ("fdrs", "peak_calling"),
            ("smoothed_signal", "peak_calling"),
            ("total_cutcounts", "peak_calling")
        ])
        available = {x for x in step_graph.nodes if self.paths.was_set(x)}
        outputs = {'peak_calling',}
        if self.save_density:
            outputs.add('normalized_density')
        result = self.resolve_required_steps(outputs, available, step_graph)
        if 'bam' in result:
            raise ValueError("Provide a bam file or cutcounts")
        return result
    
    def get_step_display_names(self, steps):
        return [self.display_names.get(step, step) for step in steps]