from pathlib import Path
import networkx as nx

def fallback_path():
    def decorator(func):
        name = func.__name__
        private_name = f"_{name}"

        @property
        def wrapper(self):
            override = getattr(self, private_name, None)

            if override is not None:
                path = Path(override)
                if not path.exists():
                    raise FileNotFoundError(f"Provided path for `{name}` does not exist: {path}")
                result = path
            else:
                result = func(self)
                if result is None:
                    raise ValueError(f"Path for `{name}` is not set")
            return str(result)

        @wrapper.setter
        def wrapper(self, value):
            setattr(self, private_name, value)

        return wrapper
    return decorator


class Hotspot3Paths:
    def __init__(
            self,
            outdir,
            sample_id,
            save_density=False,
            **kwargs
        ):
        self.outdir = outdir
        self.sample_id = sample_id
        self.outdir = Path(self.outdir)

        allowed_keys = {
            k for k, v in self.__class__.__dict__.items()
        }
        unknown_keys = set(kwargs.keys()) - allowed_keys
        if len(unknown_keys) > 0:
            raise ValueError(f"Unknown keys provided: {unknown_keys}")
        for key, value in kwargs.items():
            self.__setattr__(f"_{key}", value)
            if key == "cutcounts" and value is not None:
                self._total_cutcounts = self.total_cutcounts
        
        self.solver = StepsSolver(self, save_density)
    
    def was_set(self, prop):
        return getattr(self, f"_{prop}", None) is not None

    @property
    def debug_dir(self):
        path = self.outdir / "debug"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @fallback_path()
    def bam(self):
        return None

    @fallback_path()
    def cutcounts(self):
        return self.outdir / f"{self.sample_id}.cutcounts.bed.gz"

    @property
    def total_cutcounts(self):
        if self.cutcounts is None:
            return None
        return self.cutcounts.replace('.cutcounts.bed.gz', '.total_cutcounts')

    @fallback_path()
    def smoothed_signal(self):
        return self.debug_dir / f"{self.sample_id}.smoothed_signal.parquet"

    @fallback_path()
    def fit_params(self):
        return self.debug_dir / f"{self.sample_id}.fit_params.parquet"

    @fallback_path()
    def pvals(self):
        return self.outdir / f"{self.sample_id}.pvals.parquet"

    @fallback_path()
    def fdrs(self):
        return self.debug_dir / f"{self.sample_id}.fdrs.parquet"

    @fallback_path()
    def normalized_density(self):
        return self.outdir / f"{self.sample_id}.normalized_density.bw"
    
    @fallback_path()
    def per_region_background(self):
        return self.debug_dir / f"{self.sample_id}.per_segment.background.bw"
    
    @fallback_path()
    def per_region_stats(self):
        return self.outdir / f"{self.sample_id}.fit_stats.bed.gz"
    
    @fallback_path()
    def thresholds(self):
        return self.debug_dir / f"{self.sample_id}.thresholds.bw"
    
    @fallback_path()
    def background(self):
        return self.outdir / f"{self.sample_id}.background.bw"

    def fdrs_dir(self, fdr):
        return self.outdir / f"fdr{fdr}"

    def hotspots(self, fdr):
        return str(self.fdrs_dir(fdr) / f"{self.sample_id}.hotspots.fdr{fdr}.bed.gz")
    
    def peaks(self, fdr):
        return str(self.fdrs_dir(fdr) / f"{self.sample_id}.peaks.fdr{fdr}.bed.gz")
    
    def hotspots_bb(self, fdr):
        return str(self.fdrs_dir(fdr) / f"{self.sample_id}.hotspots.fdr{fdr}.bb")
    
    def peaks_bb(self, fdr):
        return str(self.fdrs_dir(fdr) / f"{self.sample_id}.peaks.fdr{fdr}.bb")
    
    def find_missing_steps(self):
        return self.solver.find_missing_steps_from_paths()
    
    def get_display_names(self, steps):
        return self.solver.get_step_display_names(steps)


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


    def find_missing_steps_from_paths(self):
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