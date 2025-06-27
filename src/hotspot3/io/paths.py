from pathlib import Path


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
            if key == "cutcounts":
                self._total_cutcounts = self.total_cutcounts
    
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
        return self.outdir / f"{fdr}"

    def hotspots(self, fdr):
        return self.fdrs_dir(fdr) / f"{self.sample_id}.hotspots.fdr{fdr}.bed.gz"
    
    def peaks(self, fdr):
        return self.fdrs_dir(fdr) / f"{self.sample_id}.peaks.fdr{fdr}.bed.gz"
    
    def hotspots_bb(self, fdr):
        return self.fdrs_dir(fdr) / f"{self.sample_id}.hotspots.fdr{fdr}.bb"
    
    def peaks_bb(self, fdr):
        return self.fdrs_dir(fdr) / f"{self.sample_id}.peaks.fdr{fdr}.bb"