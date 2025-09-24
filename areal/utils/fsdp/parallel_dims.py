from dataclasses import dataclass

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from areal.platforms import current_platform


@dataclass
class FSDPParallelDims:
    dp: int
    sp: int
    tp: int
    ep: int
    etp: int
    world_size: int

    _world_mesh: DeviceMesh | None = None

    def __str__(self) -> str:
        return f"FSDPParallelDims(dp={self.dp}, sp={self.sp}, tp={self.tp}, ep={self.ep}, etp={self.etp}, world_size={self.world_size})"

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp, sp, tp, ep, etp = (
            self.dp,
            self.sp,
            self.tp,
            self.ep,
            self.etp,
        )
        for d in (sp, tp, ep, etp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp"

        assert dp == -1 or dp >= 1, "dp must -1 or >=1."
        if dp < 0:
            self.dp = dp = self.world_size // (sp * tp)
        assert dp >= 1

        assert (
            dp * sp * tp == self.world_size
        ), f"Invalid parallel dims: dp({dp}) * sp({sp}) * tp({tp}) != WORLD_SIZE({self.world_size})"

        if ep > 1:
            assert etp == tp or etp == 1, "Currently we only support ETP=TP or ETP=1"
            if etp == tp:
                # ep would borrow all sp and some dp degree
                assert ep % sp == 0 and (dp * sp) % ep == 0
            elif etp == 1:
                # ep would borrow all sp and tp and some dp degree
                assert ep % (sp * tp) == 0 and (dp * sp * tp) % ep == 0

    def build_mesh(self) -> DeviceMesh:
        if self.ep > 1:
            return self._build_mesh_with_ep()
        else:
            return self._build_mesh_without_ep()

    def _build_mesh_with_ep(self) -> DeviceMesh:
        # With ep, dp and ep are derived submeshes:
        # dp = dp_mod_ep * dp_in_ep
        if self.etp == self.tp:
            # ep = dp_in_ep * sp
            dp_mod_ep = self.dp * self.sp // self.ep
            dp_in_ep = self.ep // self.sp
        else:
            assert self.etp == 1
            # ep = dp_in_ep * sp * tp
            dp_mod_ep = self.dp * self.sp * self.tp // self.ep
            dp_in_ep = self.ep // (self.sp * self.tp)

        mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(dp_mod_ep, dp_in_ep, self.sp, self.tp),
            mesh_dim_names=("dp_mod_ep", "dp_in_ep", "sp", "tp"),
        )

        # Create all the submesh here for process groups
        # Guaranteed dims:
        #     root mesh: dp_mod_ep, dp_in_ep, sp, tp
        #     sub  mesh: dp, dp_sp, sp_tp, ep
        mesh["dp_mod_ep", "dp_in_ep"]._flatten(mesh_dim_name="dp")
        mesh["dp_mod_ep", "dp_in_ep", "sp"]._flatten(mesh_dim_name="dp_sp")
        mesh["sp", "tp"]._flatten(mesh_dim_name="sp_tp")
        ep_mesh_dim_names = (
            ("dp_in_ep", "sp", "tp") if self.etp == 1 else ("dp_in_ep", "sp")
        )
        mesh[tuple(ep_mesh_dim_names)]._flatten(mesh_dim_name="ep")

        return mesh

    def _build_mesh_without_ep(self) -> DeviceMesh:
        mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(self.dp, self.sp, self.tp),
            mesh_dim_names=("dp", "sp", "tp"),
        )

        # Create all the submesh here for process groups
        # Guaranteed dims:
        #     root mesh: dp, sp, tp
        #     sub  mesh: dp_sp, sp_tp
        mesh["dp", "sp"]._flatten(mesh_dim_name="dp_sp")
        mesh["sp", "tp"]._flatten(mesh_dim_name="sp_tp")

        return mesh

    @property
    def world_mesh(self) -> DeviceMesh:
        # doing late init so ParallelDims can still be used as a lightweight
        # dataclass without having to initialize the world mesh
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @property
    def dp_enabled(self) -> bool:
        return self.dp > 1

    @property
    def sp_enabled(self) -> bool:
        return self.sp > 1

    @property
    def tp_enabled(self) -> bool:
        return self.tp > 1

    @property
    def ep_enabled(self) -> bool:
        return self.ep > 1

    @property
    def etp_enabled(self) -> bool:
        return self.etp > 1

    @property
    def fsdp_gradient_divide_factor(self) -> int:
        # This is needed for FSDP-sharded experts when Expert Parallel is enabled.
        # Although the FSDP sharding of experts is done on a mesh of a different size than
        # other parameters, the gradient division factor should be consistent with data.
        return self.dp * self.sp

    @property
    def non_data_parallel_size(self) -> int:
        return self.sp * self.tp

    @property
    def seq_len_divisor(self) -> int:
        # Sequence Parallel requires that seq_len be divisible by TP degree.
        # Ulysses Sequence Parallel requires that seq_len be divisible by SP degree
        return self.tp * self.sp
