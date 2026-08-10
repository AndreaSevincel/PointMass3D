#Optional Weights & Biases tracking.

#Every call is a no-op unless tracking is explicitly enabled AND wandb imports
#cleanly, so a missing install or a logged-out cluster degrades to plain stdout
#logging instead of killing a long run.

import os


class Run:
    #Thin wrapper so callers never branch on whether tracking is live.

    def __init__(self, run=None):
        self._run = run

    @property
    def active(self):
        return self._run is not None

    @property
    def url(self):
        return getattr(self._run, "url", None)

    def log(self, data, step=None):
        if self._run is not None:
            self._run.log(data, step=step)

    def summary(self, **kv):
        if self._run is not None:
            self._run.summary.update(kv)

    def finish(self):
        if self._run is not None:
            self._run.finish()


def init(
    enabled=False,
    project="pointmass3d-flow",
    entity=None,
    name=None,
    group=None,
    tags=None,
    offline=False,
    config=None,
):
    #Return a Run; inactive whenever tracking is off or unavailable.
    if not enabled:
        return Run()
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed, continuing untracked (pip install wandb)")
        return Run()

    #A bare `wandb/` output directory in the cwd is importable as a NAMESPACE
    #package, so the ImportError above does not fire when wandb is merely
    #missing -- the import succeeds and yields a module with no attributes. The
    #symptom is an unhelpful "no attribute 'init'" from the call below, which
    #reads as a version problem rather than a missing install. Name the real
    #cause instead: runs write ./wandb/, so any repo that has ever logged has
    #the shadowing directory sitting in it.
    if not hasattr(wandb, "init"):
        print(f"[wandb] imported {getattr(wandb, '__file__', None) or wandb.__path__}"
              " but it has no init(); this is the local wandb/ output directory"
              " shadowing the package. Install it in THIS interpreter"
              " (pip install wandb). Continuing untracked")
        return Run()

    if offline:
        os.environ["WANDB_MODE"] = "offline"
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            tags=tags,
            config=config or {},
        )
    except Exception as e:  # auth failure, no network, quota, ...
        print(f"[wandb] init failed ({type(e).__name__}: {e}), continuing untracked")
        return Run()
    return Run(run)


def add_args(ap):
    #Attach the standard tracking flags to an ArgumentParser.
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--wandb-project", type=str, default="pointmass3d-flow")
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-name", type=str, default=None, help="run name (default: auto)")
    ap.add_argument("--wandb-group", type=str, default=None,
                    help="group runs, e.g. the name of an experiment grid")
    ap.add_argument("--wandb-tags", type=str, default=None, help="comma-separated")
    ap.add_argument("--wandb-offline", action="store_true",
                    help="log to disk only; sync later with `wandb sync`")
    return ap


def from_args(args, name=None, config=None):
    tags = args.wandb_tags.split(",") if args.wandb_tags else None
    return init(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or name,
        group=args.wandb_group,
        tags=tags,
        offline=args.wandb_offline,
        config=config,
    )
