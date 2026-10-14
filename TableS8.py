"""TableS8: GroMo25 leaf-count associations under four SSA reference controls."""

from src.models import model_paths, set_global_seed
from src.reference import reference_associations
from src.tables import save_table, table_parser


def main():
    parser = table_parser(__doc__)
    parser.add_argument("--datasets", nargs="+", default=["mustard", "okra", "radish", "wheat"])
    args = parser.parse_args()
    set_global_seed(42)
    paths = model_paths(args.models)
    rows = []
    for crop in args.datasets:
        rows.extend(
            reference_associations(crop, paths, args.config, args.seeds, args.device, traits=("leaf_count",))
        )
    save_table(
        rows, "TableS8", args.output_dir, ["species", "method", "metric"], ["delta_corr_abs", "distance_corr"]
    )


if __name__ == "__main__":
    main()
