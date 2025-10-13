
from compute import compute_data
from plot import plot_all_data
from report import report_all_data


# ----------------- Main -----------------
def main(input_dir: str, output_dir: str):

    all_data = compute_data(input_dir)

    print("\n")
    print(all_data)
    print("\n")

    plot_all_data(all_data, output_dir)

    # report_dir = f'{output_dir}/report'
    # report_all_data(all_data, report_dir)


if __name__ == "__main__":
    main("./results", "./output")
