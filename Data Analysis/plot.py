
import os
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PLATFORM_COLORS = {'kv260': '#ED1C24', 'gpu': '#76B900'}


# ----------------- Plotting Functions -----------------
def plot_accuracy(all_results: Dict, platforms, output_path: str):

    tests = sorted(all_results.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    # Cria posições para cada barra
    positions = []
    labels = []
    values = []
    colors = []

    # Se for string, converte para lista de 1 elemento
    if isinstance(platforms, str):
        platforms = [platforms]

    for p in platforms:
        for idx, tname in enumerate(tests):
            metrics = all_results[tname][p]
            model_ids = metrics['model_ids']
            for mid in model_ids:
                positions.append(len(positions))
                labels.append(f'{tname}-M{mid}-{p.upper()}')
                values.append(metrics['accuracy'][mid][0] * 100)
                colors.append(PLATFORM_COLORS[p])

    ax.bar(positions, values, color=colors, alpha=0.7)
    ax.set_ylabel('Accuracy (%)')

    if len(platforms) == 1:
        ax.set_title(f'Accuracy per Model for {platforms[0].upper()}')
    else:
        ax.set_title(f'Accuracy per Model for {" vs ".join([p.upper() for p in platforms])}')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # legenda manual por plataforma
    if len(platforms) > 1:
        handles = [mpatches.Patch(color=PLATFORM_COLORS[p], label=p.upper()) for p in platforms]
        ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f'[DEBUG] Accuracy plot saved to {output_path}')


def plot_time_breakdown(all_results: Dict, platforms, output_path: str):

    if isinstance(platforms, str):
        platforms = [platforms]

    stage_mapping = {
        "load_time":            "Load 1",
        "load_time_1":          "Load 1",
        "load_time_2":          "Load 2",

        "model_load_time_1":    "Model Load 1",
        "model_load_time_2":    "Model Load 2",

        "preprocess_time":      "Preprocess 1",
        "preprocess_time_1":    "Preprocess 1",
        "preprocess_time_2":    "Preprocess 2",

        "upload_time":          "Upload 1",
        "upload_time_1":        "Upload 1",
        "upload_time_2":        "Upload 2",

        "inference_time":       "Inference 1",
        "inference_time_1":     "Inference 1",
        "inference_time_2":     "Inference 2",

        "download_time":        "Download 1",
        "download_time_1":      "Download 1",
        "download_time_2":      "Download 2",

        "postprocess_time":     "Postprocess 1",
        "postprocess_time_1":   "Postprocess 1",
        "postprocess_time_2":   "Postprocess 2",
    }

    stage_colors = {
        # Load → azuis
        "Load 1":        "navy",
        "Load 2":        "deepskyblue",

        # Model Load → laranjas
        "Model Load 1":  "darkorange",
        "Model Load 2":  "moccasin",

        # Preprocess → verdes
        "Preprocess 1":  "forestgreen",
        "Preprocess 2":  "palegreen",

        # Upload → roxos
        "Upload 1":      "indigo",
        "Upload 2":      "mediumpurple",

        # Inference → vermelhos
        "Inference 1":   "firebrick",
        "Inference 2":   "salmon",

        # Download → castanhos
        "Download 1":    "saddlebrown",
        "Download 2":    "tan",

        # Postprocess → rosas
        "Postprocess 1": "deeppink",
        "Postprocess 2": "pink",
    }

    canonical_order = [
        "Load 1",
        "Load 2",

        "Model Load 1",
        "Model Load 2",

        "Preprocess 1",
        "Preprocess 2",

        "Upload 1",
        "Upload 2",

        "Inference 1",
        "Inference 2",

        "Download 1",
        "Download 2",

        "Postprocess 1",
        "Postprocess 2",
    ]

    # Preparar figura
    fig, ax = plt.subplots(figsize=(12, 6))

    # Criar listas para todos os testes agrupados por plataforma
    all_tests = []
    all_metrics = []
    for plat in platforms:
        tests = sorted(all_results.keys())
        tests = [t for t in tests if plat in all_results[t]]  # filtra testes que têm esta plataforma
        all_tests.extend([(t, plat) for t in tests])
        all_metrics.extend([all_results[t][plat] for t in tests])

    # Descobrir todas as etapas presentes
    ordered_stages = []
    for metrics in all_metrics:
        tb = metrics["time_breakdown"]
        for col in tb.keys():
            if col == "total_time":
                continue
            stage = stage_mapping.get(col, col)
            if stage not in ordered_stages:
                ordered_stages.append(stage)

    # Reordenar segundo ordem canónica
    ordered_stages = [s for s in canonical_order if s in ordered_stages]

    # Construção do gráfico stacked
    bottoms = [0] * len(all_tests)
    for stage in ordered_stages:
        values = []
        for metrics in all_metrics:
            tb = metrics["time_breakdown"]
            stage_sum = sum(val for col, val in tb.items() if stage_mapping.get(col, col) == stage and col != "total_time")
            values.append(stage_sum)
        ax.bar(range(len(all_tests)), values, bottom=bottoms, color=stage_colors.get(stage), label=stage)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    # Labels de x
    labels = [f"{tname}-{plat}" for tname, plat in all_tests]
    ax.set_xticks(range(len(all_tests)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Time per Image (s)")
    ax.set_title("Time Breakdown per Test")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"[DEBUG] Time breakdown plot saved to {output_path}")


def plot_time_breakdown_split(all_results: Dict, platforms, output_dir: str):
    """
    Cria dois plots de time breakdown:
    - Um com os testes A0, B1, B2, B3, C1, C2, C3 (sem B0 e C0)
    - Outro apenas com B0 e C0 (que têm tempos de Model Load muito grandes)
    """

    if isinstance(platforms, str):
        platforms = [platforms]

    # Grupos de testes
    normal_tests = ["A0", "A1", "B1", "B2", "B3", "C1", "C2", "C3"]
    heavy_tests = ["B0", "C0"]

    # Reutilizamos a função existente, mas chamamos separadamente
    def _plot_subset(tests_subset, label):
        # Filtrar apenas os testes que existem no all_results
        filtered = {t: all_results[t] for t in tests_subset if t in all_results}
        if not filtered:
            print(f"[DEBUG] Nenhum teste válido encontrado para {label}, ignorando plot.")
            return

        output_path = os.path.join(output_dir, f"time_breakdown_{'_'.join(platforms)}_{label}.png")
        # Chama a função principal de plot, agora suportando múltiplas plataformas
        plot_time_breakdown(filtered, platforms, output_path)
        print(f"[DEBUG] Time breakdown plot ({label}) salvo em {output_path}")

    # Criar os dois plots
    _plot_subset(normal_tests, "normal")
    # _plot_subset(heavy_tests, "heavy")


def plot_throughput(all_results: Dict, platforms, output_path: str):

    tests = sorted(all_results.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    if isinstance(platforms, str):
        platforms = [platforms]

    positions, labels, values, colors = [], [], [], []

    for p in platforms:
        for tname in tests:
            positions.append(len(positions))
            labels.append(f"{tname}-{p.upper()}")
            values.append(all_results[tname][p]['throughput'])
            colors.append(PLATFORM_COLORS[p])

    ax.bar(positions, values, color=colors, alpha=0.7)
    ax.set_ylabel('Throughput (images/s)')
    title = f"Throughput per Test for {' vs '.join([p.upper() for p in platforms])}"
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # legenda manual por plataforma
    if len(platforms) > 1:
        handles = [mpatches.Patch(color=PLATFORM_COLORS[p], label=p.upper()) for p in platforms]
        ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'[DEBUG] Throughput plot saved to {output_path}')


def plot_power(all_results: Dict, platforms, output_path: str):

    tests = sorted(all_results.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    if isinstance(platforms, str):
        platforms = [platforms]

    positions, labels, values, colors = [], [], [], []

    for p in platforms:
        for tname in tests:
            positions.append(len(positions))
            labels.append(f"{tname}-{p.upper()}")
            values.append(all_results[tname][p]['mean_power_w'])
            colors.append(PLATFORM_COLORS[p])

    ax.bar(positions, values, color=colors, alpha=0.7)
    ax.set_ylabel('Mean Power (W)')
    title = f"Mean Power per Test for {' vs '.join([p.upper() for p in platforms])}"
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # legenda manual por plataforma
    if len(platforms) > 1:
        handles = [mpatches.Patch(color=PLATFORM_COLORS[p], label=p.upper()) for p in platforms]
        ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'[DEBUG] Power plot saved to {output_path}')


def plot_energy(all_results: Dict, platforms, output_path: str):

    tests = sorted(all_results.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    if isinstance(platforms, str):
        platforms = [platforms]

    positions, labels, values, colors = [], [], [], []

    for p in platforms:
        for tname in tests:
            positions.append(len(positions))
            labels.append(f"{tname}-{p.upper()}")
            values.append(all_results[tname][p]['energy_j'])
            colors.append(PLATFORM_COLORS[p])

    ax.bar(positions, values, color=colors, alpha=0.7)
    ax.set_ylabel('Energy (J)')
    title = f"Energy per Test for {' vs '.join([p.upper() for p in platforms])}"
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # legenda manual por plataforma
    if len(platforms) > 1:
        handles = [mpatches.Patch(color=PLATFORM_COLORS[p], label=p.upper()) for p in platforms]
        ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'[DEBUG] Energy plot saved to {output_path}')


def plot_efficiency_metric(all_results: Dict, platforms, metric_name: str,
                           output_path: str, ylabel: str, scale: float = 1.0,
                           aggregate: str = "sum"):
    """
    aggregate: 'sum' ou 'mean' -> define como combinar eficiências de múltiplos modelos no mesmo teste.
    """

    tests = sorted(all_results.keys())
    fig, ax = plt.subplots(figsize=(12, 6))

    if isinstance(platforms, str):
        platforms = [platforms]

    positions, labels, values, colors = [], [], [], []

    for p in platforms:
        for tname in tests:
            metrics = all_results[tname][p]
            eff = metrics['efficiency'][metric_name]
            model_ids = metrics['model_ids']

            # --- Agregar eficiências de todos os modelos do mesmo teste ---
            if isinstance(eff, dict):
                eff_values = [eff[mid] for mid in model_ids if not np.isnan(eff[mid])]
                if len(eff_values) > 0:
                    if aggregate == "sum":
                        eff_value = np.sum(eff_values)
                    elif aggregate == "mean":
                        eff_value = np.mean(eff_values)
                    else:
                        raise ValueError("aggregate must be 'sum' or 'mean'")
                else:
                    eff_value = np.nan
            else:
                eff_value = eff  # métricas globais como throughput_per_watt

            # --- Armazenar resultado ---
            positions.append(len(positions))
            labels.append(f"{tname}-{p.upper()}")
            values.append(eff_value * scale)
            colors.append(PLATFORM_COLORS[p])

    # --- Gráfico ---
    ax.bar(positions, values, color=colors, alpha=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} for {' vs '.join([p.upper() for p in platforms])}")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    if len(platforms) > 1:
        handles = [mpatches.Patch(color=PLATFORM_COLORS[p], label=p.upper()) for p in platforms]
        ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[DEBUG] Efficiency plot ({metric_name}) saved to {output_path}")


def plot_all_data(all_results, output_dir: str):

    acc_dir = f'{output_dir}/1_accuracy'
    throughput_dir = f'{output_dir}/2_throughput'
    power_dir = f'{output_dir}/3_power'
    time_breakdown_dir = f'{output_dir}/4_time_breakdown'
    efficiency_dir = f'{output_dir}/5_efficiency'

    if not os.path.exists(acc_dir): os.makedirs(acc_dir)
    if not os.path.exists(throughput_dir): os.makedirs(throughput_dir)
    if not os.path.exists(power_dir): os.makedirs(power_dir)
    if not os.path.exists(time_breakdown_dir): os.makedirs(time_breakdown_dir)
    if not os.path.exists(efficiency_dir): os.makedirs(efficiency_dir)

    plot_accuracy(all_results,              'kv260',            f'{acc_dir}/accuracy_kv260.png')
    plot_accuracy(all_results,              'gpu',              f'{acc_dir}/accuracy_gpu.png')
    plot_accuracy(all_results,              ['kv260', 'gpu'],   f'{acc_dir}/accuracy_kv260_gpu.png')

    plot_throughput(all_results,            'kv260',            f'{throughput_dir}/throughput_kv260.png')
    plot_throughput(all_results,            'gpu',              f'{throughput_dir}/throughput_gpu.png')
    plot_throughput(all_results,            ['kv260', 'gpu'],   f'{throughput_dir}/throughput_kv260_gpu.png')

    plot_power(all_results,                 'kv260',            f'{power_dir}/power_kv260.png')
    plot_power(all_results,                 'gpu',              f'{power_dir}/power_gpu.png')
    plot_power(all_results,                 ['kv260', 'gpu'],   f'{power_dir}/power_kv260_gpu.png')
    plot_energy(all_results,                'kv260',            f'{power_dir}/energy_kv260.png')
    plot_energy(all_results,                'gpu',              f'{power_dir}/energy_gpu.png')
    plot_energy(all_results,                ['kv260', 'gpu'],   f'{power_dir}/energy_kv260_gpu.png')

    plot_time_breakdown(all_results,        'kv260',            f'{time_breakdown_dir}/time_kv260.png')
    plot_time_breakdown(all_results,        'gpu',              f'{time_breakdown_dir}/time_gpu.png')
    plot_time_breakdown(all_results,        ['kv260', 'gpu'],   f'{time_breakdown_dir}/time_kv260_gpu.png')
    plot_time_breakdown_split(all_results,  "gpu",              f'{time_breakdown_dir}')
    plot_time_breakdown_split(all_results,  "kv260",            f'{time_breakdown_dir}')
    plot_time_breakdown_split(all_results,  ['kv260', 'gpu'],   f'{time_breakdown_dir}')

    # Efficiency_metrics
    plot_efficiency_metric(all_results,     'kv260',            'acc_per_joule',            f'{efficiency_dir}/acc_per_joule_kv260.png',                ylabel='Accuracy per Joule',        scale=1000)
    plot_efficiency_metric(all_results,     'gpu',              'acc_per_joule',            f'{efficiency_dir}/acc_per_joule_gpu.png',                  ylabel='Accuracy per Joule',        scale=1000)
    plot_efficiency_metric(all_results,     ['kv260', 'gpu'],   'acc_per_joule',            f'{efficiency_dir}/acc_per_joule_kv260_gpu.png',            ylabel='Accuracy per Joule',        scale=1000)

    plot_efficiency_metric(all_results,     'kv260',            'correct_imgs_per_joule',   f'{efficiency_dir}/correct_imgs_per_joule_kv260.png',       ylabel='Correct Images per Joule',  scale=1.0)
    plot_efficiency_metric(all_results,     'gpu',              'correct_imgs_per_joule',   f'{efficiency_dir}/correct_imgs_per_joule_gpu.png',         ylabel='Correct Images per Joule',  scale=1.0)
    plot_efficiency_metric(all_results,     ['kv260', 'gpu'],   'correct_imgs_per_joule',   f'{efficiency_dir}/correct_imgs_per_joule_kv260_gpu.png',   ylabel='Correct Images per Joule',  scale=1.0)

    plot_efficiency_metric(all_results,     'kv260',            'throughput_per_watt',      f'{efficiency_dir}/throughput_per_watt_kv260.png',          ylabel='Throughput per Watt',       scale=1.0)
    plot_efficiency_metric(all_results,     'gpu',              'throughput_per_watt',      f'{efficiency_dir}/throughput_per_watt_gpu.png',            ylabel='Throughput per Watt',       scale=1.0)
    plot_efficiency_metric(all_results,     ['kv260', 'gpu'],   'throughput_per_watt',      f'{efficiency_dir}/throughput_per_watt_kv260_gpu.png',      ylabel='Throughput per Watt',       scale=1.0)
