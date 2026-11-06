"""The author's adopted plotting functions, with temporary filename adapters."""
from pathlib import Path
import csv
import json
import shutil
import tempfile


def plot_all(data, output, main_cells):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    from . import grid, main, scatter, trim, delta_d
    with tempfile.TemporaryDirectory(prefix='swaap-plot-inputs-') as folder:
        inputs = Path(folder)
        for source in Path(data).glob('FIGURE_*.csv'):
            shutil.copyfile(source, inputs/source.name.removeprefix('FIGURE_'))
        with (inputs/'table1_all_cells_rounding_validation.csv').open('w',newline='') as f:
            w = csv.DictWriter(f,fieldnames=['task','method','metric','recovered_mean','recovered_sample_sd'])
            w.writeheader()
            for row in main_cells:
                if row['method']!='Clean Reference' or row['metric'] not in ('Return','delta_m'):
                    continue
                w.writerow(dict(task=row['task'],method=row['method'],metric=row['metric'],
                                recovered_mean=row['mean'],recovered_sample_sd=row['sample_sd']))
        args = ['--data',str(inputs),'--output',str(output)]
        grid.main(args)
        main.main(args)
        scatter.main(args)
        try:
            trim.main(['--data',str(inputs/'TRIM_FOUR_CATEGORY_METRICS.csv'),
                       '--output',str(output),'--no-contact'])
        except SystemExit as error:
            if error.code not in (None,0):raise
        delta_d.main(args)
    report = dict(status='RECOMPUTED_FROM_SAVED_INPUTS',
                  figures=['three-task alpha comparison','nine-task alpha/budget grid',
                           'two developmental target scatters','TRIM','incoming data deviation'],
                  style='Adopted plotting functions; no numeric or style redesign.',
                  limitations=['Historical training curves and PCA are not regenerated: their raw arrays are unavailable.',
                               'Full-grid red reference uses fitted Clean own-rollout error; its Clean delta_m legend matches the author-adopted main18/PDF16 correction, with data and style preserved.'],
                  environment_episodes=0)
    (output/'figure_reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
