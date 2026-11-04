"""
Write a single-file UCSC track hub to view hotspot3 results in the genome browser:
normalized density with the fitted background overlaid, and hotspots / peaks at several FDRs.

Files are local paths or URLs. URLs are used as is; local paths are written relative to the
hub file, so host them together. Load the hub in the UCSC browser with My Data > Track Hubs,
or open the link printed with --hub_url.

The track description shown in the browser is trackhub_description.html from the hotspot3
GitHub repository (--description_url to use another page).

Example (a regulome sample):
    S=https://regulome.altius.org/data/per_sample/AG95384
    hotspot3-trackhub --name AG95384 \\
        --density_bw $S/AG95384.normalized_density.bw \\
        --background_bw $S/peaks/AG95384.background.bw \\
        --hotspots_bb 0.001 $S/peaks/fdr0.001/AG95384.hotspots.fdr0.001.bb \\
        --hotspots_bb 0.05 $S/peaks/fdr0.05/AG95384.hotspots.fdr0.05.bb \\
        --peaks_bb 0.001 $S/peaks/fdr0.001/AG95384.peaks.fdr0.001.bb \\
        --peaks_bb 0.05 $S/peaks/fdr0.05/AG95384.peaks.fdr0.05.bb \\
        --out AG95384.hub.txt
"""
import argparse
import os
import urllib.parse

BACKGROUND_RGB = (253, 141, 60)
DESCRIPTION_URL = 'https://raw.githubusercontent.com/vierstralab/hotspot3/main/docs/trackhub_description.html'
UCSC_URL = 'https://genome.ucsc.edu/cgi-bin/hgTracks'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='Sample name, used in track names')
    parser.add_argument('--label', default=None, help='Label shown in the browser (default: --name)')
    parser.add_argument('--density_bw', required=True, help='Normalized density bigWig ({id}.normalized_density.bw)')
    parser.add_argument('--background_bw', required=True, help='Background bigWig ({id}.background.bw)')
    parser.add_argument('--hotspots_bb', nargs=2, action='append', metavar=('FDR', 'BIGBED'), help='Hotspots bigBed at an FDR; repeat for several FDRs')
    parser.add_argument('--peaks_bb', nargs=2, action='append', metavar=('FDR', 'BIGBED'), help='Peaks bigBed at an FDR; repeat for several FDRs')
    parser.add_argument('--out', required=True, help='Hub file to write, e.g. {name}.hub.txt')
    parser.add_argument('--hub_url', default=None, help='URL the hub file will be served from; prints a link that opens the hub')
    parser.add_argument('--genome', default='hg38', help='UCSC genome assembly')
    parser.add_argument('--default_pos', default=None, help='Position to open the browser at with --hub_url, e.g. chr19:47514993-47515640')
    parser.add_argument('--view_limits', default='0:2', help='Y range of the density track')
    parser.add_argument('--email', default='', help='Contact email in the hub')
    parser.add_argument('--description_url', default=DESCRIPTION_URL, help='URL of the track description page (default: the one in the hotspot3 GitHub repository)')
    return parser.parse_args()


def _hub_path(path, hub_file):
    if '://' in path:
        return path
    return os.path.relpath(os.path.abspath(path), os.path.dirname(os.path.abspath(hub_file)))


def hub_text(args, description):
    name = args.name
    label = args.label or name
    text = f"""hub {name}
shortLabel {label}
longLabel {label} hotspot3
useOneFile on
email {args.email}
descriptionUrl {description}

genome {args.genome}

track {name}_signal
container multiWig
aggregate transparentOverlay
showSubtrackColorOnUi on
shortLabel {label}
longLabel {label} density and background
type bigWig
visibility full
viewLimits {args.view_limits}
maxHeightPixels 100:32:8
priority 1
html {description}

    track {name}_normalized_density
    bigDataUrl {_hub_path(args.density_bw, args.out)}
    shortLabel {name} Density
    longLabel {name} Normalized density
    parent {name}_signal
    color 0,0,0
    priority 1
    type bigWig
    html {description}

    track {name}_background
    bigDataUrl {_hub_path(args.background_bw, args.out)}
    shortLabel {name} Background
    longLabel {name} Background
    parent {name}_signal
    color {','.join(map(str, BACKGROUND_RGB))}
    priority 2
    type bigWig
    html {description}
"""
    priority = 2
    for kind, files in [('hotspots', args.hotspots_bb), ('peaks', args.peaks_bb)]:
        if not files:
            continue
        priority += 1
        text += f"""
track {name}_{kind}
compositeTrack on
shortLabel {name} {kind.capitalize()}
longLabel {name} {kind.capitalize()}
visibility dense
type bigBed 12
priority {priority}
html {description}
itemRgb on
"""
        for fdr, path in files:
            priority += 1
            text += f"""
    track {name}_{kind}_fdr{fdr.replace('.', '')}
    bigDataUrl {_hub_path(path, args.out)}
    shortLabel FDR {fdr} {kind.capitalize()}
    longLabel {kind.capitalize()} at FDR {fdr}
    parent {name}_{kind}
    type bigBed 12
    visibility dense
    priority {priority}
    html {description}
    itemRgb on
"""
    return text


def main():
    args = parse_args()
    with open(args.out, 'w') as f:
        f.write(hub_text(args, args.description_url))
    print(f'Track hub written to {args.out}')
    if args.hub_url:
        params = {'db': args.genome, 'hubUrl': args.hub_url}
        if args.default_pos:
            params['position'] = args.default_pos
        print(f'{UCSC_URL}?{urllib.parse.urlencode(params)}')


if __name__ == '__main__':
    main()
