"""Export run records/results; include large checkpoint files only when requested."""
import argparse
from pathlib import Path
from zipfile import ZipFile,ZIP_STORED


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--include-checkpoints',action='store_true')
    args=parser.parse_args(argv);root=Path(args.root).resolve();out=Path(args.output).resolve()
    if not root.is_dir():raise ValueError('Export root must be a directory')
    if out==root or root in out.parents:raise ValueError('Archive must be outside the source root')
    if out.exists():raise ValueError('Refusing to overwrite archive')
    out.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(out,'w',compression=ZIP_STORED) as archive:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.is_symlink():continue
            if path.suffix in ('.pt','.pth') and not args.include_checkpoints:continue
            archive.write(path,str(path.relative_to(root)))
    print(out)


if __name__=='__main__':main()
