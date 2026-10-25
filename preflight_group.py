"""Check crop-support gradients, frozen references and resumed updates before training."""

from training.group_trainer import run_main


def main(argv=None):
    run_main(argv, preflight=True)


if __name__ == "__main__":
    main()
