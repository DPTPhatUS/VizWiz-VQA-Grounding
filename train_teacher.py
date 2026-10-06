"""Train this experiment's joint-skip teacher with 50% answer dropout."""
from train import train_main

if __name__ == "__main__":
    train_main(teacher_training=True)
