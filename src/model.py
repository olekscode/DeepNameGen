from train import start_training
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model that learns method names.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', '-t', action='store_true', help='train the model')
    group.add_argument('--deploy-server', '-d', action='store_true', help='deploy the trained model on server')
    args = parser.parse_args()
    return args


def run_train():
    """WARNING: Time consuming operation. Takes around 6 hours"""
    start_training()


def run_deploy_server():
    pass


if __name__ == '__main__':
    args = parse_args()

    if args.train:
        run_train()
    elif args.deploy_server:
        run_deploy_server()
