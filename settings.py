import argparse
class settings:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Sequence Modeling - some models')
        self.parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                            help='batch size (default: 64)')
        self.parser.add_argument('--cuda', action='store_true',
                            help='use CUDA (default: True)')
        self.parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                            help='interval of reportting epoch number (default: 100')
        self.parser.add_argument('--output_interval', type=int, default=2, metavar='N',
                            help='report interval (default: 100')
        self.parser.add_argument('--lr', type=float, default=1e-3,  # transformer是0.00001
                                 help='initial learning rate (default: 2e-3)')
        self.parser.add_argument('--epochs', type=int, default=2000,
                                 help='upper epoch limit (default: 20)')
        self.parser.add_argument('--dropout', type=float, default=0.2,
                            help='dropout applied to layers (default: 0.2)')
        self.parser.add_argument('--clip', type=float, default=1,
                            help='gradient clip, -1 means no clip (default: 1)')
        self.parser.add_argument('--optim', type=str, default='Adam',
                            help='optimizer to use (default: Adam)')
        self.args, self.unknown = self.parser.parse_known_args()
