from ..config.utils import boolean_feature, empty_beam_parser as basic_beam_parser


def get_ssl_parser():

    parser = basic_beam_parser()

    boolean_feature(parser, "verbose-lgb", False, "Print progress in lgb training")
    parser.add_argument('--similarity', type=str, metavar='hparam', default='cosine',
                        help='Similarity distance in UniversalSSL')
    parser.add_argument('--p-dim', type=int, default=None, help='Prediction/Projection output dimension')
    parser.add_argument('--temperature', type=float, default=1.0, metavar='hparam', help='Softmax temperature')
    parser.add_argument('--var-eps', type=float, default=0.0001, metavar='hparam', help='Std epsilon in VICReg')
    parser.add_argument('--lambda-vicreg', type=float, default=25., metavar='hparam',
                        help='Lambda weight in VICReg')
    parser.add_argument('--mu-vicreg', type=float, default=25., metavar='hparam', help='Mu weight in VICReg')
    parser.add_argument('--nu-vicreg', type=float, default=1., metavar='hparam', help='Nu weight in VICReg')
    parser.add_argument('--lambda-mean-vicreg', type=float, default=20., metavar='hparam',
                        help='lambda-mean weight in BeamVICReg')
    parser.add_argument('--tau', type=float, default=.99, metavar='hparam', help='Target update factor')
    parser.add_argument('--lambda-twins', type=float, default=0.005, metavar='hparam',
                        help='Off diagonal weight factor for Barlow Twins loss')

    parser.add_argument('--lgb-rounds', type=int, default=40, help='LGB argument: num_round')
    parser.add_argument('--lgb-num-leaves', type=int, default=31, help='LGB argument: num_leaves')
    parser.add_argument('--lgb-max-depth', type=int, default=4, help='LGB argument: max_depth')
    parser.add_argument('--lgb-device', type=int, default=None, help='LGB argument: device')
    return parser
