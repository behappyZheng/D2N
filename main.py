import tensorflow as tf
from Domain_Adaptation import Parking_lot
from Domain_Adaptation_test import Parking_lot_test
# from Domain_Adaptation_center import Parking_lot
# from Domain_Adaptation_pretrain import Parking_lot_pretrain
# from Domain_Adaptation_center_test import Parking_lot_test
# from Domain_Adaptation_center_test_tsne import Parking_lot_test
# from Domain_Adaptation_semi import Parking_lot_semi
# from Domain_Adaptation_siamese2 import Parking_lot
# from Domain_Adaptation_siamese_test import Parking_lot_test

from add_perceptual import Parking_lot_perce
import argparse
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of DRIT"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='guide', help='[train, test, guide]')
    parser.add_argument('--dataset', type=str, default='parkinglot_90-45_V2', help='dataset_name')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='using learning rate decay')

    parser.add_argument('--epoch', type=int, default=15, help='The number of epochs to run') #14
    # parser.add_argument('--epoch_step2', type=int, default=15, help='The number of epochs to run') #14
    parser.add_argument('--decay_epoch', type=int, default=7, help='The number of decay epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')

    parser.add_argument('--num_attribute', type=int, default=3, help='number of attributes to sample')
    parser.add_argument('--direction', type=str, default='a2b', help='direction of guided image translation')
    parser.add_argument('--guide_img', type=str, default='guide.jpg', help='Style guided image translation')

    parser.add_argument('--gan_type', type=str, default='lsgan', help='GAN loss type [gan / lsgan]')

    parser.add_argument('--lr', type=float, default=0.00005, help='The learning rate')
    parser.add_argument('--content_adv_w', type=int, default=1, help='weight of content adversarial loss')
    parser.add_argument('--domain_adv_w', type=int, default=1, help='weight of domain adversarial loss')
    parser.add_argument('--cycle_w', type=int, default=5, help='weight of cross-cycle reconstruction loss')
    parser.add_argument('--recon_w', type=int, default=5, help='weight of self-reconstruction loss')
    parser.add_argument('--recon_latent_w', type=float, default=5, help='weight of style & content reconstruction loss')
    parser.add_argument('--latent_w', type=int, default=5, help='wight of latent regression loss')
    parser.add_argument('--cls_w', type=int, default=5, help='wight of classifier loss')
    parser.add_argument('--center_w', type=int, default=5, help='wight of center loss')
    parser.add_argument('--kl_w', type=float, default=0.01, help='weight of kl-divergence loss')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--concat', type=str2bool, default=False, help='using concat networks')  #True False

    # concat = False : for the shape variation translation (cat <-> dog)
    # concat = True : for the shape preserving translation (winter <-> summer)

    parser.add_argument('--n_z', type=int, default=8, help='length of z')
    parser.add_argument('--n_layer', type=int, default=4, help='number of layers in G, D')

    parser.add_argument('--n_dis', type=int, default=4, help='number of discriminator layer')

    # If you don't use multi-discriminator, then recommend n_dis = 6

    parser.add_argument('--n_scale', type=int, default=3, help='number of scales for discriminator')

    # using the multiscale discriminator often gets better results

    parser.add_argument('--n_d_con', type=int, default=1, help='# of iterations for updating content discrimnator')

    # model can still generate diverse results with n_d_con = 1

    parser.add_argument('--sn', type=str2bool, default=False, help='using spectral normalization')

    parser.add_argument('--style_layer', type=str, default='relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', help='Which layers to extract style from')
    parser.add_argument('--content_layer', type=str, default='relu4_2', help='which VGG layer extract content from')
    parser.add_argument('--style_layer_w', type=int, default=1e2 ,help='Weight for style features loss')
    parser.add_argument('--content_layer_w', type=int, default=5e0 ,help='Weight for content features loss')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

# def parse_args():
#     desc = "Tensorflow implementation of DRIT"
#
#     parser = argparse.ArgumentParser(description=desc)
#     parser.add_argument('--phase', type=str, default='train', help='[train, test, guide]')
#     parser.add_argument('--dataset', type=str, default='parkinglot_basic+cyc+rec+lat+sem', help='dataset_name')
#     parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')
#     parser.add_argument('--decay_flag', type=str2bool, default=True, help='using learning rate decay')
#
#     parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')  # 14
#     parser.add_argument('--decay_epoch', type=int, default=8, help='The number of decay epochs to run')
#     parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
#     parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
#     parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
#     parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
#
#     parser.add_argument('--num_attribute', type=int, default=3, help='number of attributes to sample')
#     parser.add_argument('--direction', type=str, default='a2b', help='direction of guided image translation')
#     parser.add_argument('--guide_img', type=str, default='guide.jpg', help='Style guided image translation')
#
#     parser.add_argument('--gan_type', type=str, default='lsgan', help='GAN loss type [gan / lsgan]')
#
#     parser.add_argument('--lr', type=float, default=0.00005, help='The learning rate')
#     parser.add_argument('--content_adv_w', type=int, default=1, help='weight of content adversarial loss')
#     parser.add_argument('--domain_adv_w', type=int, default=1, help='weight of domain adversarial loss')
#     parser.add_argument('--cycle_w', type=int, default=5, help='weight of cross-cycle reconstruction loss')
#     parser.add_argument('--recon_w', type=int, default=5, help='weight of self-reconstruction loss')
#     parser.add_argument('--recon_latent_w', type=float, default=0, help='weight of style & content reconstruction loss')
#     parser.add_argument('--latent_w', type=int, default=5, help='wight of latent regression loss')
#     parser.add_argument('--cls_w', type=int, default=0, help='wight of classifier loss')
#     parser.add_argument('--center_w', type=int, default=0, help='wight of center loss')
#     parser.add_argument('--kl_w', type=float, default=0, help='weight of kl-divergence loss')
#     parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
#     parser.add_argument('--concat', type=str2bool, default=False, help='using concat networks')  # True False
#
#     # concat = False : for the shape variation translation (cat <-> dog)
#     # concat = True : for the shape preserving translation (winter <-> summer)
#
#     parser.add_argument('--n_z', type=int, default=8, help='length of z')
#     parser.add_argument('--n_layer', type=int, default=4, help='number of layers in G, D')
#
#     parser.add_argument('--n_dis', type=int, default=4, help='number of discriminator layer')
#
#     # If you don't use multi-discriminator, then recommend n_dis = 6
#
#     parser.add_argument('--n_scale', type=int, default=3, help='number of scales for discriminator')
#
#     # using the multiscale discriminator often gets better results
#
#     parser.add_argument('--n_d_con', type=int, default=1, help='# of iterations for updating content discrimnator')
#
#     # model can still generate diverse results with n_d_con = 1
#
#     parser.add_argument('--sn', type=str2bool, default=False, help='using spectral normalization')
#
#     parser.add_argument('--style_layer', type=str, default='relu1_1,relu2_1,relu3_1,relu4_1,relu5_1',
#                         help='Which layers to extract style from')
#     parser.add_argument('--content_layer', type=str, default='relu4_2', help='which VGG layer extract content from')
#     parser.add_argument('--style_layer_w', type=int, default=1e2, help='Weight for style features loss')
#     parser.add_argument('--content_layer_w', type=int, default=5e0, help='Weight for content features loss')
#
#     parser.add_argument('--img_size', type=int, default=256, help='The size of image')
#     parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
#
#     parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
#                         help='Directory name to save the checkpoints')
#     parser.add_argument('--result_dir', type=str, default='results',
#                         help='Directory name to save the generated images')
#     parser.add_argument('--log_dir', type=str, default='logs',
#                         help='Directory name to save training logs')
#     parser.add_argument('--sample_dir', type=str, default='samples',
#                         help='Directory name to save the samples on training')
#
#     return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    tf.reset_default_graph()
    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = Parking_lot_test(sess, args)  #Parking_lot   Parking_lot_test

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            gan.train()
            # gan.train_step2()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

        if args.phase == 'guide' :
            gan.guide_test()
            print(" [*] Guide finished!")

if __name__ == '__main__':
    main()
