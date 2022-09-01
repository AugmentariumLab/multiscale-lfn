import json
import os

import configargparse
import cv2

VIDEO = "video"
HOLOSTUDIO = "holostudio"
NERFSYNTHETIC = "nerfsynthetic"
ADAPTIVE_MLP = "AdaptiveMLP"
MLP = "MLP"
ADAPTIVE_RESNET = "AdaptiveResnet"
LEARNED_RAY = "LearnedRay"
SUBNET = "SubNet"
MIPNET = "MipNet"
SMIPNET = "SMipNet"

TRAIN = "train"
INFERENCE = "inference"
VISUALIZE_CAMERAS = 'visualize_cameras'
SAVE_PROTO = "save_proto"
VIEWER = "viewer"
BENCHMARK = "benchmark"
DEBUG = "debug"
EVAL_MEMORIZATION = "eval_memorization"
EVAL_MEMORIZATION_MULTIRES = "eval_memorization_multires"
EVAL = "eval"
EVAL_MULTIRES = "eval_multires"
GET_TRAIN_TIME = "get_train_time"
RENDER_OCCUPANCY_MAP = "render_occupancy_map"
RENDER_TRANSITION = "render_transition"
RENDER_FOVEATION = "render_foveation"


def str2bool(val):
    """Converts the string value to a bool.
    Args:
      val: string representing true or false
    Returns:
      bool
    """
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def dataset_interp_mode(val: str):
    mapping = {
        cv2.INTER_AREA: cv2.INTER_AREA,
        cv2.INTER_LINEAR: cv2.INTER_LINEAR,
        cv2.INTER_CUBIC: cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC
    }
    return mapping[val]


def parse_options():
    parser = configargparse.ArgParser(description='Training program')
    parser.add_argument('-c', '--config', required=True, is_config_file=True,
                        help='config file path')
    parser.add_argument('--script-mode',
                        type=str,
                        help='script mode',
                        choices=[
                            TRAIN,
                            INFERENCE,
                            VISUALIZE_CAMERAS,
                            SAVE_PROTO,
                            VIEWER,
                            BENCHMARK,
                            DEBUG,
                            EVAL_MEMORIZATION,
                            EVAL_MEMORIZATION_MULTIRES,
                            EVAL,
                            EVAL_MULTIRES,
                            GET_TRAIN_TIME,
                            RENDER_OCCUPANCY_MAP,
                            RENDER_TRANSITION,
                            RENDER_FOVEATION
                        ],
                        default=TRAIN)
    parser.add_argument("--device",
                        type=str,
                        help="preferred device",
                        default="cuda")
    parser.add_argument('--checkpoints-dir',
                        type=str,
                        help='checkpoints directory',
                        default="checkpoints")
    parser.add_argument("--checkpoint-count",
                        type=int,
                        help="how many checkpoints to keep",
                        default=3)
    parser.add_argument("--checkpoint-interval",
                        type=int,
                        help="checkpoint interval",
                        default=1000)
    parser.add_argument("--learning-rate",
                        type=float,
                        help="learning rate",
                        default=0.0001)
    parser.add_argument("--latent-learning-rate",
                        type=float,
                        help="learning rate",
                        default=0.001)
    parser.add_argument("--use-latent-codes",
                        type=str2bool,
                        help="use latent codes for time",
                        default=True)
    parser.add_argument("--dataset",
                        type=str,
                        help="dataset type",
                        choices=[VIDEO, HOLOSTUDIO, NERFSYNTHETIC],
                        default="video")
    parser.add_argument("--dataset-path",
                        type=str,
                        help="path to input dataset",
                        default="")
    parser.add_argument("--dataset-max-frames",
                        type=int,
                        help="path to input dataset",
                        default=60)
    parser.add_argument("--dataset-render-poses",
                        type=int,
                        help="number of render poses",
                        default=30)
    parser.add_argument("--dataset-render-poses-height",
                        type=float,
                        help="height of render poses cameras",
                        default=0.4)
    parser.add_argument("--dataset-render-poses-centeroffset",
                        type=float,
                        nargs="*",
                        help="offset center of render poses cameras")
    parser.add_argument("--dataset-mip-factors",
                        type=int,
                        nargs="*",
                        help="dataset mip factors")
    parser.add_argument("--dataset-interp-mode",
                        type=dataset_interp_mode,
                        default=cv2.INTER_AREA)
    parser.add_argument("--dataset-loadeverynthview",
                        type=int, 
                        default=1,
                        help="interval for poses to load")
    parser.add_argument("--dataset-ignore-poses",
                        type=int,
                        nargs="*",
                        help="indices of ignored poses")
    parser.add_argument("--dataset-val-poses",
                        type=int,
                        nargs="*",
                        help="indices of validation poses")
    parser.add_argument("--dataset-test-poses",
                        type=int,
                        nargs="*",
                        help="indices of test poses")
    parser.add_argument("--dataloader-num-workers",
                        type=int,
                        help="dataloader workers",
                        default=4)
    parser.add_argument("--dataloader-prefetch-factor",
                        type=int,
                        help="dataloader prefetch factor",
                        default=2)
    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs',
                        default=1)
    parser.add_argument('--batch-size',
                        type=int,
                        help='ray batch size',
                        default=64)
    parser.add_argument('--frame-batch-size',
                        type=int,
                        help='frame batch size',
                        default=1)
    parser.add_argument('--val-batch-size',
                        type=int,
                        help='number of rays for validation',
                        default=64)
    parser.add_argument("--validation-interval",
                        type=int,
                        help="validation interval",
                        default=200)
    parser.add_argument("--train-tensorboard-interval",
                        type=int,
                        help="train tensorboard interval",
                        default=100)
    parser.add_argument("--positional-encoding-functions",
                        type=int,
                        help="positional encoding functions",
                        default=0)
    parser.add_argument("--num-frames-factor",
                        type=int,
                        help="number of frames per input pano",
                        default=30)
    parser.add_argument("--video-framerate",
                        type=int,
                        help="framerate of the output video in inference mode",
                        default=10)
    parser.add_argument("--num-latent-codes",
                        type=int,
                        help="number of latent codes, -1 for 1 per frame, -2 for 1 per 2 frames.",
                        default=-1)
    parser.add_argument("--latent-code-dim",
                        type=int,
                        help="latent code dimension",
                        default=256)
    parser.add_argument("--train-frame-factor",
                        type=float,
                        help="percent of each frame to use for training",
                        default=0.2)
    parser.add_argument("--dataset-start-frame",
                        type=int,
                        help="which frame to start training on",
                        default=0)
    parser.add_argument("--efficiency-loss-lambda",
                        type=float,
                        help="Lambda factor for expected number of layers in loss function.",
                        default=0.1)
    parser.add_argument("--model",
                        type=str,
                        choices=[ADAPTIVE_MLP, MLP,
                                 ADAPTIVE_RESNET, SUBNET, MIPNET,
                                 SMIPNET],
                        help="Which model to use",
                        default=MLP)
    parser.add_argument("--model-layers",
                        type=int,
                        help="Number of model layers",
                        default=6)
    parser.add_argument("--model-width",
                        type=int,
                        help="Neurons per layer",
                        default=256)
    parser.add_argument("--model-use-layernorm",
                        type=str2bool,
                        help="Whether to use layernorm",
                        default=True)
    parser.add_argument("--dataset-resize-factor",
                        type=int,
                        help="Factor by which to downsize the dataset.",
                        default=1)
    parser.add_argument("--lossfn-color",
                        type=str,
                        choices=["l1", "l2"],
                        help="Loss function for color",
                        default="l1")
    parser.add_argument("--lossfn-color-mask-factor",
                        type=float,
                        help="Multiplier for the mask color",
                        default=0.0)
    parser.add_argument("--use-importance-training",
                        type=str2bool,
                        help="Use importance training",
                        default=False)
    parser.add_argument("--importance-layers",
                        type=int,
                        help="Layers for importance map",
                        default=3)
    parser.add_argument("--importance-features",
                        type=int,
                        help="Features for importance map",
                        default=64)
    parser.add_argument("--importance-loss-lambda",
                        type=float,
                        help="Multiplier for the importance loss",
                        default=0.01)
    parser.add_argument("--importance-lerp",
                        type=float,
                        help="Offset factor for importance map: [0, 1]. 0 means importance map is useless.",
                        default=1)
    parser.add_argument("--random-seed",
                        type=int,
                        help="Random seed",
                        default=42)
    parser.add_argument("--multimodel-num-models",
                        type=int,
                        help="Number of models for the multimodel model",
                        default=64)
    parser.add_argument("--multimodel-selection-mode",
                        type=str,
                        choices=["angle", "mlp", "cylinder"],
                        help="multimodel selection mode",
                        default="angle")
    parser.add_argument("--multimodel-selection-freeze-epochs",
                        type=int,
                        help="multimodel selection freeze epochs",
                        default=-1)
    parser.add_argument("--multimodel-selection-layers",
                        type=int,
                        help="Number of layers for the selection model",
                        default=5)
    parser.add_argument("--multimodel-selection-hidden-features",
                        type=int,
                        help="Number of features for the selection model",
                        default=64)
    parser.add_argument("--multimodel-importance-loss-lambda",
                        type=float,
                        help="Multiplier for the importance loss",
                        default=0)
    parser.add_argument("--multimodel-selection-lerp",
                        type=float,
                        help="Lerp multimodel selection probabilities",
                        default=0.5)
    parser.add_argument("--multimodel-loadbalance-loss-lambda",
                        type=float,
                        help="Load balance loss lambda",
                        default=0)
    parser.add_argument("--multimodel-clustering-loss-lambda",
                        type=float,
                        help="Clustering loss lambda",
                        default=0)
    parser.add_argument("--multimodel-clustering-loss-version",
                        type=int,
                        help="Clustering loss version",
                        default=0)
    parser.add_argument("--multimodel-num-top-outputs",
                        type=int,
                        help="Number of outputs for multimodel",
                        default=1)
    parser.add_argument("--predict-alpha",
                        type=str2bool,
                        help="Enable alpha",
                        default=False)
    parser.add_argument("--model-output-every",
                        type=int,
                        help="For adaptive resnet",
                        default=3)
    parser.add_argument("--skip-gif-generation",
                        type=str2bool,
                        help="Skip generating gif",
                        default=False)
    parser.add_argument("--atlasnet-atlas-layers",
                        type=int,
                        help="atlas layers",
                        default=10)
    parser.add_argument("--atlasnet-atlas-features",
                        type=int,
                        help="atlas features",
                        default=512)
    parser.add_argument("--use-aux-network",
                        type=str2bool,
                        help="Learn auxiliary network",
                        default=False)
    parser.add_argument("--aux-layers",
                        type=int,
                        help="aux layers",
                        default=3)
    parser.add_argument("--aux-features",
                        type=int,
                        help="aux features",
                        default=16)
    parser.add_argument("--aux-layernorm",
                        type=str2bool,
                        help="aux layernorm",
                        default=False)
    parser.add_argument("--aux-encode-saliency",
                        type=str2bool,
                        help="aux encode-saliency",
                        default=False)
    parser.add_argument("--subnet-factors",
                        type=lambda x: json.loads(x) if x else "",
                        nargs="*",
                        help="subnet factors",
                        default="")
    parser.add_argument("--subnet-interleave",
                        type=str2bool,
                        help="subnet",
                        default=False)
    parser.add_argument("--subnet-optimized-training",
                        type=str2bool,
                        help="subnet optimized training",
                        default=False)
    parser.add_argument("--subset-bg-pixels",
                        type=float,
                        help="fraction of each back is bg pixels, -1 for disable",
                        default=-1.0)
    parser.add_argument("--lr-schedule-gamma",
                        type=float,
                        help="Exponential LR schedule gamma",
                        default=0.0)
    parser.add_argument("--viewer-batch-size",
                        type=int,
                        help="Viewer ray batch size",
                        default=2000000)
    parser.add_argument("--lod-factor-schedule",
                        type=lambda x: json.loads(x) if x else "",
                        nargs="*",
                        help="list of (epoch, lod, factor) tuples",
                        default="")
    parser.add_argument("--lod-factor-schedule-use-mip",
                        type=str2bool,
                        help="Sample ray position from high-res images instead of low-res images.",
                        default=False)
    parser.add_argument("--mipnet-share-gradients",
                        type=str2bool,
                        help="mipnet gradients propagate across levels",
                        default=False)
    parser.add_argument("--render-truncate-alpha",
                        type=float,
                        help="If alpha < value, set alpha to 0.",
                        default=0.0)
    parser.add_argument("--inference-indices",
                        type=int,
                        nargs="*",
                        help="indices of inference poses to render")
    parser.add_argument("--inference-lods",
                        type=int,
                        nargs="*",
                        help="indices of inference poses to render")
    parser.add_argument("--inference-halves",
                        type=str2bool,
                        default=False,
                        help="Use half precision")
    parser.add_argument("--training-val-psnr-cutoff",
                        type=float,
                        default=-1,
                        help="Cutoff training is validation psnr surpasses value.")
    parser.add_argument("--training-val-psnr-cutoff-run",
                        type=str,
                        default="",
                        help="Look for psnr cutoff from this run.")
    parser.add_argument("--cache-lowres",
                        type=str2bool,
                        default=False,
                        help="Cache lowres images")
    parsed_args = parser.parse_args()
    if parsed_args.training_val_psnr_cutoff_run:
        run, resolution, lod = parsed_args.training_val_psnr_cutoff_run.split(',')
        lod = int(lod)
        print(f"Pulling training-val-psnr-cutoff from {run} {resolution} {lod}")
        val_file = os.path.join("runs", run, f"eval_val_dataset_{resolution}.json")
        if not os.path.isfile(val_file):
            raise ValueError("File not found")
        with open(val_file, "r") as f:
            val_results = json.load(f)
        cutoff = val_results['avg_cropped_psnr_values'][lod]
        parsed_args.training_val_psnr_cutoff = cutoff
        print(f"Cutoff training at {parsed_args.training_val_psnr_cutoff}")
    return parsed_args
