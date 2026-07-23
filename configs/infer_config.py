import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    ## general
    parser.add_argument('--image_dir', type=str, default='./test/images/fruit.png', help='Image file path')
    parser.add_argument('--out_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to use')
    parser.add_argument('--exp_name',  type=str, default=None, help='Experiment name, use image file name by default')

    ## renderer
    parser.add_argument('--mode',  type=str,  default='single_view', help="Currently we support 'single_view_txt' and 'single_view_target'")
    # parser.add_argument('--traj_txt',  type=str, help="Required for 'single_view_txt' mode, a txt file that specify camera trajectory")
    parser.add_argument("--elevation", default=5.0, type=float,
                        help="Initial angle, no exceptions to change")    
    parser.add_argument("--d_r", default=1.0, type=float,
                        help="Camera distance, default is 1.0, range 0.25 to 2.5")
    parser.add_argument("--d_theta", default=0.0, type=float,
                        help="Vertical rotation, <0 up, >0 down, range -90 to 30; generally not recommended to angle too much downwards")
    parser.add_argument("--d_phi", default=0.0, type=float,
                        help="Horizontal rotation, <0 right, >0 left, supports 360 degrees; range -360 to 360")
    parser.add_argument("--x_offset", default=0.0, type=float,
                        help="Horizontal translation, <0 left, >0 right, range -0.5 to 0.5; depends on depth, excessive movement may cause artifacts")
    parser.add_argument("--y_offset", default=0.0, type=float,
                        help="Vertical translation, <0 up, >0 down, range -0.5 to 0.5; depends on depth, excessive movement may cause artifacts")
    parser.add_argument("--z_offset", default=0.0, type=float,
                        help="Forward and backward translation, <0 back, >0 forward, range -0.5 to 0.5 is ok; depends on depth, excessive movement may cause artifacts")
    parser.add_argument("--traj_type", default="custom", type=str,
                        choices=["custom", "rel_target", "freeze", "free1", "free2", "free3", "free4", "free5", "swing1", "swing2", "orbit"],
                        help="custom refers to a custom trajectory, while the others are pre-defined camera trajectories (see traj_map for details)")
    parser.add_argument("--focal_length", default=1.0, type=float,
                        help="Focal length, range 0.25 to 2.5; changing focal length zooms in and out")
    
## depthcrafter
    parser.add_argument(
        '--unet_path',
        type=str,
        default="/mnt/workspace/ywb/TrajectoryCrafter/checkpoints/DepthCrafter",
        help='Path to the UNet model',
    )

    parser.add_argument(
        '--pre_train_path',
        type=str,
        default="/mnt/workspace/ywb/TrajectoryCrafter/checkpoints/stable-video-diffusion-img2vid",
        help='Path to the pre-trained model',
    )
    parser.add_argument(
        '--cpu_offload', type=str, default='model', help='CPU offload strategy'
    )
    parser.add_argument(
        '--depth_inference_steps', type=int, default=5, help='Number of inference steps'
    )
    parser.add_argument(
        '--depth_guidance_scale',
        type=float,
        default=1.0,
        help='Guidance scale for inference',
    )
    parser.add_argument(
        '--window_size', type=int, default=110, help='Window size for processing'
    )
    parser.add_argument(
        '--overlap', type=int, default=25, help='Overlap size for processing'
    )
    parser.add_argument(
        '--max_res', type=int, default=1024, help='Maximum resolution for processing'
    )
    parser.add_argument('--fps', type=int, default=16, help='Fps for saved video')

    ## warp
    parser.add_argument(
        '--stride', type=int, default=1, help='Sampling stride for input video'
    )
    parser.add_argument(
        '--radius_scale',
        type=float,
        default=1.0,
        help='Scale factor for the spherical radius',
    )
    parser.add_argument('--camera', type=str, default='target', help='traj or target')
    # parser.add_argument(
    #     '--dv', type=str, default='gradual', help='gradual, bullet or direct'
    # )
    parser.add_argument(
        '--mask', default=False, help='Clean the pcd if true'
    )
    # parser.add_argument(
    #     '--traj_txt',
    #     type=str,
    #     help="Required for 'traj' camera, a txt file that specify camera trajectory",
    # )
    parser.add_argument(
        '--target_pose',
        nargs=5,
        type=float,
        help="Required for 'target' mode, specify target camera pose, <theta phi r x y>",
    )
    parser.add_argument(
        '--near', type=float, default=0.0001, help='Near clipping plane distance'
    )
    parser.add_argument(
        '--far', type=float, default=10000.0, help='Far clipping plane distance'
    )
    parser.add_argument('--anchor_idx', type=int, default=0, help='One GT frame')
    
    ## diffusion
    parser.add_argument('--low_gpu_memory_mode', type=bool, default=False, help='Enable low GPU memory mode')
    parser.add_argument('--model_name', type=str, default='checkpoints/CogVideoX-Fun-V1.1-5b-InP', help='Path to the model')
    parser.add_argument('--sampler_name', type=str, choices=["Flow", "Euler", "Euler A", "DPM++", "PNDM", "DDIM_Cog", "DDIM_Origin"], default="Flow", help='Choose the sampler')
    parser.add_argument('--transformer_path', type=str, default=None, help='Path to the pretrained transformer model')
    parser.add_argument('--transformer_path_2', type=str, default=None, help='Path to the pretrained transformer model')
    
    parser.add_argument('--lora_path', type=str, default=None, help='Path to the pretrained transformer model')    
    parser.add_argument('--diffusion_guidance_scale', type=float, default=6.0, help='Guidance scale for inference')
    parser.add_argument('--diffusion_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt for video generation')
    parser.add_argument('--negative_prompt', type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", help='Negative prompt for video generation')
    #parser.add_argument('--negative_prompt', type=str, default="Human", help='Negative prompt for video generation')
    parser.add_argument('--refine_prompt', type=str, default=". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.", help='Prompt for video generation')
    parser.add_argument("--seed", type=int, default=43, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=81, help="inference video length, change to 16 if you use 16 frame model")
    parser.add_argument('--blip_path',type=str,default="./checkpoints/blip2-opt-2.7b")
    parser.add_argument("--height", type=int, default=480, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=832, help="image width, in pixel space")

    ## dust3r
    parser.add_argument('--model_path', type=str, default='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', help='The path of the model')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--niter', default=300)
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--min_conf_thr', default=3.0) # minimum=1.0, maximum=20
    parser.add_argument('--dpt_trd',  type=float, default=1., help='Required for mulitpule reference images and iterative mode, limit the max depth by * dpt_trd')

    ## stream3r
    parser.add_argument('--stream3r_path', default='./checkpoints/STream3R', help='The path of the model')

    ## moge
    parser.add_argument('--moge_path', default='./checkpoints/moge/model.pt', help='The path of the model')
    parser.add_argument('--segnet_path', default='./checkpoints/tracer_b7.pth')

    ## sam
    parser.add_argument('--sam2_ckpt', default='./checkpoints/sam2/sam2_hiera_large.pt', help='The path of the model')
    #parser.add_argument('--sam_model_type', default='vit_h')
    
    ## rose
    parser.add_argument('--rose_ckpt', default='./checkpoints/rose/transformer', help='The path of the model')
    parser.add_argument('--rose_model', default='./checkpoints/rose/Wan2.1-Fun-1.3B-InP', help='The path of the model')
    
    return parser
