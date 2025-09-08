import os
import torch
import sys

import random
import gradio as gr
import random
from configs.infer_config import get_parser
from huggingface_hub import hf_hub_download
from datetime import datetime

# 统一缓存根目录
CACHE_ROOT = "/mnt/data/ywb/cache_ckpts"
os.environ["HF_HOME"] = f"{CACHE_ROOT}/huggingface"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_ROOT}/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_ROOT}/huggingface/datasets"
os.environ["HF_HUB_CACHE"] = f"{CACHE_ROOT}/huggingface/hub"
os.environ["TORCH_HOME"] = f"{CACHE_ROOT}/torch"
os.environ["XDG_CACHE_HOME"] = CACHE_ROOT

traj_examples = [
    ['60; -35; -0.5; 0.5; -0.5'],
    # ['0 -3 -15 -20 -17 -5 0; 0 -2 -5 -10 -8 -5 0 2 5 3 0; 0 0'],
    # ['0 3 10 20 17 10 0; 0 -2 -8 -6 0 2 5 3 0; 0 -0.02 -0.09 -0.16 -0.09 0'],
    # ['0 30; 0 -1 -5 -4 0 1 5 4 0; 0 -0.2'],
]

video_examples = [
    ['test/videos/2.mp4', 1, 5, 1],
    # ['test/videos/0-NNvgaTcVzAG0-r.mp4', 1, 5, 1],
    ['test/videos/9.mp4', 1, 5, 1],
    ['test/videos/p7.mp4', 1, 5, 1],
    ['test/videos/UST-fn-RvhJwMR5S.mp4', 1, 5, 1],
    ['test/videos/1.mp4', 1, 5, 1],
    ['test/videos/7.mp4', 1, 5, 1],
    ['test/videos/ori1.mp4', 1, 5, 1],
    ['test/videos/part-2-3.mp4', 1, 5, 1],
]


img_examples = [
    # ['test/images/qwz.jpg',5,1],
    ['test/images/jacky.jpg',5,1],
    ['test/images/back.jpg',5,1],
    ['test/images/zj.jpg',5,1],
    ['test/images/car.jpeg',5,1],
    ['test/images/fruit.jpg',5,1],
    # ['test/images/room.png',10,1],
    ['test/images/castle.png',-4,1],
    #  ['test/images/dgn.png',5,1],
    # ['test/images/car2.png',15,1],
    ['test/images/flower2.png',10,1],
    ['test/images/vac.png',10,1],
    # ['test/images/real_westlake.jpg',10,1],
    # ['test/images/zelda.jpg',10,1],
    ['test/images/lake.png',5, 0.1],
    # ['test/images/bridge.png',5, 0.1],
    # ['test/images/minecraft.jpg',0,1],
]

max_seed = 2 ** 31


parser = get_parser() # infer_config.py
opts = parser.parse_args() # default device: 'cuda:0'
prefix = datetime.now().strftime("%Y%m%d_%H%M")
opts.save_dir = f'./output/gradio/{prefix}'
os.makedirs(opts.save_dir,exist_ok=True)
test_tensor = torch.Tensor([0]).cuda()
opts.device = str(test_tensor.device)
opts.weight_dtype = torch.bfloat16
opts.blip_path='/mnt/workspace/ywb/cogvideo_ViewCrafter/checkpoints/blip2-opt-2.7b'
opts.transformer_path = '/mnt/workspace/ywb/VideoX-Fun/experiments/good_04gt_block0_openvid_dl3dv_real10k_4-1-1/checkpoint-50_good/transformer'
# opts.transformer_path = '/mnt/workspace/ywb/VideoX-Fun/Wan2.1-VACE-14B-diffusers/transformer'
# opts.transformer_path_2 = '/mnt/workspace/ywb/VideoX-Fun/experiments/resizegt_0.8gtmask_all_dl3dvreal10k_simpleprompt/checkpoint-50/transformer'

opts.model_name='/mnt/workspace/ywb/VideoX-Fun/Wan2.1-VACE-14B-diffusers'
# opts.lora_path='/mnt/workspace/ywb/VideoX-Fun/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors'
opts.lora_path='/mnt/data/ywb/VideoX-Fun/loras/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors'

# from demo_dynamic import UniScene
from demo_dynamic_vipe import UniScene



def show_traj(mode):
    if mode == "左":
        return gr.update(value='80; 0; 0; 0; 0',visible=True),gr.update(visible=False)
    elif mode == "右":
        return gr.update(value='-80; 0; 0; 0; 0',visible=True),gr.update(visible=False)
    elif mode == "上":
        return gr.update(value='0; -50; 0; 0; 0',visible=True),gr.update(visible=False)
    elif mode == "下":
        return gr.update(value='0; 30; 0; 0; 0',visible=True), gr.update(visible=False)
    elif mode == "前":
        return gr.update(value='0; 0; 0; 0; 1.',visible=True), gr.update(visible=False)
    elif mode == "后":
        return gr.update(value='0; 0; 0; 0; -1.',visible=True), gr.update(visible=False)
    elif mode == "360度":
        return gr.update(value='-360; 0; 0; 0; 0',visible=True), gr.update(visible=False)
    elif mode == "自定义":
        return gr.update(value='0; 0; 0; 0; 0',visible=True), gr.update(visible=True)
    elif mode == "swing":
        # 通过在 i2v_pose 中写入标记，让后续函数切换为 swing1 轨迹
        return gr.update(value='swing1',visible=True), gr.update(visible=False)
    elif mode == "重设":
        return gr.update(value='0; 0; 0; 0; 0',visible=False), gr.update(visible=False)

def show_traj_dyn(mode):
    if mode == "左":
        return gr.update(value='80; 0; 0; 0; 0',visible=True),gr.update(visible=False)
    elif mode == "右":
        return gr.update(value='-80; 0; 0; 0; 0',visible=True),gr.update(visible=False)
    elif mode == "上":
        return gr.update(value='0; -40; 0; 0; 0',visible=True),gr.update(visible=False)
    elif mode == "下":
        return gr.update(value='0; 20; 0; 0; 0',visible=True), gr.update(visible=False)
    elif mode == "前":
        return gr.update(value='0; 0; 0; 0; 0.6',visible=True), gr.update(visible=False)
    elif mode == "后":
        return gr.update(value='0; 0; 0; 0; -0.6',visible=True), gr.update(visible=False)
    elif mode == "自定义":
        return gr.update(value='0; 0; 0; 0; 0',visible=True), gr.update(visible=True)
    elif mode == "swing":
        # 动态视角同样用标记值，交给后续解析为 swing1 轨迹
        return gr.update(value='swing1',visible=True), gr.update(visible=False)
    elif mode == "重设":
        return gr.update(value='0; 0; 0; 0; 0',visible=False), gr.update(visible=False)

# def update_source_method(selected):
#     return (
#         gr.update(visible=selected == "单图新视角生成"),
#         gr.update(visible=selected == "多图新视角生成"),
#         gr.update(visible=selected == "视频新视角生成"),
#     )

def update_source_method(selected):
    return (
        gr.update(visible=selected == "图像新视角生成"),
        # gr.update(visible=selected == "多图新视角生成"),
        gr.update(visible=selected == "视频新视角生成"),
    )

def uniscene_demo(opts):
    css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height:576px} #random_button {max-width: 100px !important}"""
    # init model
    image2video = UniScene(opts, gradio = True)

    with gr.Blocks(analytics_enabled=False, css=css) as uniscene_iface:
        gr.Markdown("<div align='center'> <h1> 兔灵View空间智能模型 </span> </h1>")

        # source_method = gr.Radio(
        #     choices=["单图新视角生成", "多图新视角生成", "视频新视角生成"],
        #     value="单图新视角生成",
        #     label="选择输入类型",
        #     interactive=True
        # )

        source_method = gr.Radio(
            choices=["图像新视角生成", "视频新视角生成"],
            value="单图新视角生成",
            label="选择输入类型",
            interactive=True
        )

        with gr.Column(visible=True) as single_view_block:
            with gr.Row():
                with gr.Column():
                    # # step 1: input an image
                    gr.Markdown(
                        "**1.输入图像**",
                        show_label=False, 
                        visible=True
                    )
                    # gr.Markdown("<div align='left' style='font-size:18px;color: #000000'>1. Estimate an elevation angle  that represents the angle at which the image was taken; a value bigger than 0 indicates a top-down view, and it doesn't need to be precise. <br>2. The origin of the world coordinate system is by default defined at the point cloud corresponding to the center pixel of the input image. You can adjust the position of the origin by modifying center_scale; a value smaller than 1 brings the origin closer to you.</div>")

                    with gr.Column():
                        i2v_input_image = gr.Image(label="输入图像",elem_id="input_img")
                        gr.Markdown(
                        "**2.选择相机初始俯仰角和移动半径，一般采用默认值，如果生成视频移动角度过大，可减小相机移动半径**"
                        , 
                        show_label=False, 
                        visible=True
                    )
                        with gr.Row():
                            i2v_elevation = gr.Slider(minimum=-45, maximum=45, step=1, elem_id="elevation", label="相机俯仰角", value=5)
                            i2v_center_scale = gr.Slider(minimum=0.1, maximum=2, step=0.1, elem_id="i2v_center_scale", label="相机移动半径", value=1)
                    with  gr.Column():
                        gr.Markdown(
                            "**3.点击相机运动按钮，可以选择左、右、上、下、前、后移动以及自定义相机运动**"
                            ,show_label=False, 
                            visible=True
                        )
                        with gr.Row():
                            left = gr.Button(value = "左")
                            right = gr.Button(value = "右")
                            up = gr.Button(value = "上")
                        with gr.Row(): 
                            down = gr.Button(value = "下")                              
                            zin = gr.Button(value = "前")
                            zout = gr.Button(value = "后")
                        with gr.Row(): 
                            # circle =  gr.Button(value = "360度")
                            custom = gr.Button(value = "自定义")
                            swing = gr.Button(value = "swing")
                            reset = gr.Button(value = "重设")

                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                i2v_pose = gr.Text(value = '0; 0; 0; 0; 0', label="相机轨迹 (经度角度序列; 纬角度序列; x序列; y序列; z序列)",visible=False)
                                with gr.Column(visible=False) as i2v_egs:
                                    gr.Markdown("<div align='left' style='font-size:18px;color: #000000'>Please refer to the <a href='https://github.com/Drexubery/ViewCrafter/blob/main/docs/gradio_tutorial.md' target='_blank'>tutorial</a> for customizing camera trajectory.</div>")
                                    gr.Examples(examples=traj_examples,
                                            inputs=[i2v_pose],
                                        )     


                # step 3 - Generate video
                with gr.Column():
                    gr.Markdown(
                        "**4.点击生成视频按钮，预计等待70秒**"
                        ,show_label=False, 
                        visible=True
                    )     
                    i2v_output_video = gr.Video(label="左侧:原视频  右侧:新视角视频",elem_id="output_vid",autoplay=True,show_share_button=True)
                    with gr.Row():
                        i2v_steps = gr.Slider(minimum=4, maximum=10, step=1, elem_id="i2v_steps", label="去噪步数", value=8)
                        i2v_seed = gr.Slider(label='随机种子', minimum=0, maximum=max_seed, step=1, value=0)           
                    i2v_end_btn = gr.Button("生成视频")                    
                    # i2v_traj_video = gr.Video(label="Camera Trajectory",elem_id="traj_vid",autoplay=True,show_share_button=True)
                    
            gr.Examples(examples=img_examples,
                inputs=[i2v_input_image,i2v_elevation, i2v_center_scale,],
                # examples_per_page=6
            )            



            i2v_end_btn.click(inputs=[i2v_input_image, i2v_elevation, i2v_center_scale, i2v_pose, i2v_steps, i2v_seed],
                            outputs=[i2v_output_video],
                            fn = image2video.run_single_view_gradio
            )

            left.click(inputs=[left],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj
                        )
            right.click(inputs=[right],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj
                        )
            up.click(inputs=[up],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj
                        )
            down.click(inputs=[down],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj
                        )
            zin.click(inputs=[zin],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj
                        )
            zout.click(inputs=[zout],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj
                        )
            # circle.click(inputs=[circle],
            #             outputs=[i2v_pose,i2v_egs],
            #             fn = show_traj
            #             )
            custom.click(inputs=[custom],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj
                        )
            swing.click(inputs=[swing],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj
                        )
            reset.click(inputs=[reset],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj
                        )

        # with gr.Column(visible=False) as multiview_view_block:
        #     with gr.Row():
        #         with gr.Column():
        #             # # step 1: input an image
        #             gr.Markdown(
        #                 "**1.输入多张图像**",
        #                 show_label=False, 
        #                 visible=True
        #             )
        #             # gr.Markdown("<div align='left' style='font-size:18px;color: #000000'>1. Estimate an elevation angle  that represents the angle at which the image was taken; a value bigger than 0 indicates a top-down view, and it doesn't need to be precise. <br>2. The origin of the world coordinate system is by default defined at the point cloud corresponding to the center pixel of the input image. You can adjust the position of the origin by modifying center_scale; a value smaller than 1 brings the origin closer to you.</div>")

        #             with gr.Column():
        #                 # i2v_input_image = gr.Image(label="输入图像",elem_id="input_img")
        #                 i2v_input_image = gr.File(file_count="multiple", label="输入多张图像", interactive=True)
                        
        #         # step 3 - Generate video
        #         with gr.Column():
        #             gr.Markdown(
        #                 "**2.点击生成视频按钮，预计等待70秒**"
        #                 ,show_label=False, 
        #                 visible=True
        #             )   
        #             i2v_output_video = gr.Video(label="生成视频",elem_id="output_vid",autoplay=True,show_share_button=True)
        #             with gr.Row():
        #                 i2v_steps = gr.Slider(minimum=4, maximum=10, step=1, elem_id="i2v_steps", label="去噪步数", value=4)
        #                 i2v_seed = gr.Slider(label='随机种子', minimum=0, maximum=max_seed, step=1, value=0)             
        #             i2v_end_btn = gr.Button("生成视频")                    
        #             # i2v_traj_video = gr.Video(label="Camera Trajectory",elem_id="traj_vid",autoplay=True,show_share_button=True)
                    
        #     # gr.Examples(examples=img_examples,
        #     #     inputs=[i2v_input_image,i2v_elevation, i2v_center_scale,],
        #     #     # examples_per_page=6
        #     # )   
        #     #          
        #     i2v_end_btn.click(inputs=[i2v_input_image, i2v_steps, i2v_seed],
        #                     outputs=[i2v_output_video],
        #                     fn = image2video.run_sparse_view_gradio
        #     )

        with gr.Column(visible=False) as dynamic_view_block:
            with gr.Row():
                with gr.Column():
                    # # step 1: input an video
                    gr.Markdown(
                        "**1.输入视频**",
                        show_label=False, 
                        visible=True
                    )
                    # gr.Markdown("<div align='left' style='font-size:18px;color: #000000'>1. Estimate an elevation angle  that represents the angle at which the image was taken; a value bigger than 0 indicates a top-down view, and it doesn't need to be precise. <br>2. The origin of the world coordinate system is by default defined at the point cloud corresponding to the center pixel of the input image. You can adjust the position of the origin by modifying center_scale; a value smaller than 1 brings the origin closer to you.</div>")

                    with gr.Column():
                        i2v_input_video = gr.Video(
                            label="输入视频", elem_id="input_video", format="mp4"
                        )

                        gr.Markdown(
                        "**2.选择视频采帧间隔、相机初始俯仰角和相机移动半径；一般采用默认值，如移动角度过大可减小相机移动半径**"
                        , 
                        show_label=False, 
                        visible=True
                    )
                        with gr.Row():
                            i2v_stride = gr.Slider(
                                minimum=1,
                                maximum=5,
                                step=1,
                                elem_id="stride",
                                label="采帧间隔",
                                value=1,
                            )
                            i2v_elevation = gr.Slider(minimum=-45, maximum=45, step=1, elem_id="elevation_dyn", label="相机俯仰角", value=5)
                            i2v_center_scale = gr.Slider(minimum=0.1, maximum=2, step=0.1, elem_id="i2v_center_scale", label="相机移动半径", value=1)
                    with  gr.Column():
                        gr.Markdown(
                            "**3.点击相机运动按钮，可以选择左、右、上、下、前、后移动以及自定义相机运动**"
                            ,show_label=False, 
                            visible=True
                        )
                        with gr.Row():
                            left = gr.Button(value = "左")
                            right = gr.Button(value = "右")
                            up = gr.Button(value = "上")
                        with gr.Row(): 
                            down = gr.Button(value = "下")                              
                            zin = gr.Button(value = "前")
                            zout = gr.Button(value = "后")
                        with gr.Row(): 
                            custom = gr.Button(value = "自定义")
                            swing = gr.Button(value = "swing")
                            reset = gr.Button(value = "重设")

                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                # 与 VIPE 版动态渲染保持一致："经度phi; 纬度theta; x; y; z"
                                i2v_pose = gr.Text(value = '0; 0; 0; 0; 0', label="相机轨迹 (经度; 纬度; x; y; z)",visible=False)
                                with gr.Column(visible=False) as i2v_egs:
                                    gr.Markdown("<div align='left' style='font-size:18px;color: #000000'>Please refer to the <a href='https://github.com/Drexubery/ViewCrafter/blob/main/docs/gradio_tutorial.md' target='_blank'>tutorial</a> for customizing camera trajectory.</div>")
                                    gr.Examples(examples=traj_examples,
                                            inputs=[i2v_pose],
                                        )     

                # step 3 - Generate video
                with gr.Column():
                    gr.Markdown(
                        "**4.点击生成视频按钮，预计等待70秒**"
                        ,show_label=False, 
                        visible=True
                    )    
                    i2v_output_video = gr.Video(label="生成视频",elem_id="output_vid",autoplay=True,show_share_button=True)
                    with gr.Row():
                        i2v_steps = gr.Slider(minimum=4, maximum=10, step=1, elem_id="i2v_steps", label="去噪步数", value=8)
                        i2v_seed = gr.Slider(label='随机种子', minimum=0, maximum=max_seed, step=1, value=0)            
                    i2v_end_btn = gr.Button("生成视频")                    
                    # i2v_traj_video = gr.Video(label="Camera Trajectory",elem_id="traj_vid",autoplay=True,show_share_button=True)
                           
            gr.Examples(
                examples=video_examples,
                inputs=[
                    i2v_input_video,
                    i2v_stride,
                    i2v_elevation,
                    i2v_center_scale,
                ],
            )    

            i2v_end_btn.click(inputs=[i2v_input_video, i2v_stride, i2v_elevation, i2v_center_scale, i2v_pose, i2v_steps, i2v_seed],
                            outputs=[i2v_output_video],
                            fn = image2video.run_dynamic_view_gradio
            )

            left.click(inputs=[left],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj_dyn
                        )
            right.click(inputs=[right],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj_dyn
                        )
            up.click(inputs=[up],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj_dyn
                        )
            down.click(inputs=[down],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj_dyn
                        )
            zin.click(inputs=[zin],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj_dyn
                        )
            zout.click(inputs=[zout],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj_dyn
                        )
            custom.click(inputs=[custom],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj_dyn
                        )
            swing.click(inputs=[swing],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj_dyn
                        )
            reset.click(inputs=[reset],
                        outputs=[i2v_pose,i2v_egs],
                        fn = show_traj_dyn
                        )

        # # === 绑定切换逻辑 ===
        # source_method.change(
        #     fn=update_source_method,
        #     inputs=source_method,
        #     outputs=[single_view_block, multiview_view_block, dynamic_view_block]
        # )

        # === 绑定切换逻辑 ===
        source_method.change(
            fn=update_source_method,
            inputs=source_method,
            outputs=[single_view_block, dynamic_view_block]
        )

    return uniscene_iface


uniscene_iface = uniscene_demo(opts)
uniscene_iface.queue(max_size=10)
uniscene_iface.launch(debug=True) #fixme

# os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
# server_name = os.getenv("SERVER_NAME", "0.0.0.0")
# server_port = int(os.getenv("SERVER_PORT", "7860"))
# uniscene_iface = uniscene_demo(opts)
# uniscene_iface.launch(server_name=server_name, server_port=server_port, share=True)
