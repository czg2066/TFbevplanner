from moviepy.editor import ImageSequenceClip
import os

# 图片文件夹路径
image_folder = '/media/zr/vec_env/7vtfpp/carla_garage/results/enh2_epoch20/longest6_route1_05_08_12_06_30/'
# 输出视频的文件名
output_path = image_folder+'video_quality15.mp4'

# 按文件名排序的图片列表（保持顺序）
# images = sorted([
#     os.path.join(image_folder, img) 
#     for img in os.listdir(image_folder) 
#     if img.endswith(".jpg")
# ])  # 按修改时间排序（可选）
images = sorted(
    [
        os.path.join(image_folder, img)
        for img in os.listdir(image_folder)
        if img.endswith(".png") #and 10000 < int(os.path.splitext(os.path.basename(img))[0]) < 20000
    ],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)  # 按文件名排序（可选）
images.append(images[-1])  # 添加最后一帧作为结尾帧
# 生成视频（支持自定义fps和时长）
fps = 20  # 每秒帧数
clip = ImageSequenceClip(images, fps=fps)
clip.write_videofile(output_path)
print("视频生成完成！")