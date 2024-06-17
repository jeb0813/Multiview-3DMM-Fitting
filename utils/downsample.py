import os
# import ipdb

if __name__=="__main__":
    target_path='/data/chenziang/codes/Multiview-3DMM-Fitting/MEAD/M003_25'
    audio_path=os.path.join(target_path,'audio')
    video_path=os.path.join(target_path,'video')


    # # ipdb.set_trace()
    # for emo in sorted(os.listdir(audio_path)):
    #     # print("Processing %s %s" % (id, emo))
    #     audio_emo_path = os.path.join(audio_path, emo)

    #     # 遍历level
    #     for level in sorted(os.listdir(audio_emo_path)):
    #         audio_level_path = os.path.join(audio_emo_path, level)

    #         for audio in sorted(os.listdir(audio_level_path)):
    #             # if audio.split('.')[-1] != 'm4a':
    #             #     # 删除
    #             #     os.remove(os.path.join(audio_level_path, audio))
    #             #     continue
    #             _audio_path = os.path.join(audio_level_path, audio)
    #             # ffmpeg降采样到16kHz
    #             os.system("ffmpeg -i %s -y -ar 16000 -ac 1 %s" % (_audio_path, _audio_path.split('.')[0] + '_16k.wav'))

    front_view=os.path.join(video_path,'front')
    for emo in sorted(os.listdir(front_view)):
        video_emo_path = os.path.join(front_view, emo)

        for level in sorted(os.listdir(video_emo_path)):
            video_level_path = os.path.join(video_emo_path, level)

            for video in sorted(os.listdir(video_level_path)):
                # if video.split('.')[0].split('_')[-1] == '25fps':
                #     os.remove(os.path.join(video_level_path, video))
                #     continue
                _video_path = os.path.join(video_level_path, video)
                # 30fps降采样到25fps
                os.system("ffmpeg -i %s -y -filter:v \"fps=25\" -crf 10 %s" % (_video_path, _video_path.split('.')[0] + '_25fps.mp4'))