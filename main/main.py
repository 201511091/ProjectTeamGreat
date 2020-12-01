if __name__ == "__main__":
    import torch
    from trainer.train_utils import generate_lyrics

    # 모델 불러오기
    pretrained_model = "../data/model/lyrics_generator_2020-12-01 12:17:46.pt"
    model = torch.load(pretrained_model)

    # 가사 생성하기
    lyrics = generate_lyrics(["사랑"])
    print(lyrics)