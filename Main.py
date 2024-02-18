from Train import train, eval


def main(model_config = None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 200,
        "batch_size": 20,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 64,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./test/CheckPoints_cluster_T1000_epoch1000_imgsize64_label/",
        "test_load_weight": "ckpt_999_.pt",
        "sampled_dir": "./test/CheckPoints_cluster_T1000_epoch1000_imgsize64_label/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs4.png",
        "sampledImgName": "SampledNoGuidenceImgs4.png",
        "nrow": 8,
        "data_dir": '/home/chase/shy/DenoisingDiffusionProbabilityModel-ddpm-/biotite_class',
        "num_labels":5,
        "w": 0.5
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()