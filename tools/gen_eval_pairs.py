from metric_trainer.utils.pair import gen_pair

if __name__ == "__main__":
    gen_pair(
        pic_root="/dataset/dataset/face_test",
        save_path="pair.txt",
        sim_ratio=0.5,
        total_num=10000,
        interval=500,
    )
