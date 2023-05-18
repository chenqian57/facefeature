import os.path as osp


class CallBackSaveLog(object):
    def __init__(self, save_dir, val_targets) -> None:
        val_targets = ["val/" + val for val in val_targets]
        self.save_dir = save_dir
        self.keys = [
            "epoch",
            "img_size",
            "lr",
            "loss",
        ] + val_targets

    def __call__(self, vals):
        if not osp.exists(self.save_dir):
            print(f"{self.save_dir} is not existed, skip!")
            return

        file = osp.join(self.save_dir, "results.csv")
        n = len(vals)  # number of cols
        s = (
            ""
            if osp.exists(file)
            else (("%20s," * n % tuple(self.keys)).rstrip(",") + "\n")
        )  # add header
        with open(file, "a") as f:
            f.write(s + ("%20.5g," * n % tuple(vals)).rstrip(",") + "\n")
