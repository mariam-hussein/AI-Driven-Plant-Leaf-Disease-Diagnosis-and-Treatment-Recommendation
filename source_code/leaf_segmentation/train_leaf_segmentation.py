from rfdetr import RFDETRSegNano

model = RFDETRSegNano()
model.train(
    dataset_dir="/content/dataset_312",
    epochs=20,
    batch_size=16,early_stopping=True, early_stopping_patience=3,
    grad_accum_steps=8, resolution=312 ,pretrain_weights="/content/drive/MyDrive/MARIAM_PLANET/train/RF_DETR/Leave_nano/checkpoint_best_total.pth",
    lr=1e-4,run_test=False,    output_dir="/content/drive/MyDrive/MARIAM_PLANET/train/RF_DETR/Leave_big_nano"
)