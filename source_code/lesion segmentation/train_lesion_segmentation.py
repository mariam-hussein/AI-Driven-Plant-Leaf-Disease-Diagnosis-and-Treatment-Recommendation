from rfdetr import RFDETRSegNano

model = RFDETRSegNano()
model.train(
    dataset_dir="/content/lession-1",
    epochs=20,
    batch_size=32,early_stopping=True, early_stopping_patience=4,
    grad_accum_steps=8, resolution=312 ,
    lr=1e-4,run_test=False,    output_dir="/content/drive/MyDrive/MARIAM_PLANET/train/RF_DETR/Lesion1"
)