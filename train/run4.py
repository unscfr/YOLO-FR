from ultralytics import YOLO, RTDETR

if __name__ == '__main__':
    model = YOLO(r'/data/FR/FRYOLO/train/yolo11/yolo11-2-4-C3k2_WT-72023-VCMWT-6825-C3k2_ECA-19XSSwols-3D-seg.yaml')
    #model = RTDETR(r'/data/FR/FRYOLO/ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml')
    model.train(#data=r"/data/FR/FRYOLO/ultralytics/cfg/datasets/VisDrone.yaml",
                #data=r"/home/nxnu/Documents/数据集3713/data.yaml",
                data=r"/data/FR/FRYOLO/ultralytics/cfg/datasets/coco128-seg.yaml",
                #cache='disk',
                imgsz=640,
                epochs=10,
                batch=4,
                workers=8,
                device='1',
                optimizer='SGD',  # using SGD 优化器 默认为auto建议大家使用固定的.
                resume=True,
                project='runseg',
                name='seg-yolofr-10',
                pretrained=False,
                )
'''Validating runseg/seg-yolo11-102/weights/best.pt...
Ultralytics 8.3.78 🚀 Python-3.12.7 torch-2.5.0+cu121 CUDA:1 (NVIDIA RTX A6000, 48670MiB)
YOLO11-seg summary (fused): 113 layers, 2,868,664 parameters, 0 gradients, 10.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 2/2 [00:00<00:00,  6.61it/s]
                   all        128        929          0          0          0          0          0          0          0          0
Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 0.1ms postprocess per image
FPS(include:preprocess,inference,postprocess):             1143.91            
FPS(inference):1346.30
Results saved to runseg/seg-yolo11-102'''