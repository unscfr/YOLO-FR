from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'/data/FR/FRYOLO/train/yolo11/yolo11-2-4-C3k2_WT-72023-VCM-6825-C3k2_ECA-19XSSwols-3D.yaml')
    model.train(#data=r"/data/FR/FRYOLO/ultralytics/cfg/datasets/VisDrone.yaml",
                #data=r"/data/FR/FRYOLO/ultralytics/cfg/datasets/AI-TOD.yaml",
                data=r"/data/FR/FRYOLO/ultralytics/cfg/datasets/hazydet.yaml",
                cache=False,
                imgsz=640,
                epochs=500,
                batch=16,
                workers=16,
                device='0',
                optimizer='SGD',  # using SGD 优化器 默认为auto建议大家使用固定的.
                resume=True,
                project='Revision2/HazyDet',
                name='HazyDet-YOLO-FR-500',
                pretrained=False,
                amp=True,
                )