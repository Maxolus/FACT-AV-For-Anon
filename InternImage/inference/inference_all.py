import os

def main(): 
    os.system("python video_demo.py inputs/Ueberland.mp4 configs/cityscapes/upernet_internimage_t_512x1024_160k_cityscapes.py ckpts/upernet_internimage_t_512x1024_160k_cityscapes.pth --palette cityscapes --output-file inputs/UeberlandInternImage-82_58.avi")

    os.system("python video_demo.py inputs/Ueberland.mp4 configs/cityscapes/segformer_internimage_l_512x1024_160k_mapillary2cityscapes.py ckpts/segformer_internimage_l_512x1024_160k_mapillary2cityscapes.pth --palette cityscapes --output-file inputs/UeberlandInternImage-83_68.avi")

    os.system("python video_demo.py inputs/Ueberland.mp4 configs/cityscapes/upernet_internimage_l_512x1024_160k_cityscapes.py ckpts/upernet_internimage_l_512x1024_160k_cityscapes.pth --palette cityscapes --output-file inputs/UeberlandInternImage-85_16.avi")

    os.system("python video_demo.py inputs/Ueberland.mp4 configs/cityscapes/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py ckpts/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.pth --palette cityscapes --output-file inputs/UeberlandInternImage-86_20.avi")

    os.system("python video_demo.py inputs/3Spurig.mp4 configs/cityscapes/upernet_internimage_t_512x1024_160k_cityscapes.py ckpts/upernet_internimage_t_512x1024_160k_cityscapes.pth --palette cityscapes --output-file inputs/3SpurigInternImage-82_58.avi")

    os.system("python video_demo.py inputs/3Spurig.mp4 configs/cityscapes/segformer_internimage_l_512x1024_160k_mapillary2cityscapes.py ckpts/segformer_internimage_l_512x1024_160k_mapillary2cityscapes.pth --palette cityscapes --output-file inputs/3SpurigInternImage-83_68.avi")

    os.system("python video_demo.py inputs/3Spurig.mp4 configs/cityscapes/upernet_internimage_l_512x1024_160k_cityscapes.py ckpts/upernet_internimage_l_512x1024_160k_cityscapes.pth --palette cityscapes --output-file inputs/3SpurigInternImage-85_16.avi")

    os.system("python video_demo.py inputs/3Spurig.mp4 configs/cityscapes/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py ckpts/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.pth --palette cityscapes --output-file inputs/3SpurigInternImage-86_20.avi")

    os.system("python video_demo.py inputs/Spielstrasse.mp4 configs/cityscapes/upernet_internimage_t_512x1024_160k_cityscapes.py ckpts/upernet_internimage_t_512x1024_160k_cityscapes.pth --palette cityscapes --output-file inputs/SpielstrasseInternImage-82_58.avi")

    os.system("python video_demo.py inputs/Spielstrasse.mp4 configs/cityscapes/segformer_internimage_l_512x1024_160k_mapillary2cityscapes.py ckpts/segformer_internimage_l_512x1024_160k_mapillary2cityscapes.pth --palette cityscapes --output-file inputs/SpielstrasseInternImage-83_68.avi")

    os.system("python video_demo.py inputs/Spielstrasse.mp4 configs/cityscapes/upernet_internimage_l_512x1024_160k_cityscapes.py ckpts/upernet_internimage_l_512x1024_160k_cityscapes.pth --palette cityscapes --output-file inputs/SpielstrasseInternImage-85_16.avi")

    os.system("python video_demo.py inputs/Spielstrasse.mp4 configs/cityscapes/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py ckpts/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.pth --palette cityscapes --output-file inputs/SpielstrasseInternImage-86_20.avi")

if __name__ == '__main__':
    main()