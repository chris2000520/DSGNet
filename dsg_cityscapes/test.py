from models.cgfnet import CGFNet
import torch as t




if __name__ == "__main__":
   
    print('-----' * 5)

    rgb = t.randn(1, 3, 512, 512)
    net = CGFNet(11)
   
    net.eval()
    out = net(rgb)
    print(out.shape)

    # 使用cityscapes预训练权重训练camvid数据集时需要删掉分割头和辅助头的参数，其余部分不变
    # checkpoint = t.load('./save/dsg_cityscapes_75.19_best.pth') 
    # checkpoint['state_dict'].pop('seg_head.1.weight')
    # checkpoint['state_dict'].pop('aux_head.1.weight')    
    # t.save(checkpoint, './save/dsg_cityscapes_75.19_best.pth')